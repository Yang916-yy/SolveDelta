#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <tuple>

namespace {

constexpr int kRank = 128;
constexpr int kChunk = 16;
constexpr int kComponents = 4;
constexpr int kThreads = 512;
constexpr int kWarps = 16;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float prefix_exclusive(float value, float carry) {
    const int lane = threadIdx.x & 31;
    float total = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float other = __shfl_up_sync(0xffffffffu, total, offset);
        if (lane >= offset) total += other;
    }
    return carry + total - value;
}

__device__ __forceinline__ float suffix_exclusive(float value, float carry) {
    const int lane = threadIdx.x & 31;
    float total = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float other = __shfl_down_sync(0xffffffffu, total, offset);
        if (lane + offset < 32) total += other;
    }
    return carry + total - value;
}

template <bool Upper>
__device__ __forceinline__ void rank1_actions(
    const float* __restrict__ left,
    const float* __restrict__ right,
    const float* __restrict__ source_u,
    const float* __restrict__ source_h,
    float (&action_u)[4],
    float (&action_h)[4],
    float (&transpose_u)[4]) {
    const int lane = threadIdx.x & 31;
    float carry_u = 0.0f;
    float carry_h = 0.0f;
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int block = Upper ? 3 - pass : pass;
        const int coordinate = block * 32 + lane;
        const float rv = right[coordinate];
        const float product_u = rv * source_u[coordinate];
        const float product_h = rv * source_h[coordinate];
        const float exclusive_u = Upper
            ? suffix_exclusive(product_u, carry_u)
            : prefix_exclusive(product_u, carry_u);
        const float exclusive_h = Upper
            ? suffix_exclusive(product_h, carry_h)
            : prefix_exclusive(product_h, carry_h);
        action_u[block] = left[coordinate] * exclusive_u;
        action_h[block] = left[coordinate] * exclusive_h;
        const int terminal = Upper ? 0 : 31;
        const float inclusive_u = exclusive_u - carry_u + product_u;
        const float inclusive_h = exclusive_h - carry_h + product_h;
        carry_u += __shfl_sync(0xffffffffu, inclusive_u, terminal);
        carry_h += __shfl_sync(0xffffffffu, inclusive_h, terminal);
    }
    float carry_t = 0.0f;
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int block = Upper ? pass : 3 - pass;
        const int coordinate = block * 32 + lane;
        const float product = left[coordinate] * source_u[coordinate];
        const float exclusive = Upper
            ? prefix_exclusive(product, carry_t)
            : suffix_exclusive(product, carry_t);
        transpose_u[block] = right[coordinate] * exclusive;
        const int terminal = Upper ? 31 : 0;
        const float inclusive = exclusive - carry_t + product;
        carry_t += __shfl_sync(0xffffffffu, inclusive, terminal);
    }
}

template <bool Upper, bool UseH>
__device__ __forceinline__ void dense_actions(
    const float* __restrict__ matrix,
    const float* __restrict__ source_u,
    const float* __restrict__ source_h,
    float* __restrict__ workspace,
    float (&action)[4],
    float (&transpose_u)[4]) {
    const int lane = threadIdx.x & 31;
    const float* rhs = UseH ? source_h : source_u;
#pragma unroll 1
    for (int coordinate = 0; coordinate < kRank; ++coordinate) {
        float row_value = 0.0f;
        float transpose_value = 0.0f;
        if constexpr (Upper) {
            for (int column = coordinate + 1 + lane; column < kRank; column += 32) {
                row_value = fmaf(
                    matrix[coordinate * kRank + column], rhs[column], row_value);
            }
            for (int row = lane; row < coordinate; row += 32) {
                transpose_value = fmaf(
                    matrix[row * kRank + coordinate], source_u[row], transpose_value);
            }
        } else {
            for (int column = lane; column < coordinate; column += 32) {
                row_value = fmaf(
                    matrix[coordinate * kRank + column], rhs[column], row_value);
            }
            for (int row = coordinate + 1 + lane; row < kRank; row += 32) {
                transpose_value = fmaf(
                    matrix[row * kRank + coordinate], source_u[row], transpose_value);
            }
        }
        row_value = warp_sum(row_value);
        transpose_value = warp_sum(transpose_value);
        if (lane == 0) {
            workspace[coordinate] = row_value;
            workspace[kRank + coordinate] = transpose_value;
        }
    }
    __syncwarp();
#pragma unroll
    for (int block = 0; block < 4; ++block) {
        const int coordinate = block * 32 + lane;
        action[block] = workspace[coordinate];
        transpose_u[block] = workspace[kRank + coordinate];
    }
}

__launch_bounds__(256, 2)
__global__ void correction_coefficients_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ weights,
    const float* __restrict__ eta,
    float* __restrict__ c0,
    float* __restrict__ c1,
    float* __restrict__ c2,
    const int panels) {
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    const int item = threadIdx.x;
    if (item < kComponents) {
        float value = 0.0f;
        for (int target = 0; target < kChunk; ++target) {
            const float a = alpha[panel * kChunk + target];
            value = fmaf(-eta[(panel * kChunk + target) * 4 + item], a * a, value);
        }
        c0[panel * 4 + item] = value;
    }
    if (item < kChunk * kComponents) {
        const int source = item / kComponents;
        const int component = item % kComponents;
        float value = 0.0f;
        for (int target = source; target < kChunk; ++target) {
            const float weight = weights[(panel * kChunk + source) * kChunk + target];
            const float a = alpha[panel * kChunk + target];
            value = fmaf(
                -eta[(panel * kChunk + target) * 4 + component],
                weight * a, value);
        }
        c1[(panel * kChunk + source) * 4 + component] = value;
    }
    if (item < kChunk * kChunk) {
        const int source = item / kChunk;
        const int generator = item % kChunk;
        for (int component = 0; component < kComponents; ++component) {
            float value = 0.0f;
            const int begin = source > generator ? source : generator;
            for (int target = begin; target < kChunk; ++target) {
                const float ws = weights[
                    (panel * kChunk + source) * kChunk + target];
                const float wg = weights[
                    (panel * kChunk + generator) * kChunk + target];
                value = fmaf(
                    -eta[(panel * kChunk + target) * 4 + component],
                    ws * wg, value);
            }
            c2[((panel * kChunk + source) * kChunk + generator) * 4 + component]
                = value;
        }
    }
}

__launch_bounds__(kThreads, 1)
__global__ void correction_boundary_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ alpha,
    const float* __restrict__ diagonal_h,
    const float* __restrict__ diagonal_r,
    const float* __restrict__ c0,
    const float* __restrict__ c1,
    float* __restrict__ grad_boundary_h,
    float* __restrict__ grad_boundary_r,
    float* __restrict__ boundary_gram,
    const int panels) {
    __shared__ float reduction[kComponents][kWarps];
    const int panel = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (panel >= panels) return;
    const int matrix_base = panel * kRank * kRank;
    float gram[kComponents] = {};
    for (int entry = threadIdx.x; entry < kRank * kRank; entry += kThreads) {
        const int row = entry / kRank;
        const int col = entry % kRank;
        const float bh = boundary_h[matrix_base + entry];
        const float br = boundary_r[matrix_base + entry];
        float gh = 0.0f;
        float gr = 0.0f;
        int h_component = -1;
        int r_component = -1;
        if (row > col) {
            h_component = 0;
            r_component = 1;
        } else if (row < col) {
            h_component = 2;
            r_component = 3;
        }
        if (h_component >= 0) {
            gh = c0[panel * 4 + h_component] * bh;
            gr = c0[panel * 4 + r_component] * br;
            for (int source = 0; source < kChunk; ++source) {
                const int row_index = (panel * kChunk + source) * kRank + row;
                const int col_index = (panel * kChunk + source) * kRank + col;
                const float ur = u[row_index];
                gh = fmaf(
                    c1[(panel * kChunk + source) * 4 + h_component],
                    ur * u[col_index], gh);
                gr = fmaf(
                    c1[(panel * kChunk + source) * 4 + r_component],
                    ur * h[col_index], gr);
            }
            gram[h_component] = fmaf(bh, bh, gram[h_component]);
            gram[r_component] = fmaf(br, br, gram[r_component]);
        } else {
            for (int target = 0; target < kChunk; ++target) {
                const int vector = (panel * kChunk + target) * kRank + row;
                const float a = alpha[panel * kChunk + target];
                gh = fmaf(a, diagonal_h[vector], gh);
                gr = fmaf(a, diagonal_r[vector], gr);
            }
        }
        grad_boundary_h[matrix_base + entry] = gh;
        grad_boundary_r[matrix_base + entry] = gr;
    }
#pragma unroll
    for (int component = 0; component < kComponents; ++component) {
        const float sum = warp_sum(gram[component]);
        if (lane == 0) reduction[component][warp] = sum;
    }
    __syncthreads();
    if (threadIdx.x < kComponents) {
        float sum = 0.0f;
#pragma unroll
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            sum += reduction[threadIdx.x][warp_index];
        }
        boundary_gram[panel * 4 + threadIdx.x] = sum;
    }
}

__launch_bounds__(256, 2)
__global__ void boundary_actions_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ u,
    const float* __restrict__ h,
    float* __restrict__ actions,
    const int panels) {
    __shared__ float bh_tile[16][16];
    __shared__ float bht_tile[16][16];
    __shared__ float br_tile[16][16];
    __shared__ float brt_tile[16][16];
    __shared__ float u_tile[16][16];
    __shared__ float h_tile[16][16];
    const int row_tile = blockIdx.x;
    const int panel = blockIdx.y;
    const int side = blockIdx.z;
    const int local_row = threadIdx.x >> 4;
    const int source = threadIdx.x & 15;
    const int row = row_tile * 16 + local_row;
    if (panel >= panels) return;
    float bh_u = 0.0f;
    float bht_u = 0.0f;
    float br_h = 0.0f;
    float brt_u = 0.0f;
    const int matrix_base = panel * kRank * kRank;
    for (int k_block = 0; k_block < 8; ++k_block) {
        const int k = k_block * 16 + source;
        const int rhs_k = k_block * 16 + local_row;
        const bool direct_mask = side == 0 ? row > k : row < k;
        const bool transpose_mask = side == 0 ? k > row : k < row;
        bh_tile[local_row][source] = direct_mask
            ? boundary_h[matrix_base + row * kRank + k] : 0.0f;
        br_tile[local_row][source] = direct_mask
            ? boundary_r[matrix_base + row * kRank + k] : 0.0f;
        bht_tile[local_row][source] = transpose_mask
            ? boundary_h[matrix_base + k * kRank + row] : 0.0f;
        brt_tile[local_row][source] = transpose_mask
            ? boundary_r[matrix_base + k * kRank + row] : 0.0f;
        u_tile[local_row][source] =
            u[(panel * kChunk + source) * kRank + rhs_k];
        h_tile[local_row][source] =
            h[(panel * kChunk + source) * kRank + rhs_k];
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < 16; ++inner) {
            bh_u = fmaf(bh_tile[local_row][inner], u_tile[inner][source], bh_u);
            bht_u = fmaf(bht_tile[local_row][inner], u_tile[inner][source], bht_u);
            br_h = fmaf(br_tile[local_row][inner], h_tile[inner][source], br_h);
            brt_u = fmaf(brt_tile[local_row][inner], u_tile[inner][source], brt_u);
        }
        __syncthreads();
    }
    const int side_base = side * 4;
    const int vector = ((panel * 8 + side_base) * kChunk + source) * kRank + row;
    const int stride = kChunk * kRank;
    actions[vector + 0 * stride] = bh_u;
    actions[vector + 1 * stride] = bht_u;
    actions[vector + 2 * stride] = br_h;
    actions[vector + 3 * stride] = brt_u;
}

__launch_bounds__(kThreads, 1)
__global__ void correction_source_kernel(
    const float* __restrict__ boundary_actions,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ c1,
    const float* __restrict__ c2,
    float* __restrict__ grad_u,
    float* __restrict__ grad_h,
    float* __restrict__ boundary_local_gram,
    float* __restrict__ local_gram,
    const int panels) {
    const int panel = blockIdx.x;
    const int source = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (panel >= panels) return;
    const int source_base = (panel * kChunk + source) * kRank;
    float source_u[4], source_h[4], gu[4] = {}, gh[4] = {};
#pragma unroll
    for (int block = 0; block < 4; ++block) {
        const int coordinate = block * 32 + lane;
        source_u[block] = u[source_base + coordinate];
        source_h[block] = h[source_base + coordinate];
    }
    for (int side = 0; side < 2; ++side) {
        float bh_action[4], bh_transpose[4];
        float br_action[4], br_transpose[4];
#pragma unroll
        for (int block = 0; block < 4; ++block) {
            const int coordinate = block * 32 + lane;
            const int base = ((panel * 8 + side * 4) * kChunk + source) * kRank
                + coordinate;
            const int stride = kChunk * kRank;
            bh_action[block] = boundary_actions[base + 0 * stride];
            bh_transpose[block] = boundary_actions[base + 1 * stride];
            br_action[block] = boundary_actions[base + 2 * stride];
            br_transpose[block] = boundary_actions[base + 3 * stride];
        }
        const int hc = side == 0 ? 0 : 2;
        const int rc = hc + 1;
        const float cbh = c1[(panel * kChunk + source) * 4 + hc];
        const float cbr = c1[(panel * kChunk + source) * 4 + rc];
        float blh = 0.0f;
        float blr = 0.0f;
#pragma unroll
        for (int block = 0; block < 4; ++block) {
            gu[block] += cbh * (bh_action[block] + bh_transpose[block])
                + cbr * br_action[block];
            gh[block] += cbr * br_transpose[block];
            blh = fmaf(source_u[block], bh_action[block], blh);
            blr = fmaf(source_u[block], br_action[block], blr);
        }
        blh = warp_sum(blh);
        blr = warp_sum(blr);
        if (lane == 0) {
            boundary_local_gram[(panel * kChunk + source) * 4 + hc] = blh;
            boundary_local_gram[(panel * kChunk + source) * 4 + rc] = blr;
        }
        for (int generator = 0; generator < kChunk; ++generator) {
            const int generator_base = (panel * kChunk + generator) * kRank;
            float hu[4], hh[4], htu[4];
            float ru[4], rh[4], rtu[4];
            if (side == 0) {
                rank1_actions<false>(
                    u + generator_base, u + generator_base,
                    u + source_base, h + source_base, hu, hh, htu);
                rank1_actions<false>(
                    u + generator_base, h + generator_base,
                    u + source_base, h + source_base, ru, rh, rtu);
            } else {
                rank1_actions<true>(
                    u + generator_base, u + generator_base,
                    u + source_base, h + source_base, hu, hh, htu);
                rank1_actions<true>(
                    u + generator_base, h + generator_base,
                    u + source_base, h + source_base, ru, rh, rtu);
            }
            const float clh = c2[
                ((panel * kChunk + source) * kChunk + generator) * 4 + hc];
            const float clr = c2[
                ((panel * kChunk + source) * kChunk + generator) * 4 + rc];
            float llh = 0.0f;
            float llr = 0.0f;
#pragma unroll
            for (int block = 0; block < 4; ++block) {
                gu[block] += clh * (hu[block] + htu[block]) + clr * rh[block];
                gh[block] += clr * rtu[block];
                llh = fmaf(source_u[block], hu[block], llh);
                llr = fmaf(source_u[block], rh[block], llr);
            }
            llh = warp_sum(llh);
            llr = warp_sum(llr);
            if (lane == 0) {
                const int gram_base =
                    ((panel * kChunk + generator) * kChunk + source) * 4;
                local_gram[gram_base + hc] = llh;
                local_gram[gram_base + rc] = llr;
            }
        }
    }
#pragma unroll
    for (int block = 0; block < 4; ++block) {
        const int coordinate = block * 32 + lane;
        grad_u[source_base + coordinate] = gu[block];
        grad_h[source_base + coordinate] = gh[block];
    }
}

__launch_bounds__(kThreads, 1)
__global__ void correction_scalar_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ alpha,
    const float* __restrict__ weights,
    const float* __restrict__ eta,
    const float* __restrict__ diagonal_h,
    const float* __restrict__ diagonal_r,
    const float* __restrict__ boundary_gram,
    const float* __restrict__ boundary_local_gram,
    const float* __restrict__ local_gram,
    float* __restrict__ grad_u,
    float* __restrict__ grad_h,
    float* __restrict__ grad_alpha,
    float* __restrict__ grad_weights,
    const int panels) {
    const int panel = blockIdx.x;
    const int target = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (panel >= panels) return;
    const int scalar = panel * kChunk + target;
    float ga = 0.0f;
#pragma unroll
    for (int component = 0; component < kComponents; ++component) {
        float inner = alpha[scalar] * boundary_gram[panel * 4 + component];
        for (int generator = 0; generator <= target; ++generator) {
            inner = fmaf(
                weights[(panel * kChunk + generator) * kChunk + target],
                boundary_local_gram[(panel * kChunk + generator) * 4 + component],
                inner);
        }
        ga = fmaf(-eta[scalar * 4 + component], inner, ga);
    }
    float diag_boundary = 0.0f;
    for (int coordinate = lane; coordinate < kRank; coordinate += 32) {
        const int vector = scalar * kRank + coordinate;
        const int diagonal = panel * kRank * kRank + coordinate * (kRank + 1);
        diag_boundary = fmaf(diagonal_h[vector], boundary_h[diagonal], diag_boundary);
        diag_boundary = fmaf(diagonal_r[vector], boundary_r[diagonal], diag_boundary);
    }
    ga += warp_sum(diag_boundary);
    if (lane == 0) grad_alpha[scalar] = ga;

    const int source = target;
    for (int output_target = source; output_target < kChunk; ++output_target) {
        float gw = 0.0f;
        const int output_scalar = panel * kChunk + output_target;
#pragma unroll
        for (int component = 0; component < kComponents; ++component) {
            float inner = alpha[output_scalar]
                * boundary_local_gram[(panel * kChunk + source) * 4 + component];
            for (int generator = 0; generator <= output_target; ++generator) {
                inner = fmaf(
                    weights[(panel * kChunk + generator) * kChunk + output_target],
                    local_gram[
                        ((panel * kChunk + generator) * kChunk + source) * 4 + component],
                    inner);
            }
            gw = fmaf(-eta[output_scalar * 4 + component], inner, gw);
        }
        float diag_local = 0.0f;
        for (int coordinate = lane; coordinate < kRank; coordinate += 32) {
            const int source_vector = (panel * kChunk + source) * kRank + coordinate;
            const int target_vector = output_scalar * kRank + coordinate;
            const float uv = u[source_vector];
            diag_local = fmaf(diagonal_h[target_vector], uv * uv, diag_local);
            diag_local = fmaf(diagonal_r[target_vector], uv * h[source_vector], diag_local);
        }
        gw += warp_sum(diag_local);
        if (lane == 0) {
            grad_weights[(panel * kChunk + source) * kChunk + output_target] = gw;
        }
    }

    float local_diag_h[4] = {};
    float local_diag_r[4] = {};
    for (int output_target = source; output_target < kChunk; ++output_target) {
        const float weight = weights[
            (panel * kChunk + source) * kChunk + output_target];
#pragma unroll
        for (int block = 0; block < 4; ++block) {
            const int coordinate = block * 32 + lane;
            const int vector = (panel * kChunk + output_target) * kRank + coordinate;
            local_diag_h[block] = fmaf(weight, diagonal_h[vector], local_diag_h[block]);
            local_diag_r[block] = fmaf(weight, diagonal_r[vector], local_diag_r[block]);
        }
    }
#pragma unroll
    for (int block = 0; block < 4; ++block) {
        const int coordinate = block * 32 + lane;
        const int vector = (panel * kChunk + source) * kRank + coordinate;
        const float uv = u[vector];
        grad_u[vector] +=
            2.0f * local_diag_h[block] * uv + local_diag_r[block] * h[vector];
        grad_h[vector] += local_diag_r[block] * uv;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor>
radial_correction_cuda(
    const at::Tensor& boundary_h,
    const at::Tensor& boundary_r,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& weights,
    const at::Tensor& alpha,
    const at::Tensor& eta,
    const at::Tensor& diagonal_h,
    const at::Tensor& diagonal_r) {
    const int panels = static_cast<int>(boundary_h.size(0));
    auto c0 = at::empty({panels, 4}, eta.options());
    auto c1 = at::empty({panels, kChunk, 4}, eta.options());
    auto c2 = at::empty({panels, kChunk, kChunk, 4}, eta.options());
    auto grad_boundary_h = at::empty_like(boundary_h);
    auto grad_boundary_r = at::empty_like(boundary_r);
    auto grad_u = at::empty_like(u);
    auto grad_h = at::empty_like(h);
    auto grad_alpha = at::empty_like(alpha);
    auto grad_weights = at::zeros_like(weights);
    auto boundary_gram = at::empty({panels, 4}, eta.options());
    auto boundary_local_gram = at::empty({panels, kChunk, 4}, eta.options());
    auto local_gram = at::empty({panels, kChunk, kChunk, 4}, eta.options());
    auto boundary_actions = at::empty({panels, 8, kChunk, kRank}, u.options());
    c10::cuda::CUDAGuard guard(boundary_h.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    correction_coefficients_kernel<<<panels, 256, 0, stream>>>(
        alpha.data_ptr<float>(), weights.data_ptr<float>(), eta.data_ptr<float>(),
        c0.data_ptr<float>(), c1.data_ptr<float>(), c2.data_ptr<float>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    correction_boundary_kernel<<<panels, kThreads, 0, stream>>>(
        boundary_h.data_ptr<float>(), boundary_r.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), alpha.data_ptr<float>(),
        diagonal_h.data_ptr<float>(), diagonal_r.data_ptr<float>(),
        c0.data_ptr<float>(), c1.data_ptr<float>(),
        grad_boundary_h.data_ptr<float>(), grad_boundary_r.data_ptr<float>(),
        boundary_gram.data_ptr<float>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dim3 action_grid(8, panels, 2);
    boundary_actions_kernel<<<action_grid, 256, 0, stream>>>(
        boundary_h.data_ptr<float>(), boundary_r.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), boundary_actions.data_ptr<float>(),
        panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    correction_source_kernel<<<panels, kThreads, 0, stream>>>(
        boundary_actions.data_ptr<float>(), u.data_ptr<float>(), h.data_ptr<float>(),
        c1.data_ptr<float>(),
        c2.data_ptr<float>(), grad_u.data_ptr<float>(), grad_h.data_ptr<float>(),
        boundary_local_gram.data_ptr<float>(), local_gram.data_ptr<float>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    correction_scalar_kernel<<<panels, kThreads, 0, stream>>>(
        boundary_h.data_ptr<float>(), boundary_r.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), alpha.data_ptr<float>(),
        weights.data_ptr<float>(), eta.data_ptr<float>(),
        diagonal_h.data_ptr<float>(), diagonal_r.data_ptr<float>(),
        boundary_gram.data_ptr<float>(), boundary_local_gram.data_ptr<float>(),
        local_gram.data_ptr<float>(), grad_u.data_ptr<float>(), grad_h.data_ptr<float>(),
        grad_alpha.data_ptr<float>(), grad_weights.data_ptr<float>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_boundary_h, grad_boundary_r, grad_u, grad_h,
            grad_weights, grad_alpha};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "packet_frame_radial_vjp128(Tensor boundary_h, Tensor boundary_r, Tensor u, Tensor h, "
        "Tensor weights, Tensor alpha, Tensor eta, Tensor diagonal_h, "
        "Tensor diagonal_r) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("packet_frame_radial_vjp128", &radial_correction_cuda);
}
