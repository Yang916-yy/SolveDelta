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
constexpr int kPacket = kChunk * kChunk;
constexpr int kThreads = 256;
constexpr int kRhs = 2;
constexpr int kTransposeTile = 16;

__device__ __forceinline__ float reduce4(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 2, 4);
    value += __shfl_down_sync(0xffffffffu, value, 1, 4);
    return value;
}

struct SolveShared {
    float z[kPacket];
    float weights[kPacket];
    float solution[kChunk][kRank];
    float boundary_h[kTransposeTile][kRank];
    float boundary_r[kTransposeTile][kRank];
    float u_coordinate[kChunk];
    float h_coordinate[kChunk];
    float alpha[kChunk];
    float p[kChunk];
    float q[kChunk];
};

template <bool Upper>
__device__ __forceinline__ void transpose_solve_phase(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ diagonal,
    const float* __restrict__ rhs,
    SolveShared& shared,
    const int panel,
    const bool divide_rhs) {
    const int matrix_base = panel * kRank * kRank;
    const int vector_base = panel * kChunk * kRank;
    for (int tile = 0; tile < kRank / kTransposeTile; ++tile) {
        const int tile_base = Upper
            ? tile * kTransposeTile
            : kRank - (tile + 1) * kTransposeTile;
        for (int item = threadIdx.x; item < kRank * kTransposeTile;
             item += blockDim.x) {
            const int row = item / kTransposeTile;
            const int local_coordinate = item % kTransposeTile;
            const int coordinate = tile_base + local_coordinate;
            shared.boundary_h[local_coordinate][row] =
                boundary_h[matrix_base + row * kRank + coordinate];
            shared.boundary_r[local_coordinate][row] =
                boundary_r[matrix_base + row * kRank + coordinate];
        }
        __syncthreads();
#pragma unroll
        for (int local_step = 0; local_step < kTransposeTile; ++local_step) {
            const int local_coordinate = Upper
                ? local_step : kTransposeTile - 1 - local_step;
            const int coordinate = tile_base + local_coordinate;
        if (threadIdx.x < kChunk) {
            shared.u_coordinate[threadIdx.x] =
                u[vector_base + threadIdx.x * kRank + coordinate];
            shared.h_coordinate[threadIdx.x] =
                h[vector_base + threadIdx.x * kRank + coordinate];
        }
        __syncthreads();
        if (threadIdx.x < kChunk * 4) {
            const int target = threadIdx.x >> 2;
            const int lane = threadIdx.x & 3;
            float bh = 0.0f;
            float br = 0.0f;
            if constexpr (Upper) {
                for (int row = lane; row < coordinate; row += 4) {
                    const float value = shared.solution[target][row];
                    bh = fmaf(shared.boundary_h[local_coordinate][row], value, bh);
                    br = fmaf(shared.boundary_r[local_coordinate][row], value, br);
                }
            } else {
                for (int row = coordinate + 1 + lane; row < kRank; row += 4) {
                    const float value = shared.solution[target][row];
                    bh = fmaf(shared.boundary_h[local_coordinate][row], value, bh);
                    br = fmaf(shared.boundary_r[local_coordinate][row], value, br);
                }
            }
            float local = 0.0f;
            for (int source = lane; source <= target; source += 4) {
                const int packet = source * kChunk + target;
                const float factor = shared.p[target] * shared.u_coordinate[source]
                    + shared.q[target] * shared.h_coordinate[source];
                local = fmaf(
                    shared.weights[packet] * factor, shared.z[packet], local);
            }
            bh = reduce4(bh);
            br = reduce4(br);
            local = reduce4(local);
            if (lane == 0) {
                const int index = vector_base + target * kRank + coordinate;
                float value = divide_rhs
                    ? shared.solution[target][coordinate] / diagonal[index]
                    : rhs[index];
                const float boundary = shared.alpha[target]
                    * (shared.p[target] * bh + shared.q[target] * br);
                shared.solution[target][coordinate] = value - boundary - local;
            }
        }
        __syncthreads();
        for (int packet = threadIdx.x; packet < kPacket; packet += kThreads) {
            const int source = packet / kChunk;
            const int target = packet % kChunk;
            if (source <= target) {
                shared.z[packet] = fmaf(
                    shared.u_coordinate[source], shared.solution[target][coordinate],
                    shared.z[packet]);
            }
        }
        __syncthreads();
        }
    }
}

__launch_bounds__(kThreads, 1)
__global__ void blocked_transpose_solve_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ grad_d,
    float* __restrict__ c_upper,
    float* __restrict__ c_lower,
    const int panels) {
    __shared__ SolveShared shared;
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    const int packet_base = panel * kPacket;
    const int scalar_base = panel * kChunk;
    for (int packet = threadIdx.x; packet < kPacket; packet += kThreads) {
        shared.z[packet] = 0.0f;
        shared.weights[packet] = weights[packet_base + packet];
    }
    if (threadIdx.x < kChunk) {
        const int target = threadIdx.x;
        shared.alpha[target] = alpha[scalar_base + target];
        shared.p[target] = coefficient[(scalar_base + target) * 4 + 2];
        shared.q[target] = coefficient[(scalar_base + target) * 4 + 3];
    }
    __syncthreads();
    transpose_solve_phase<true>(
        boundary_h, boundary_r, u, h, diagonal, grad_d,
        shared, panel, false);
    const int vector_base = panel * kChunk * kRank;
    for (int index = threadIdx.x; index < kChunk * kRank; index += kThreads) {
        c_upper[vector_base + index] = shared.solution[index / kRank][index % kRank];
    }
    for (int packet = threadIdx.x; packet < kPacket; packet += kThreads) {
        shared.z[packet] = 0.0f;
    }
    if (threadIdx.x < kChunk) {
        const int target = threadIdx.x;
        shared.p[target] = coefficient[(scalar_base + target) * 4 + 0];
        shared.q[target] = coefficient[(scalar_base + target) * 4 + 1];
    }
    __syncthreads();
    transpose_solve_phase<false>(
        boundary_h, boundary_r, u, h, diagonal, nullptr,
        shared, panel, true);
    for (int index = threadIdx.x; index < kChunk * kRank; index += kThreads) {
        c_lower[vector_base + index] = shared.solution[index / kRank][index % kRank];
    }
}

struct DirectShared {
    float z_u[kRhs][kPacket];
    float z_h[kRhs][kPacket];
    float weights[kPacket];
    float input[kRhs][kChunk][kRank];
    float output[kRhs][kChunk][kRank];
    float boundary_h[kRank];
    float boundary_r[kRank];
    float u_coordinate[kChunk];
    float h_coordinate[kChunk];
    float alpha[kChunk];
    float p[kChunk];
    float q[kChunk];
};

template <bool Upper>
__device__ __forceinline__ void direct_phase(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ u,
    const float* __restrict__ h,
    DirectShared& shared,
    const int panel) {
    const int matrix_base = panel * kRank * kRank;
    const int vector_base = panel * kChunk * kRank;
    for (int step = 0; step < kRank; ++step) {
        const int coordinate = Upper ? kRank - 1 - step : step;
        if (threadIdx.x < kRank) {
            const int col = threadIdx.x;
            shared.boundary_h[col] =
                boundary_h[matrix_base + coordinate * kRank + col];
            shared.boundary_r[col] =
                boundary_r[matrix_base + coordinate * kRank + col];
        }
        if (threadIdx.x < kChunk) {
            shared.u_coordinate[threadIdx.x] =
                u[vector_base + threadIdx.x * kRank + coordinate];
            shared.h_coordinate[threadIdx.x] =
                h[vector_base + threadIdx.x * kRank + coordinate];
        }
        __syncthreads();
        if (threadIdx.x < kChunk * kRhs * 4) {
            const int item = threadIdx.x >> 2;
            const int lane = threadIdx.x & 3;
            const int target = item / kRhs;
            const int rhs = item % kRhs;
            float bh = 0.0f;
            float br = 0.0f;
            if constexpr (Upper) {
                for (int col = coordinate + 1 + lane; col < kRank; col += 4) {
                    const float value = shared.input[rhs][target][col];
                    bh = fmaf(shared.boundary_h[col], value, bh);
                    br = fmaf(shared.boundary_r[col], value, br);
                }
            } else {
                for (int col = lane; col < coordinate; col += 4) {
                    const float value = shared.input[rhs][target][col];
                    bh = fmaf(shared.boundary_h[col], value, bh);
                    br = fmaf(shared.boundary_r[col], value, br);
                }
            }
            float local = 0.0f;
            for (int source = lane; source <= target; source += 4) {
                const int packet = source * kChunk + target;
                const float score = shared.p[target] * shared.z_u[rhs][packet]
                    + shared.q[target] * shared.z_h[rhs][packet];
                local = fmaf(
                    shared.weights[packet] * shared.u_coordinate[source],
                    score, local);
            }
            bh = reduce4(bh);
            br = reduce4(br);
            local = reduce4(local);
            if (lane == 0) {
                const float boundary = shared.alpha[target]
                    * (shared.p[target] * bh + shared.q[target] * br);
                shared.output[rhs][target][coordinate] =
                    shared.input[rhs][target][coordinate] + boundary + local;
            }
        }
        __syncthreads();
        for (int index = threadIdx.x; index < kRhs * kPacket; index += kThreads) {
            const int rhs = index / kPacket;
            const int packet = index % kPacket;
            const int source = packet / kChunk;
            const int target = packet % kChunk;
            if (source <= target) {
                const float value = shared.input[rhs][target][coordinate];
                shared.z_u[rhs][packet] = fmaf(
                    shared.u_coordinate[source], value, shared.z_u[rhs][packet]);
                shared.z_h[rhs][packet] = fmaf(
                    shared.h_coordinate[source], value, shared.z_h[rhs][packet]);
            }
        }
        __syncthreads();
    }
}

__launch_bounds__(kThreads, 1)
__global__ void blocked_direct2_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ dual_lower,
    const float* __restrict__ grad_e,
    const float* __restrict__ grad_chi,
    float* __restrict__ grad_z,
    float* __restrict__ grad_dual_lower,
    float* __restrict__ grad_rhs,
    float* __restrict__ grad_diagonal_dual,
    const int panels) {
    __shared__ DirectShared shared;
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    const int packet_base = panel * kPacket;
    const int scalar_base = panel * kChunk;
    for (int index = threadIdx.x; index < kRhs * kPacket; index += kThreads) {
        shared.z_u[index / kPacket][index % kPacket] = 0.0f;
        shared.z_h[index / kPacket][index % kPacket] = 0.0f;
    }
    for (int packet = threadIdx.x; packet < kPacket; packet += kThreads) {
        shared.weights[packet] = weights[packet_base + packet];
    }
    for (int index = threadIdx.x; index < kChunk * kRank; index += kThreads) {
        shared.input[0][index / kRank][index % kRank] =
            grad_e[panel * kChunk * kRank + index];
        shared.input[1][index / kRank][index % kRank] =
            grad_chi[panel * kChunk * kRank + index];
    }
    if (threadIdx.x < kChunk) {
        const int target = threadIdx.x;
        shared.alpha[target] = alpha[scalar_base + target];
        shared.p[target] = coefficient[(scalar_base + target) * 4 + 2];
        shared.q[target] = coefficient[(scalar_base + target) * 4 + 3];
    }
    __syncthreads();
    direct_phase<true>(boundary_h, boundary_r, u, h, shared, panel);
    for (int index = threadIdx.x; index < kRhs * kChunk * kRank; index += kThreads) {
        const int rhs = index / (kChunk * kRank);
        const int local = index % (kChunk * kRank);
        const int target = local / kRank;
        const int coordinate = local % kRank;
        const int vector_index = panel * kChunk * kRank + local;
        const float gz = shared.output[rhs][target][coordinate];
        grad_z[(panel * kRhs + rhs) * kChunk * kRank + local] = gz;
        const float gl = diagonal[vector_index] * gz;
        shared.input[rhs][target][coordinate] = gl;
        grad_dual_lower[(panel * kRhs + rhs) * kChunk * kRank + local] = gl;
    }
    for (int index = threadIdx.x; index < kRhs * kPacket; index += kThreads) {
        shared.z_u[index / kPacket][index % kPacket] = 0.0f;
        shared.z_h[index / kPacket][index % kPacket] = 0.0f;
    }
    if (threadIdx.x < kChunk) {
        const int target = threadIdx.x;
        shared.p[target] = coefficient[(scalar_base + target) * 4 + 0];
        shared.q[target] = coefficient[(scalar_base + target) * 4 + 1];
    }
    __syncthreads();
    direct_phase<false>(boundary_h, boundary_r, u, h, shared, panel);
    for (int index = threadIdx.x; index < kRhs * kChunk * kRank; index += kThreads) {
        const int rhs = index / (kChunk * kRank);
        const int local = index % (kChunk * kRank);
        grad_rhs[(panel * kRhs + rhs) * kChunk * kRank + local] =
            shared.output[rhs][local / kRank][local % kRank];
    }
    for (int local = threadIdx.x; local < kChunk * kRank; local += kThreads) {
        const int vector_index = panel * kChunk * kRank + local;
        grad_diagonal_dual[vector_index] =
            grad_z[(panel * kRhs + 0) * kChunk * kRank + local]
                * dual_lower[(panel * kChunk * kRhs + (local / kRank) * kRhs + 0)
                    * kRank + local % kRank]
            + grad_z[(panel * kRhs + 1) * kChunk * kRank + local]
                * dual_lower[(panel * kChunk * kRhs + (local / kRank) * kRhs + 1)
                    * kRank + local % kRank];
    }
}

template <bool Upper>
__launch_bounds__(256, 2)
__global__ void direct_boundary_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int panels) {
    __shared__ float bh_tile[16][16];
    __shared__ float br_tile[16][16];
    __shared__ float x_tile[16][16];
    const int row_tile = blockIdx.x;
    const int item_tile = blockIdx.y;
    const int panel = blockIdx.z;
    const int local_row = threadIdx.x >> 4;
    const int local_item = threadIdx.x & 15;
    const int row = row_tile * 16 + local_row;
    const int item = item_tile * 16 + local_item;
    const int target = item >> 1;
    const int rhs = item & 1;
    if (panel >= panels) return;
    float bh = 0.0f;
    float br = 0.0f;
    const int matrix_base = panel * kRank * kRank;
    for (int k_block = 0; k_block < 8; ++k_block) {
        const int k = k_block * 16 + local_item;
        const bool mask = Upper ? row < k : row > k;
        bh_tile[local_row][local_item] = mask
            ? boundary_h[matrix_base + row * kRank + k] : 0.0f;
        br_tile[local_row][local_item] = mask
            ? boundary_r[matrix_base + row * kRank + k] : 0.0f;
        const int rhs_k = k_block * 16 + local_row;
        x_tile[local_row][local_item] = input[
            (panel * kRhs + (local_item & 1)) * kChunk * kRank
            + (item_tile * 8 + (local_item >> 1)) * kRank + rhs_k];
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < 16; ++inner) {
            bh = fmaf(bh_tile[local_row][inner], x_tile[inner][local_item], bh);
            br = fmaf(br_tile[local_row][inner], x_tile[inner][local_item], br);
        }
        __syncthreads();
    }
    const int scalar = panel * kChunk + target;
    const int component = Upper ? 2 : 0;
    const float boundary = alpha[scalar]
        * (coefficient[scalar * 4 + component] * bh
           + coefficient[scalar * 4 + component + 1] * br);
    const int index = (panel * kRhs + rhs) * kChunk * kRank
        + target * kRank + row;
    output[index] = input[index] + boundary;
}

template <bool Upper>
__launch_bounds__(128, 2)
__global__ void direct_local_kernel(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ coefficient,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int panels) {
    const int panel = blockIdx.x;
    const int lane = threadIdx.x & 3;
    const int item = threadIdx.x >> 2;
    const int target = item >> 1;
    const int rhs = item & 1;
    if (panel >= panels) return;
    const int component = Upper ? 2 : 0;
    const int scalar = panel * kChunk + target;
    const float p = coefficient[scalar * 4 + component];
    const float q = coefficient[scalar * 4 + component + 1];
    float zu[4] = {};
    float zh[4] = {};
    for (int step = 0; step < kRank; ++step) {
        const int coordinate = Upper ? kRank - 1 - step : step;
        float local = 0.0f;
        for (int source = lane; source <= target; source += 4) {
            const int slot = source >> 2;
            const int source_index = (panel * kChunk + source) * kRank + coordinate;
            const float score = p * zu[slot] + q * zh[slot];
            local = fmaf(
                weights[(panel * kChunk + source) * kChunk + target]
                    * u[source_index],
                score, local);
            const float value = input[
                (panel * kRhs + rhs) * kChunk * kRank
                + target * kRank + coordinate];
            zu[slot] = fmaf(u[source_index], value, zu[slot]);
            zh[slot] = fmaf(h[source_index], value, zh[slot]);
        }
        local = reduce4(local);
        if (lane == 0) {
            output[(panel * kRhs + rhs) * kChunk * kRank
                + target * kRank + coordinate] += local;
        }
    }
}

__global__ void scale_direct_kernel(
    const float* __restrict__ diagonal,
    const float* __restrict__ input,
    float* __restrict__ grad_z,
    float* __restrict__ grad_dual_lower,
    float* __restrict__ scaled,
    const int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int local = index % (kChunk * kRank);
    const int panel = index / (kRhs * kChunk * kRank);
    const float value = input[index];
    const float result = diagonal[panel * kChunk * kRank + local] * value;
    grad_z[index] = value;
    grad_dual_lower[index] = result;
    scaled[index] = result;
}

__global__ void dual_diagonal_kernel(
    const float* __restrict__ grad_z,
    const float* __restrict__ dual_lower,
    float* __restrict__ grad_diagonal_dual,
    const int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int panel = index / (kChunk * kRank);
    const int local = index % (kChunk * kRank);
    const int target = local / kRank;
    const int coordinate = local % kRank;
    grad_diagonal_dual[index] =
        grad_z[(panel * kRhs + 0) * kChunk * kRank + local]
            * dual_lower[(panel * kChunk * kRhs + target * kRhs + 0)
                * kRank + coordinate]
        + grad_z[(panel * kRhs + 1) * kChunk * kRank + local]
            * dual_lower[(panel * kChunk * kRhs + target * kRhs + 1)
                * kRank + coordinate];
}

__global__ void finalize_kernel(
    const float* __restrict__ c_upper,
    const float* __restrict__ c_lower,
    const float* __restrict__ diagonal,
    const float* __restrict__ y,
    const float* __restrict__ grad_diagonal_dual,
    const float* __restrict__ grad_rhs,
    float* __restrict__ grad_diagonal,
    float* __restrict__ grad_key,
    float* __restrict__ grad_b,
    float* __restrict__ grad_query,
    const int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float scale = diagonal[index];
    grad_diagonal[index] = grad_diagonal_dual[index]
        - c_upper[index] * y[index] / (scale * scale);
    grad_key[index] = c_lower[index];
    const int panel = index / (kChunk * kRank);
    const int local = index % (kChunk * kRank);
    grad_b[index] = grad_rhs[(panel * kRhs + 0) * kChunk * kRank + local];
    grad_query[index] = grad_rhs[(panel * kRhs + 1) * kChunk * kRank + local];
}

std::tuple<at::Tensor, at::Tensor>
blocked_action_cuda(
    const at::Tensor& boundary_h,
    const at::Tensor& boundary_r,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& weights,
    const at::Tensor& alpha,
    const at::Tensor& coefficient,
    const at::Tensor& diagonal,
    const at::Tensor& grad_d) {
    const int panels = static_cast<int>(boundary_h.size(0));
    auto c_upper = at::empty_like(grad_d);
    auto c_lower = at::empty_like(grad_d);
    c10::cuda::CUDAGuard guard(boundary_h.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    blocked_transpose_solve_kernel<<<panels, kThreads, 0, stream>>>(
        boundary_h.data_ptr<float>(), boundary_r.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), weights.data_ptr<float>(),
        alpha.data_ptr<float>(), coefficient.data_ptr<float>(),
        diagonal.data_ptr<float>(),
        grad_d.data_ptr<float>(), c_upper.data_ptr<float>(),
        c_lower.data_ptr<float>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {c_upper, c_lower};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "packet_frame_action_vjp128(Tensor boundary_h, Tensor boundary_r, Tensor u, Tensor h, "
        "Tensor weights, Tensor alpha, Tensor coefficient, Tensor diagonal, "
        "Tensor grad_d) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("packet_frame_action_vjp128", &blocked_action_cuda);
}
