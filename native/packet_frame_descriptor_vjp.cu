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
constexpr int kWarps = kChunk;
constexpr int kThreads = kWarps * 32;
constexpr int kAdjoints = 4;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

template <bool Upper>
__device__ __forceinline__ void transpose_solve(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float alpha,
    const float p,
    const float q,
    const float* __restrict__ rhs,
    const float* __restrict__ diagonal,
    float* __restrict__ output,
    const bool divide_rhs,
    const int panel,
    const int target) {
    const int lane = threadIdx.x & 31;
    const int source = lane;
    const int matrix_base = panel * kRank * kRank;
    const int vector_base = (panel * kChunk + target) * kRank;
    const int source_base = (panel * kChunk + source) * kRank;
    const float weight = source <= target
        ? weights[(panel * kChunk + source) * kChunk + target]
        : 0.0f;
    float z_u = 0.0f;

#pragma unroll 1
    for (int step = 0; step < kRank; ++step) {
        const int coordinate = Upper ? step : (kRank - 1 - step);
        float boundary_h_action = 0.0f;
        float boundary_d_action = 0.0f;
        if constexpr (Upper) {
            for (int row = lane; row < coordinate; row += 32) {
                const float value = output[row];
                boundary_h_action = fmaf(
                    boundary_h[matrix_base + row * kRank + coordinate],
                    value,
                    boundary_h_action);
                boundary_d_action = fmaf(
                    boundary_d[matrix_base + row * kRank + coordinate],
                    value,
                    boundary_d_action);
            }
        } else {
            for (int row = coordinate + 1 + lane; row < kRank; row += 32) {
                const float value = output[row];
                boundary_h_action = fmaf(
                    boundary_h[matrix_base + row * kRank + coordinate],
                    value,
                    boundary_h_action);
                boundary_d_action = fmaf(
                    boundary_d[matrix_base + row * kRank + coordinate],
                    value,
                    boundary_d_action);
            }
        }
        const float local = source <= target
            ? weight
                * (p * u[source_base + coordinate]
                   + q * h[source_base + coordinate])
                * z_u
            : 0.0f;
        const float action = alpha
            * (p * warp_sum(boundary_h_action)
               + q * warp_sum(boundary_d_action))
            + warp_sum(local);
        if (lane == 0) {
            float value = rhs[coordinate];
            if (divide_rhs) {
                value /= diagonal[vector_base + coordinate];
            }
            output[coordinate] = value - action;
        }
        __syncwarp();
        if (source <= target) {
            z_u = fmaf(
                u[source_base + coordinate], output[coordinate], z_u);
        }
        __syncwarp();
    }
}

template <bool Upper>
__device__ __forceinline__ void row_multiply2(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float alpha,
    const float p,
    const float q,
    const float* __restrict__ rhs0,
    const float* __restrict__ rhs1,
    float* __restrict__ output0,
    float* __restrict__ output1,
    const int panel,
    const int target) {
    const int lane = threadIdx.x & 31;
    const int source = lane;
    const int matrix_base = panel * kRank * kRank;
    const int source_base = (panel * kChunk + source) * kRank;
    const float weight = source <= target
        ? weights[(panel * kChunk + source) * kChunk + target]
        : 0.0f;
    float z_u0 = 0.0f;
    float z_h0 = 0.0f;
    float z_u1 = 0.0f;
    float z_h1 = 0.0f;

#pragma unroll 1
    for (int step = 0; step < kRank; ++step) {
        const int coordinate = Upper ? (kRank - 1 - step) : step;
        float bh0 = 0.0f;
        float bd0 = 0.0f;
        float bh1 = 0.0f;
        float bd1 = 0.0f;
        if constexpr (Upper) {
            for (int column = coordinate + 1 + lane; column < kRank; column += 32) {
                const float h_value = boundary_h[
                    matrix_base + coordinate * kRank + column];
                const float d_value = boundary_d[
                    matrix_base + coordinate * kRank + column];
                bh0 = fmaf(h_value, rhs0[column], bh0);
                bd0 = fmaf(d_value, rhs0[column], bd0);
                bh1 = fmaf(h_value, rhs1[column], bh1);
                bd1 = fmaf(d_value, rhs1[column], bd1);
            }
        } else {
            for (int column = lane; column < coordinate; column += 32) {
                const float h_value = boundary_h[
                    matrix_base + coordinate * kRank + column];
                const float d_value = boundary_d[
                    matrix_base + coordinate * kRank + column];
                bh0 = fmaf(h_value, rhs0[column], bh0);
                bd0 = fmaf(d_value, rhs0[column], bd0);
                bh1 = fmaf(h_value, rhs1[column], bh1);
                bd1 = fmaf(d_value, rhs1[column], bd1);
            }
        }
        float local0 = 0.0f;
        float local1 = 0.0f;
        if (source <= target) {
            const float u_i = u[source_base + coordinate];
            local0 = weight * u_i * (p * z_u0 + q * z_h0);
            local1 = weight * u_i * (p * z_u1 + q * z_h1);
        }
        const float action0 = alpha
            * (p * warp_sum(bh0) + q * warp_sum(bd0))
            + warp_sum(local0);
        const float action1 = alpha
            * (p * warp_sum(bh1) + q * warp_sum(bd1))
            + warp_sum(local1);
        if (lane == 0) {
            output0[coordinate] = rhs0[coordinate] + action0;
            output1[coordinate] = rhs1[coordinate] + action1;
        }
        if (source <= target) {
            const float u_i = u[source_base + coordinate];
            const float h_i = h[source_base + coordinate];
            z_u0 = fmaf(u_i, rhs0[coordinate], z_u0);
            z_h0 = fmaf(h_i, rhs0[coordinate], z_h0);
            z_u1 = fmaf(u_i, rhs1[coordinate], z_u1);
            z_h1 = fmaf(h_i, rhs1[coordinate], z_h1);
        }
        __syncwarp();
    }
}

__launch_bounds__(kThreads, 1)
__global__ void packet_action_vjp_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ y,
    const float* __restrict__ d,
    const float* __restrict__ dual_lower,
    const float* __restrict__ grad_d,
    const float* __restrict__ grad_e,
    const float* __restrict__ grad_chi,
    float* __restrict__ c_upper,
    float* __restrict__ c_lower,
    float* __restrict__ grad_dual_lower,
    float* __restrict__ grad_diagonal,
    float* __restrict__ grad_key,
    float* __restrict__ grad_b,
    float* __restrict__ grad_query,
    const int panels) {
    __shared__ float workspace[kWarps * kAdjoints * kRank];
    const int panel = blockIdx.x;
    const int target = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (panel >= panels) {
        return;
    }
    const int vector_base = (panel * kChunk + target) * kRank;
    const int coefficient_base = (panel * kChunk + target) * 4;
    const float alpha_t = alpha[panel * kChunk + target];
    const float pl = coefficient[coefficient_base + 0];
    const float ql = coefficient[coefficient_base + 1];
    const float pu = coefficient[coefficient_base + 2];
    const float qu = coefficient[coefficient_base + 3];
    float* cu = workspace + (target * kAdjoints + 0) * kRank;
    float* cl = workspace + (target * kAdjoints + 1) * kRank;
    float* gl0 = workspace + (target * kAdjoints + 2) * kRank;
    float* gl1 = workspace + (target * kAdjoints + 3) * kRank;

    transpose_solve<true>(
        boundary_h, boundary_d, u, h, weights,
        alpha_t, pu, qu, grad_d + vector_base, nullptr,
        cu, false, panel, target);
    transpose_solve<false>(
        boundary_h, boundary_d, u, h, weights,
        alpha_t, pl, ql, cu, diagonal, cl, true, panel, target);

    row_multiply2<true>(
        boundary_h, boundary_d, u, h, weights,
        alpha_t, pu, qu,
        grad_e + vector_base, grad_chi + vector_base,
        gl0, gl1, panel, target);
    __syncwarp();
    if (lane < 4) {
        // Keep all lanes in the warp converged before gl is overwritten by
        // the diagonal pullback.
    }
    for (int coordinate = lane; coordinate < kRank; coordinate += 32) {
        const float scale = diagonal[vector_base + coordinate];
        gl0[coordinate] *= scale;
        gl1[coordinate] *= scale;
    }
    __syncwarp();
    row_multiply2<false>(
        boundary_h, boundary_d, u, h, weights,
        alpha_t, pl, ql, gl0, gl1,
        grad_b + vector_base, grad_query + vector_base,
        panel, target);

    for (int coordinate = lane; coordinate < kRank; coordinate += 32) {
        const int index = vector_base + coordinate;
        const float scale = diagonal[index];
        const float gz0 = gl0[coordinate] / scale;
        const float gz1 = gl1[coordinate] / scale;
        c_upper[index] = cu[coordinate];
        c_lower[index] = cl[coordinate];
        grad_dual_lower[(panel * kChunk * 2 + target * 2 + 0) * kRank + coordinate]
            = gl0[coordinate];
        grad_dual_lower[(panel * kChunk * 2 + target * 2 + 1) * kRank + coordinate]
            = gl1[coordinate];
        grad_diagonal[index] =
            -cu[coordinate] * y[index] / (scale * scale)
            + gz0 * dual_lower[(panel * kChunk * 2 + target * 2 + 0) * kRank + coordinate]
            + gz1 * dual_lower[(panel * kChunk * 2 + target * 2 + 1) * kRank + coordinate];
        grad_key[index] = cl[coordinate];
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
packet_action_vjp_cuda(
    const at::Tensor& boundary_h,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& weights,
    const at::Tensor& alpha,
    const at::Tensor& coefficient,
    const at::Tensor& diagonal,
    const at::Tensor& y,
    const at::Tensor& d,
    const at::Tensor& dual_lower,
    const at::Tensor& grad_d,
    const at::Tensor& grad_e,
    const at::Tensor& grad_chi) {
    const auto panels = boundary_h.size(0);
    auto cu = at::empty_like(y);
    auto cl = at::empty_like(y);
    auto gl = at::empty({panels, kChunk, 2, kRank}, y.options());
    auto gdiag = at::empty_like(diagonal);
    auto gkey = at::empty_like(y);
    auto gb = at::empty_like(y);
    auto gq = at::empty_like(y);
    c10::cuda::CUDAGuard guard(boundary_h.device());
    packet_action_vjp_kernel<<<
        panels, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        boundary_h.data_ptr<float>(), boundary_d.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), weights.data_ptr<float>(),
        alpha.data_ptr<float>(), coefficient.data_ptr<float>(),
        diagonal.data_ptr<float>(), y.data_ptr<float>(), d.data_ptr<float>(),
        dual_lower.data_ptr<float>(), grad_d.data_ptr<float>(),
        grad_e.data_ptr<float>(), grad_chi.data_ptr<float>(),
        cu.data_ptr<float>(), cl.data_ptr<float>(), gl.data_ptr<float>(),
        gdiag.data_ptr<float>(), gkey.data_ptr<float>(),
        gb.data_ptr<float>(), gq.data_ptr<float>(),
        static_cast<int>(panels));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cu, cl, gl, gdiag, gkey, gb, gq};
}

__device__ __forceinline__ float lower_packet_entry(
    const float* __restrict__ y,
    const float* __restrict__ b,
    const float* __restrict__ query,
    const float* __restrict__ c_lower,
    const float* __restrict__ grad_dual_lower,
    const int vector_base,
    const int dual_base,
    const int row,
    const int column) {
    return -c_lower[vector_base + row] * y[vector_base + column]
        + b[vector_base + row] * grad_dual_lower[dual_base + column]
        + query[vector_base + row]
            * grad_dual_lower[dual_base + kRank + column];
}

__device__ __forceinline__ float upper_packet_entry(
    const float* __restrict__ diagonal,
    const float* __restrict__ write_direction,
    const float* __restrict__ dual_lower,
    const float* __restrict__ grad_e,
    const float* __restrict__ grad_chi,
    const float* __restrict__ c_upper,
    const int vector_base,
    const int dual_base,
    const int row,
    const int column) {
    const float scale = diagonal[vector_base + row];
    return -c_upper[vector_base + row]
            * write_direction[vector_base + column]
        + scale * dual_lower[dual_base + row]
            * grad_e[vector_base + column]
        + scale * dual_lower[dual_base + kRank + row]
            * grad_chi[vector_base + column];
}

__launch_bounds__(kThreads, 1)
__global__ void factor_boundary_contract_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_d,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ y,
    const float* __restrict__ write_direction,
    const float* __restrict__ b,
    const float* __restrict__ query,
    const float* __restrict__ dual_lower,
    const float* __restrict__ grad_e,
    const float* __restrict__ grad_chi,
    const float* __restrict__ c_upper,
    const float* __restrict__ c_lower,
    const float* __restrict__ grad_dual_lower,
    float* __restrict__ grad_boundary_h,
    float* __restrict__ grad_boundary_d,
    float* __restrict__ grad_alpha,
    float* __restrict__ grad_coefficient,
    const int panels) {
    const int panel = blockIdx.x;
    const int target = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (panel >= panels) {
        return;
    }
    const int matrix_base = panel * kRank * kRank;

    // The unavoidable dense boundary adjoint is accumulated without ever
    // materializing any tokenwise factor cotangent.
    for (int entry = threadIdx.x; entry < kRank * kRank; entry += kThreads) {
        const int row = entry / kRank;
        const int column = entry % kRank;
        float grad_h_value = 0.0f;
        float grad_d_value = 0.0f;
#pragma unroll
        for (int token = 0; token < kChunk; ++token) {
            const int vector_base = (panel * kChunk + token) * kRank;
            const int dual_base = (panel * kChunk * 2 + token * 2) * kRank;
            const int coeff_base = (panel * kChunk + token) * 4;
            float packet = 0.0f;
            float ph = 0.0f;
            float pd = 0.0f;
            if (row > column) {
                packet = lower_packet_entry(
                    y, b, query, c_lower, grad_dual_lower,
                    vector_base, dual_base, row, column);
                ph = coefficient[coeff_base + 0];
                pd = coefficient[coeff_base + 1];
            } else if (row < column) {
                packet = upper_packet_entry(
                    diagonal, write_direction, dual_lower, grad_e, grad_chi,
                    c_upper, vector_base, dual_base, row, column);
                ph = coefficient[coeff_base + 2];
                pd = coefficient[coeff_base + 3];
            }
            const float scale = alpha[panel * kChunk + token] * packet;
            grad_h_value = fmaf(scale, ph, grad_h_value);
            grad_d_value = fmaf(scale, pd, grad_d_value);
        }
        grad_boundary_h[matrix_base + entry] = grad_h_value;
        grad_boundary_d[matrix_base + entry] = grad_d_value;
    }

    // One warp owns each token's boundary scalar contractions.
    const int vector_base = (panel * kChunk + target) * kRank;
    const int dual_base = (panel * kChunk * 2 + target * 2) * kRank;
    float h_lower = 0.0f;
    float d_lower = 0.0f;
    float h_upper = 0.0f;
    float d_upper = 0.0f;
    for (int entry = lane; entry < kRank * kRank; entry += 32) {
        const int row = entry / kRank;
        const int column = entry % kRank;
        const float boundary_h_value = boundary_h[matrix_base + entry];
        const float boundary_d_value = boundary_d[matrix_base + entry];
        if (row > column) {
            const float packet = lower_packet_entry(
                y, b, query, c_lower, grad_dual_lower,
                vector_base, dual_base, row, column);
            h_lower = fmaf(packet, boundary_h_value, h_lower);
            d_lower = fmaf(packet, boundary_d_value, d_lower);
        } else if (row < column) {
            const float packet = upper_packet_entry(
                diagonal, write_direction, dual_lower, grad_e, grad_chi,
                c_upper, vector_base, dual_base, row, column);
            h_upper = fmaf(packet, boundary_h_value, h_upper);
            d_upper = fmaf(packet, boundary_d_value, d_upper);
        }
    }
    h_lower = warp_sum(h_lower);
    d_lower = warp_sum(d_lower);
    h_upper = warp_sum(h_upper);
    d_upper = warp_sum(d_upper);
    if (lane == 0) {
        const int scalar = panel * kChunk + target;
        const int coeff_base = scalar * 4;
        const float pl = coefficient[coeff_base + 0];
        const float ql = coefficient[coeff_base + 1];
        const float pu = coefficient[coeff_base + 2];
        const float qu = coefficient[coeff_base + 3];
        const float alpha_t = alpha[scalar];
        grad_coefficient[coeff_base + 0] = alpha_t * h_lower;
        grad_coefficient[coeff_base + 1] = alpha_t * d_lower;
        grad_coefficient[coeff_base + 2] = alpha_t * h_upper;
        grad_coefficient[coeff_base + 3] = alpha_t * d_upper;
        grad_alpha[scalar] =
            pl * h_lower + ql * d_lower + pu * h_upper + qu * d_upper;
    }
}

template <bool Upper>
__device__ __forceinline__ void local_packet_contract(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ diagonal,
    const float* __restrict__ y,
    const float* __restrict__ write_direction,
    const float* __restrict__ b,
    const float* __restrict__ query,
    const float* __restrict__ dual_lower,
    const float* __restrict__ grad_e,
    const float* __restrict__ grad_chi,
    const float* __restrict__ c_upper,
    const float* __restrict__ c_lower,
    const float* __restrict__ grad_dual_lower,
    float* __restrict__ grad_u,
    float* __restrict__ grad_h,
    const int panel,
    const int target,
    const int source,
    const float weight,
    const float p,
    const float q,
    float& contraction_h,
    float& contraction_d) {
    const int target_base = (panel * kChunk + target) * kRank;
    const int source_base = (panel * kChunk + source) * kRank;
    const int dual_base = (panel * kChunk * 2 + target * 2) * kRank;
    float prefix_u0 = 0.0f;
    float prefix_u1 = 0.0f;
    float prefix_u2 = 0.0f;
    float prefix_h0 = 0.0f;
    float prefix_h1 = 0.0f;
    float prefix_h2 = 0.0f;

#pragma unroll 1
    for (int step = 0; step < kRank; ++step) {
        const int coordinate = Upper ? (kRank - 1 - step) : step;
        float a0;
        float a1;
        float a2;
        float right0;
        float right1;
        float right2;
        if constexpr (Upper) {
            const float scale = diagonal[target_base + coordinate];
            a0 = -c_upper[target_base + coordinate];
            a1 = scale * dual_lower[dual_base + coordinate];
            a2 = scale * dual_lower[dual_base + kRank + coordinate];
            right0 = write_direction[target_base + coordinate];
            right1 = grad_e[target_base + coordinate];
            right2 = grad_chi[target_base + coordinate];
        } else {
            a0 = -c_lower[target_base + coordinate];
            a1 = b[target_base + coordinate];
            a2 = query[target_base + coordinate];
            right0 = y[target_base + coordinate];
            right1 = grad_dual_lower[dual_base + coordinate];
            right2 = grad_dual_lower[dual_base + kRank + coordinate];
        }
        const float e_u = a0 * prefix_u0 + a1 * prefix_u1 + a2 * prefix_u2;
        const float e_h = a0 * prefix_h0 + a1 * prefix_h1 + a2 * prefix_h2;
        const float u_value = u[source_base + coordinate];
        const float h_value = h[source_base + coordinate];
        contraction_h = fmaf(u_value, e_u, contraction_h);
        contraction_d = fmaf(u_value, e_h, contraction_d);
        atomicAdd(
            grad_u + source_base + coordinate,
            weight * (p * e_u + q * e_h));
        prefix_u0 = fmaf(right0, u_value, prefix_u0);
        prefix_u1 = fmaf(right1, u_value, prefix_u1);
        prefix_u2 = fmaf(right2, u_value, prefix_u2);
        prefix_h0 = fmaf(right0, h_value, prefix_h0);
        prefix_h1 = fmaf(right1, h_value, prefix_h1);
        prefix_h2 = fmaf(right2, h_value, prefix_h2);
    }

    float suffix0 = 0.0f;
    float suffix1 = 0.0f;
    float suffix2 = 0.0f;
#pragma unroll 1
    for (int step = 0; step < kRank; ++step) {
        const int coordinate = Upper ? step : (kRank - 1 - step);
        float a0;
        float a1;
        float a2;
        float right0;
        float right1;
        float right2;
        if constexpr (Upper) {
            const float scale = diagonal[target_base + coordinate];
            a0 = -c_upper[target_base + coordinate];
            a1 = scale * dual_lower[dual_base + coordinate];
            a2 = scale * dual_lower[dual_base + kRank + coordinate];
            right0 = write_direction[target_base + coordinate];
            right1 = grad_e[target_base + coordinate];
            right2 = grad_chi[target_base + coordinate];
        } else {
            a0 = -c_lower[target_base + coordinate];
            a1 = b[target_base + coordinate];
            a2 = query[target_base + coordinate];
            right0 = y[target_base + coordinate];
            right1 = grad_dual_lower[dual_base + coordinate];
            right2 = grad_dual_lower[dual_base + kRank + coordinate];
        }
        const float transposed =
            right0 * suffix0 + right1 * suffix1 + right2 * suffix2;
        const float u_value = u[source_base + coordinate];
        atomicAdd(grad_u + source_base + coordinate, weight * p * transposed);
        atomicAdd(grad_h + source_base + coordinate, weight * q * transposed);
        suffix0 = fmaf(a0, u_value, suffix0);
        suffix1 = fmaf(a1, u_value, suffix1);
        suffix2 = fmaf(a2, u_value, suffix2);
    }
}

__launch_bounds__(kThreads, 1)
__global__ void factor_local_contract_kernel(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ y,
    const float* __restrict__ write_direction,
    const float* __restrict__ b,
    const float* __restrict__ query,
    const float* __restrict__ dual_lower,
    const float* __restrict__ grad_e,
    const float* __restrict__ grad_chi,
    const float* __restrict__ c_upper,
    const float* __restrict__ c_lower,
    const float* __restrict__ grad_dual_lower,
    float* __restrict__ grad_u,
    float* __restrict__ grad_h,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_coefficient,
    const int panels) {
    const int panel = blockIdx.x;
    const int target = threadIdx.x >> 5;
    const int source = threadIdx.x & 31;
    if (panel >= panels) {
        return;
    }
    const int coeff_base = (panel * kChunk + target) * 4;
    const float pl = coefficient[coeff_base + 0];
    const float ql = coefficient[coeff_base + 1];
    const float pu = coefficient[coeff_base + 2];
    const float qu = coefficient[coeff_base + 3];
    float weight = 0.0f;
    if (source <= target) {
        weight = weights[(panel * kChunk + source) * kChunk + target];
    }
    float h_lower = 0.0f;
    float d_lower = 0.0f;
    float h_upper = 0.0f;
    float d_upper = 0.0f;
    if (source <= target) {
        local_packet_contract<false>(
            u, h, diagonal, y, write_direction, b, query, dual_lower,
            grad_e, grad_chi, c_upper, c_lower, grad_dual_lower,
            grad_u, grad_h, panel, target, source, weight, pl, ql,
            h_lower, d_lower);
        local_packet_contract<true>(
            u, h, diagonal, y, write_direction, b, query, dual_lower,
            grad_e, grad_chi, c_upper, c_lower, grad_dual_lower,
            grad_u, grad_h, panel, target, source, weight, pu, qu,
            h_upper, d_upper);
        grad_weights[(panel * kChunk + source) * kChunk + target] =
            pl * h_lower + ql * d_lower + pu * h_upper + qu * d_upper;
    }
    const float gp_l = warp_sum(weight * h_lower);
    const float gq_l = warp_sum(weight * d_lower);
    const float gp_u = warp_sum(weight * h_upper);
    const float gq_u = warp_sum(weight * d_upper);
    if (source == 0) {
        grad_coefficient[coeff_base + 0] += gp_l;
        grad_coefficient[coeff_base + 1] += gq_l;
        grad_coefficient[coeff_base + 2] += gp_u;
        grad_coefficient[coeff_base + 3] += gq_u;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
factor_contract_cuda(
    const at::Tensor& boundary_h,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& weights,
    const at::Tensor& alpha,
    const at::Tensor& coefficient,
    const at::Tensor& diagonal,
    const at::Tensor& y,
    const at::Tensor& write_direction,
    const at::Tensor& b,
    const at::Tensor& query,
    const at::Tensor& dual_lower,
    const at::Tensor& grad_e,
    const at::Tensor& grad_chi,
    const at::Tensor& c_upper,
    const at::Tensor& c_lower,
    const at::Tensor& grad_dual_lower) {
    const auto panels = boundary_h.size(0);
    auto grad_boundary_h = at::empty_like(boundary_h);
    auto grad_boundary_d = at::empty_like(boundary_d);
    auto grad_u = at::zeros_like(u);
    auto grad_h = at::zeros_like(h);
    auto grad_weights = at::zeros_like(weights);
    auto grad_alpha = at::empty_like(alpha);
    auto grad_coefficient = at::empty_like(coefficient);
    c10::cuda::CUDAGuard guard(boundary_h.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    factor_boundary_contract_kernel<<<panels, kThreads, 0, stream>>>(
        boundary_h.data_ptr<float>(), boundary_d.data_ptr<float>(),
        alpha.data_ptr<float>(), coefficient.data_ptr<float>(),
        diagonal.data_ptr<float>(), y.data_ptr<float>(),
        write_direction.data_ptr<float>(), b.data_ptr<float>(),
        query.data_ptr<float>(), dual_lower.data_ptr<float>(),
        grad_e.data_ptr<float>(), grad_chi.data_ptr<float>(),
        c_upper.data_ptr<float>(), c_lower.data_ptr<float>(),
        grad_dual_lower.data_ptr<float>(), grad_boundary_h.data_ptr<float>(),
        grad_boundary_d.data_ptr<float>(), grad_alpha.data_ptr<float>(),
        grad_coefficient.data_ptr<float>(), static_cast<int>(panels));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    factor_local_contract_kernel<<<panels, kThreads, 0, stream>>>(
        u.data_ptr<float>(), h.data_ptr<float>(), weights.data_ptr<float>(),
        coefficient.data_ptr<float>(), diagonal.data_ptr<float>(),
        y.data_ptr<float>(), write_direction.data_ptr<float>(),
        b.data_ptr<float>(), query.data_ptr<float>(),
        dual_lower.data_ptr<float>(), grad_e.data_ptr<float>(),
        grad_chi.data_ptr<float>(), c_upper.data_ptr<float>(),
        c_lower.data_ptr<float>(), grad_dual_lower.data_ptr<float>(),
        grad_u.data_ptr<float>(), grad_h.data_ptr<float>(),
        grad_weights.data_ptr<float>(), grad_coefficient.data_ptr<float>(),
        static_cast<int>(panels));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_boundary_h, grad_boundary_d, grad_u, grad_h,
        grad_weights, grad_alpha, grad_coefficient};
}

__launch_bounds__(kThreads, 1)
__global__ void descriptor_kernel(
    const float* __restrict__ key,
    const float* __restrict__ erase,
    const float* __restrict__ skew,
    const float* __restrict__ omega_key,
    const float* __restrict__ dual_rhs,
    const float* __restrict__ diagonal,
    const float* __restrict__ y,
    const float* __restrict__ d,
    const float* __restrict__ dual_lower,
    const float* __restrict__ grad_e,
    const float* __restrict__ grad_chi,
    const float* __restrict__ c_upper,
    const float* __restrict__ c_lower,
    const float* __restrict__ grad_dual_lower,
    const float* __restrict__ grad_b,
    float* __restrict__ lower_left,
    float* __restrict__ lower_right,
    float* __restrict__ upper_left,
    float* __restrict__ upper_right,
    float* __restrict__ grad_omega_key,
    float* __restrict__ grad_key_direct,
    float* __restrict__ grad_erase,
    float* __restrict__ grad_skew,
    const int panels) {
    const int panel = blockIdx.x;
    const int target = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (panel >= panels) {
        return;
    }
    const int vector_base = (panel * kChunk + target) * kRank;
    float tau = 0.0f;
    float norm_sq = 0.0f;
    float grad_scale = 0.0f;
    for (int coordinate = lane; coordinate < kRank; coordinate += 32) {
        const int index = vector_base + coordinate;
        const float kv = key[index];
        const float ov = omega_key[index];
        tau = fmaf(erase[index] * kv, kv, tau);
        norm_sq = fmaf(ov, ov, norm_sq);
    }
    tau = warp_sum(tau);
    norm_sq = warp_sum(norm_sq);
    const float inverse_norm = rsqrtf(1.0f + norm_sq);
    const float edit_scale = tau * (2.0f - tau)
        * skew[panel * kChunk + target];
    for (int coordinate = lane; coordinate < kRank; coordinate += 32) {
        const int index = vector_base + coordinate;
        grad_scale = fmaf(
            grad_b[index], omega_key[index] * inverse_norm, grad_scale);
    }
    grad_scale = warp_sum(grad_scale);
    const float grad_tau = grad_scale * (2.0f - 2.0f * tau)
        * skew[panel * kChunk + target];
    if (lane == 0) {
        grad_skew[panel * kChunk + target] = grad_scale * tau * (2.0f - tau);
    }
    float projection = 0.0f;
    for (int coordinate = lane; coordinate < kRank; coordinate += 32) {
        const int index = vector_base + coordinate;
        projection = fmaf(
            edit_scale * grad_b[index], omega_key[index], projection);
    }
    projection = warp_sum(projection);
    for (int coordinate = lane; coordinate < kRank; coordinate += 32) {
        const int index = vector_base + coordinate;
        const int rhs_base = (panel * kChunk + target) * 2 * kRank;
        const int grad_rhs_base = panel * 2 * kChunk * kRank
            + target * kRank;
        const float kv = key[index];
        const float ev = erase[index];
        const float ov = omega_key[index];
        const float gbv = grad_b[index];
        const float gw = inverse_norm * edit_scale * gbv
            - inverse_norm * inverse_norm * inverse_norm * ov * projection;
        const float b0 = ev * kv;
        const float gb0 = gbv + grad_tau * kv;
        grad_omega_key[index] = gw;
        grad_key_direct[index] =
            c_lower[index] + grad_tau * b0 + gb0 * ev;
        grad_erase[index] = gb0 * kv;
        // Descriptors use [P,C,R,5] storage; kAdjoints is four only for the
        // action workspace, so spell out the rank-five stride here.
        const int out = ((panel * kChunk + target) * kRank + coordinate) * 5;
        lower_left[out + 0] = -c_lower[index];
        lower_left[out + 1] = dual_rhs[rhs_base + coordinate];
        lower_left[out + 2] = dual_rhs[rhs_base + kRank + coordinate];
        lower_left[out + 3] = 0.5f * gw;
        lower_left[out + 4] = -0.5f * kv;
        lower_right[out + 0] = y[index];
        lower_right[out + 1] = grad_dual_lower[grad_rhs_base + coordinate];
        lower_right[out + 2] = grad_dual_lower[
            grad_rhs_base + kChunk * kRank + coordinate];
        lower_right[out + 3] = kv;
        lower_right[out + 4] = gw;
        upper_left[out + 0] = -c_upper[index];
        upper_left[out + 1] = diagonal[index] * dual_lower[
            (panel * kChunk * 2 + target * 2 + 0) * kRank + coordinate];
        upper_left[out + 2] = diagonal[index] * dual_lower[
            (panel * kChunk * 2 + target * 2 + 1) * kRank + coordinate];
        upper_left[out + 3] = 0.5f * gw;
        upper_left[out + 4] = -0.5f * kv;
        upper_right[out + 0] = d[index];
        upper_right[out + 1] = grad_e[index];
        upper_right[out + 2] = grad_chi[index];
        upper_right[out + 3] = kv;
        upper_right[out + 4] = gw;
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
descriptor_cuda(
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& skew,
    const at::Tensor& omega_key,
    const at::Tensor& dual_rhs,
    const at::Tensor& diagonal,
    const at::Tensor& y,
    const at::Tensor& d,
    const at::Tensor& dual_lower,
    const at::Tensor& grad_e,
    const at::Tensor& grad_chi,
    const at::Tensor& c_upper,
    const at::Tensor& c_lower,
    const at::Tensor& grad_dual_lower,
    const at::Tensor& grad_b) {
    const int panels = static_cast<int>(key.size(0));
    auto descriptor_options = key.options();
    auto lower_left = at::empty({panels, kChunk, kRank, 5}, descriptor_options);
    auto lower_right = at::empty({panels, kChunk, kRank, 5}, descriptor_options);
    auto upper_left = at::empty({panels, kChunk, kRank, 5}, descriptor_options);
    auto upper_right = at::empty({panels, kChunk, kRank, 5}, descriptor_options);
    auto grad_omega_key = at::empty_like(key);
    auto grad_key_direct = at::empty_like(key);
    auto grad_erase = at::empty_like(erase);
    auto grad_skew = at::empty_like(skew);
    c10::cuda::CUDAGuard guard(key.device());
    descriptor_kernel<<<panels, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        key.data_ptr<float>(), erase.data_ptr<float>(), skew.data_ptr<float>(),
        omega_key.data_ptr<float>(), dual_rhs.data_ptr<float>(),
        diagonal.data_ptr<float>(), y.data_ptr<float>(), d.data_ptr<float>(),
        dual_lower.data_ptr<float>(), grad_e.data_ptr<float>(),
        grad_chi.data_ptr<float>(), c_upper.data_ptr<float>(),
        c_lower.data_ptr<float>(), grad_dual_lower.data_ptr<float>(),
        grad_b.data_ptr<float>(), lower_left.data_ptr<float>(),
        lower_right.data_ptr<float>(), upper_left.data_ptr<float>(),
        upper_right.data_ptr<float>(), grad_omega_key.data_ptr<float>(),
        grad_key_direct.data_ptr<float>(), grad_erase.data_ptr<float>(),
        grad_skew.data_ptr<float>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {lower_left, lower_right, upper_left, upper_right,
            grad_omega_key, grad_key_direct, grad_erase, grad_skew};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "packet_frame_descriptor_vjp128(Tensor key, Tensor erase, Tensor skew, "
        "Tensor omega_key, Tensor dual_rhs, Tensor diagonal, Tensor y, Tensor d, "
        "Tensor dual_lower, Tensor grad_e, Tensor grad_chi, Tensor c_upper, "
        "Tensor c_lower, Tensor grad_dual_lower, Tensor grad_b) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("packet_frame_descriptor_vjp128", &descriptor_cuda);
}
