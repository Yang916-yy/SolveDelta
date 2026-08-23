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
constexpr float kRadius = 1.0f / 8.0f;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

struct FloatFloat {
    float hi;
    float lo;
};

__device__ __forceinline__ FloatFloat ff_add(
    FloatFloat left, FloatFloat right) {
    const float sum = __fadd_rn(left.hi, right.hi);
    const float virtual_right = __fsub_rn(sum, left.hi);
    const float sum_error = __fadd_rn(
        __fsub_rn(left.hi, __fsub_rn(sum, virtual_right)),
        __fsub_rn(right.hi, virtual_right));
    const float correction = __fadd_rn(
        sum_error, __fadd_rn(left.lo, right.lo));
    const float high = __fadd_rn(sum, correction);
    return {high, __fsub_rn(correction, __fsub_rn(high, sum))};
}

__device__ __forceinline__ float ff_value(FloatFloat value) {
    return __fadd_rn(value.hi, value.lo);
}

__device__ __forceinline__ FloatFloat ff_product(float left, float right) {
    const float product = __fmul_rn(left, right);
    return {product, __fmaf_rn(left, right, -product)};
}

__device__ __forceinline__ FloatFloat ff_multiply(
    FloatFloat value, float scale) {
    return ff_add(
        ff_product(value.hi, scale), ff_product(value.lo, scale));
}

__device__ __forceinline__ FloatFloat merged_boundary(
    const float* __restrict__ boundary,
    const int scalar,
    const int component) {
    const int index = (scalar * 4 + component) * 2;
    return {boundary[index + 0], boundary[index + 1]};
}

__launch_bounds__(512, 1)
__global__ void chart_scalar_kernel(
    const float* __restrict__ boundary_contraction,
    const float* __restrict__ local_contraction,
    const float* __restrict__ alpha,
    const float* __restrict__ norm_sq,
    const float* __restrict__ strength,
    const float* __restrict__ centered_h_diagonal,
    const float* __restrict__ r_diagonal,
    const float* __restrict__ diagonal,
    const float* __restrict__ grad_diagonal,
    float* __restrict__ eta,
    float* __restrict__ grad_h_diagonal,
    float* __restrict__ grad_r_diagonal,
    float* __restrict__ grad_strength,
    float* __restrict__ grad_alpha_direct,
    float* __restrict__ grad_initial_mass,
    const int heads,
    const int chunks,
    const int panels) {
    __shared__ float strength_partial[kChunk];
    __shared__ float mass_partial[kChunk];
    const int panel = blockIdx.x;
    const int target = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (panel >= panels) return;
    const int scalar = panel * kChunk + target;
    const int head = (panel / chunks) % heads;
    const float s = strength[head];
    float radial_strength = 0.0f;
    float radial_mass = 0.0f;
    if (lane < 4) {
        const int item = scalar * 4 + lane;
        const float denominator = kRadius * kRadius + norm_sq[item];
        const float pbar = kRadius * rsqrtf(denominator);
        const FloatFloat boundary = merged_boundary(
            boundary_contraction, scalar, lane);
        const int local = item * 2;
        const FloatFloat raw = ff_add(
            ff_multiply(boundary, alpha[scalar]),
            {local_contraction[local + 0], local_contraction[local + 1]});
        const float qbar = ff_value(ff_multiply(raw, pbar));
        eta[item] = s * s * s * qbar / denominator;
        radial_strength = kRadius * kRadius * qbar / denominator;
        radial_mass = s * kRadius * kRadius * qbar / denominator;
    }
    radial_strength = warp_sum(radial_strength);
    radial_mass = warp_sum(radial_mass);
    float diagonal_strength = 0.0f;
    float diagonal_mass = 0.0f;
    for (int coordinate = lane; coordinate < kRank; coordinate += 32) {
        const int index = scalar * kRank + coordinate;
        const float ah = centered_h_diagonal[index];
        const float ar = r_diagonal[index];
        const float th = tanhf(s * ah / kRadius);
        const float tr = tanhf(s * ar / kRadius);
        const float glog = grad_diagonal[index] * diagonal[index];
        const float sech_h = 1.0f - th * th;
        const float sech_r = 1.0f - tr * tr;
        grad_h_diagonal[index] = glog * s * sech_h;
        grad_r_diagonal[index] = glog * s * sech_r;
        diagonal_strength = fmaf(glog, ah * sech_h + ar * sech_r, diagonal_strength);
        diagonal_mass = fmaf(
            glog * s * sech_h,
            ah + 1.0f / static_cast<float>(kRank),
            diagonal_mass);
        diagonal_mass = fmaf(
            glog * s * sech_r, ar, diagonal_mass);
    }
    diagonal_strength = warp_sum(diagonal_strength);
    diagonal_mass = warp_sum(diagonal_mass);
    if (lane == 0) {
        strength_partial[target] = radial_strength + diagonal_strength;
        mass_partial[target] = -alpha[scalar] * (radial_mass + diagonal_mass);
        FloatFloat boundary_total{0.0f, 0.0f};
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            const float denominator = kRadius * kRadius
                + norm_sq[scalar * 4 + component];
            const float pbar = kRadius * rsqrtf(denominator);
            boundary_total = ff_add(
                boundary_total,
                ff_multiply(merged_boundary(
                    boundary_contraction, scalar, component),
                    pbar));
        }
        grad_alpha_direct[scalar] = s * ff_value(boundary_total);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        FloatFloat strength_sum{0.0f, 0.0f};
        FloatFloat mass_sum{0.0f, 0.0f};
#pragma unroll
        for (int token = 0; token < kChunk; ++token) {
            strength_sum = ff_add(strength_sum, {strength_partial[token], 0.0f});
            mass_sum = ff_add(mass_sum, {mass_partial[token], 0.0f});
        }
        grad_strength[panel] = ff_value(strength_sum);
        grad_initial_mass[panel] = ff_value(mass_sum);
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor>
chart_scalar_cuda(
    const at::Tensor& boundary_contraction,
    const at::Tensor& local_contraction,
    const at::Tensor& alpha,
    const at::Tensor& norm_sq,
    const at::Tensor& strength,
    const at::Tensor& centered_h_diagonal,
    const at::Tensor& r_diagonal,
    const at::Tensor& diagonal,
    const at::Tensor& grad_diagonal,
    int64_t heads,
    int64_t chunks) {
    const int panels = static_cast<int>(boundary_contraction.size(0));
    TORCH_CHECK(heads > 0, "heads must be positive");
    TORCH_CHECK(chunks > 0, "chunks must be positive");
    TORCH_CHECK(strength.numel() == heads, "strength must contain one value per head");
    TORCH_CHECK(
        panels % (heads * chunks) == 0,
        "panel count must be divisible by heads * chunks");
    auto eta = at::empty_like(norm_sq);
    auto grad_h_diagonal = at::empty_like(centered_h_diagonal);
    auto grad_r_diagonal = at::empty_like(r_diagonal);
    auto grad_strength = at::empty({panels}, strength.options());
    auto grad_alpha_direct = at::empty(
        {panels, kChunk}, boundary_contraction.options());
    auto grad_initial_mass = at::empty(
        {panels}, boundary_contraction.options());
    c10::cuda::CUDAGuard guard(boundary_contraction.device());
    chart_scalar_kernel<<<panels, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
        boundary_contraction.data_ptr<float>(), local_contraction.data_ptr<float>(),
        alpha.data_ptr<float>(),
        norm_sq.data_ptr<float>(),
        strength.data_ptr<float>(),
        centered_h_diagonal.data_ptr<float>(), r_diagonal.data_ptr<float>(),
        diagonal.data_ptr<float>(), grad_diagonal.data_ptr<float>(),
        eta.data_ptr<float>(), grad_h_diagonal.data_ptr<float>(),
        grad_r_diagonal.data_ptr<float>(), grad_strength.data_ptr<float>(),
        grad_alpha_direct.data_ptr<float>(), grad_initial_mass.data_ptr<float>(),
        static_cast<int>(heads), static_cast<int>(chunks), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        eta, grad_h_diagonal, grad_r_diagonal, grad_strength,
        grad_alpha_direct, grad_initial_mass};
}

__launch_bounds__(16, 1)
__global__ void prefix_vjp_kernel(
    const float* __restrict__ initial_mass,
    const float* __restrict__ log_decay,
    const float* __restrict__ grad_alpha,
    const float* __restrict__ grad_weights,
    float* __restrict__ grad_initial_mass,
    float* __restrict__ grad_log_decay,
    const int panels) {
    __shared__ float grad_prefix[kChunk];
    __shared__ float grad_mass0[kChunk];
    const int panel = blockIdx.x;
    const int target = threadIdx.x;
    if (panel >= panels) return;
    float prefix = 0.0f;
    for (int index = 0; index <= target; ++index) {
        prefix += log_decay[panel * kChunk + index];
    }
    const float q = expf(prefix);
    float mass = q * initial_mass[panel];
    for (int source = 0; source <= target; ++source) {
        float source_prefix = 0.0f;
        for (int index = 0; index <= source; ++index) {
            source_prefix += log_decay[panel * kChunk + index];
        }
        mass += expf(prefix - source_prefix);
    }
    const float ga = grad_alpha[panel * kChunk + target];
    float grad_mass = -ga * q / (mass * mass);
    for (int source = 0; source <= target; ++source) {
        float source_prefix = 0.0f;
        for (int index = 0; index <= source; ++index) {
            source_prefix += log_decay[panel * kChunk + index];
        }
        const float temporal = expf(prefix - source_prefix);
        grad_mass -= grad_weights[
            (panel * kChunk + source) * kChunk + target] * temporal
            / (mass * mass);
    }
    float value = (ga / mass + grad_mass * initial_mass[panel]) * q;
    for (int source = 0; source <= target; ++source) {
        float source_prefix = 0.0f;
        for (int index = 0; index <= source; ++index) {
            source_prefix += log_decay[panel * kChunk + index];
        }
        const float temporal = expf(prefix - source_prefix);
        const float grad_temporal = grad_weights[
            (panel * kChunk + source) * kChunk + target] / mass + grad_mass;
        value = fmaf(grad_temporal, temporal, value);
    }
    for (int later = target; later < kChunk; ++later) {
        float later_prefix = 0.0f;
        for (int index = 0; index <= later; ++index) {
            later_prefix += log_decay[panel * kChunk + index];
        }
        const float later_q = expf(later_prefix);
        float later_mass = later_q * initial_mass[panel];
        for (int source = 0; source <= later; ++source) {
            float source_prefix = 0.0f;
            for (int index = 0; index <= source; ++index) {
                source_prefix += log_decay[panel * kChunk + index];
            }
            later_mass += expf(later_prefix - source_prefix);
        }
        const float later_ga = grad_alpha[panel * kChunk + later];
        float later_grad_mass = -later_ga * later_q / (later_mass * later_mass);
        for (int source = 0; source <= later; ++source) {
            float source_prefix = 0.0f;
            for (int index = 0; index <= source; ++index) {
                source_prefix += log_decay[panel * kChunk + index];
            }
            const float temporal = expf(later_prefix - source_prefix);
            later_grad_mass -= grad_weights[
                (panel * kChunk + source) * kChunk + later] * temporal
                / (later_mass * later_mass);
        }
        float source_prefix = 0.0f;
        for (int index = 0; index <= target; ++index) {
            source_prefix += log_decay[panel * kChunk + index];
        }
        const float temporal = expf(later_prefix - source_prefix);
        const float grad_temporal = grad_weights[
            (panel * kChunk + target) * kChunk + later] / later_mass
            + later_grad_mass;
        value -= grad_temporal * temporal;
    }
    grad_prefix[target] = value;
    __syncthreads();
    float grad_log = 0.0f;
    for (int later = target; later < kChunk; ++later) {
        grad_log += grad_prefix[later];
    }
    grad_log_decay[panel * kChunk + target] = grad_log;
    grad_mass0[target] = grad_mass * q;
    __syncthreads();
    if (target == 0) {
        float gm0 = 0.0f;
#pragma unroll
        for (int token = 0; token < kChunk; ++token) gm0 += grad_mass0[token];
        grad_initial_mass[panel] = gm0;
    }
}

std::tuple<at::Tensor, at::Tensor> prefix_vjp_cuda(
    const at::Tensor& initial_mass,
    const at::Tensor& log_decay,
    const at::Tensor& grad_alpha,
    const at::Tensor& grad_weights) {
    const int panels = static_cast<int>(initial_mass.size(0));
    auto grad_initial_mass = at::empty_like(initial_mass);
    auto grad_log_decay = at::empty_like(log_decay);
    c10::cuda::CUDAGuard guard(initial_mass.device());
    prefix_vjp_kernel<<<panels, 16, 0, at::cuda::getCurrentCUDAStream()>>>(
        initial_mass.data_ptr<float>(), log_decay.data_ptr<float>(),
        grad_alpha.data_ptr<float>(), grad_weights.data_ptr<float>(),
        grad_initial_mass.data_ptr<float>(), grad_log_decay.data_ptr<float>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_initial_mass, grad_log_decay};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "packet_frame_chart_vjp128(Tensor boundary_contraction, "
        "Tensor local_contraction, Tensor alpha, "
        "Tensor norm_sq, Tensor strength, "
        "Tensor centered_h_diagonal, Tensor r_diagonal, "
        "Tensor diagonal, Tensor grad_diagonal, int heads, int chunks) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "packet_frame_prefix_vjp128(Tensor initial_mass, Tensor log_decay, Tensor grad_alpha, "
        "Tensor grad_weights) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("packet_frame_chart_vjp128", &chart_scalar_cuda);
    m.impl("packet_frame_prefix_vjp128", &prefix_vjp_cuda);
}
