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
constexpr int kThreads = 128;
constexpr int kQbarComponents = 4;
constexpr int kBoundaryBlocks = kRank / 16;

__device__ __forceinline__ float reduce_group8(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 4, 8);
    value += __shfl_down_sync(0xffffffffu, value, 2, 8);
    value += __shfl_down_sync(0xffffffffu, value, 1, 8);
    return value;
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

// boundary_action enters as Omega_boundary @ input. Each eight-lane group owns
// one token, appends the exact packet-local Omega action, and uses
// Omega^T = -Omega to finish the direct key cotangent.
__launch_bounds__(kThreads, 2)
__global__ void omega_transpose_local_kernel(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ coefficient,
    const float* __restrict__ input,
    float* __restrict__ boundary_action,
    const float* __restrict__ boundary_qbar,
    const float* __restrict__ grad_key_direct,
    float* __restrict__ grad_key,
    float* __restrict__ qbar_parts,
    const int panels) {
    __shared__ float shared_input[kChunk * kRank];
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    const int vector_base = panel * kChunk * kRank;
    for (int index = threadIdx.x; index < kChunk * kRank; index += kThreads) {
        shared_input[index] = input[vector_base + index];
    }
    __syncthreads();

    const int target = threadIdx.x >> 3;
    const int lane = threadIdx.x & 7;
    const int source0 = lane;
    const int source1 = lane + 8;
    const int coefficient_base = (panel * kChunk + target) * 4;
    const float h_lower = coefficient[coefficient_base + 0];
    const float r_lower = coefficient[coefficient_base + 1];
    const float h_upper = coefficient[coefficient_base + 2];
    const float r_upper = coefficient[coefficient_base + 3];
    const int packet_base = panel * kPacket + target;
    const float weight0 = source0 <= target
        ? weights[packet_base + source0 * kChunk] : 0.0f;
    const float weight1 = source1 <= target
        ? weights[packet_base + source1 * kChunk] : 0.0f;
    float zu0 = 0.0f;
    float zh0 = 0.0f;
    float zu1 = 0.0f;
    float zh1 = 0.0f;

#pragma unroll 1
    for (int coordinate = 0; coordinate < kRank; ++coordinate) {
        const float input_value = shared_input[target * kRank + coordinate];
        float local = 0.0f;
#define APPLY_FORWARD(SOURCE, WEIGHT, ZU, ZH)                                  \
        do {                                                                     \
            if ((SOURCE) <= target) {                                            \
                const int source_index =                                         \
                    vector_base + (SOURCE) * kRank + coordinate;                 \
                const float uv = u[source_index];                                \
                const float hv = h[source_index];                                \
                const float action = uv * (                                      \
                    (h_lower - h_upper) * (ZU) + r_lower * (ZH))                 \
                    - r_upper * hv * (ZU);                                       \
                local = fmaf((WEIGHT), action, local);                           \
                (ZU) = fmaf(uv, input_value, (ZU));                              \
                (ZH) = fmaf(hv, input_value, (ZH));                              \
            }                                                                    \
        } while (0)
        APPLY_FORWARD(source0, weight0, zu0, zh0);
        APPLY_FORWARD(source1, weight1, zu1, zh1);
#undef APPLY_FORWARD
        const float action = 0.5f * reduce_group8(local);
        if (lane == 0) {
            boundary_action[vector_base + target * kRank + coordinate] += action;
        }
    }

    zu0 = zh0 = zu1 = zh1 = 0.0f;
#pragma unroll 1
    for (int step = 0; step < kRank; ++step) {
        const int coordinate = kRank - 1 - step;
        const float input_value = shared_input[target * kRank + coordinate];
        float local = 0.0f;
#define APPLY_REVERSE(SOURCE, WEIGHT, ZU, ZH)                                  \
        do {                                                                     \
            if ((SOURCE) <= target) {                                            \
                const int source_index =                                         \
                    vector_base + (SOURCE) * kRank + coordinate;                 \
                const float uv = u[source_index];                                \
                const float hv = h[source_index];                                \
                const float action = uv * (                                      \
                    (h_upper - h_lower) * (ZU) + r_upper * (ZH))                 \
                    - r_lower * hv * (ZU);                                       \
                local = fmaf((WEIGHT), action, local);                           \
                (ZU) = fmaf(uv, input_value, (ZU));                              \
                (ZH) = fmaf(hv, input_value, (ZH));                              \
            }                                                                    \
        } while (0)
        APPLY_REVERSE(source0, weight0, zu0, zh0);
        APPLY_REVERSE(source1, weight1, zu1, zh1);
#undef APPLY_REVERSE
        const float action = 0.5f * reduce_group8(local);
        if (lane == 0) {
            boundary_action[vector_base + target * kRank + coordinate] += action;
        }
    }
    if (lane == 0) {
        FloatFloat boundary[4] = {};
#pragma unroll
        for (int block = 0; block < kBoundaryBlocks; ++block) {
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                boundary[component] = ff_add(boundary[component], {
                    boundary_qbar[
                        ((panel * kBoundaryBlocks + block) * kChunk + target) * 4
                        + component],
                    0.0f});
            }
        }
        const int output =
            (panel * kChunk + target) * kQbarComponents * 2;
#pragma unroll
        for (int component = 0; component < kQbarComponents; ++component) {
            qbar_parts[output + component * 2 + 0] = boundary[component].hi;
            qbar_parts[output + component * 2 + 1] = boundary[component].lo;
        }
    }
    __syncwarp();
    for (int coordinate = lane; coordinate < kRank; coordinate += 8) {
        const int index = vector_base + target * kRank + coordinate;
        grad_key[index] = grad_key_direct[index] - boundary_action[index];
    }
}

std::tuple<at::Tensor, at::Tensor> omega_transpose_local_cuda(
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& weights,
    const at::Tensor& coefficient,
    const at::Tensor& input,
    at::Tensor& boundary_action,
    const at::Tensor& boundary_qbar,
    const at::Tensor& grad_key_direct) {
    const int panels = static_cast<int>(u.size(0));
    auto grad_key = at::empty_like(input);
    auto qbar_parts = at::empty(
        {panels, kChunk, kQbarComponents, 2}, input.options());
    c10::cuda::CUDAGuard guard(u.device());
    omega_transpose_local_kernel<<<
        panels, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        u.data_ptr<float>(), h.data_ptr<float>(), weights.data_ptr<float>(),
        coefficient.data_ptr<float>(), input.data_ptr<float>(),
        boundary_action.data_ptr<float>(), boundary_qbar.data_ptr<float>(),
        grad_key_direct.data_ptr<float>(), grad_key.data_ptr<float>(),
        qbar_parts.data_ptr<float>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_key, qbar_parts};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "packet_frame_omega_vjp128(Tensor u, Tensor h, Tensor weights, Tensor coefficient, "
        "Tensor input, Tensor(a!) boundary_action, "
        "Tensor boundary_qbar, Tensor grad_key_direct) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("packet_frame_omega_vjp128", &omega_transpose_local_cuda);
}
