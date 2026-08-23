#include "solvedelta_c32.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <tuple>


namespace {

constexpr int kRank = 128;
constexpr int kChunk = 32;
constexpr int kComponents = 4;
constexpr float kRadius = 1.0f / 8.0f;

}  // namespace


namespace {

constexpr int kCompactRoutes = 3;
constexpr int kCompactWarps = 8;

template <bool Upper>
__device__ __forceinline__ void warp_strict4(
    const float value[4],
    float strict[4]) {
    float inclusive[4];
    inclusive[0] = value[0];
    inclusive[1] = inclusive[0] + value[1];
    inclusive[2] = inclusive[1] + value[2];
    inclusive[3] = inclusive[2] + value[3];
    float chunk_prefix = inclusive[3];
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float previous = __shfl_up_sync(
            0xffffffffu, chunk_prefix, offset);
        if (lane >= offset) chunk_prefix += previous;
    }
    const float before = chunk_prefix - inclusive[3];
    const float total = __shfl_sync(0xffffffffu, chunk_prefix, 31);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const float prefix = before + (item == 0 ? 0.0f : inclusive[item - 1]);
        const float including = before + inclusive[item];
        strict[item] = Upper ? total - including : prefix;
    }
}

__device__ __forceinline__ float warp_sum4(const float value[4]) {
    float result = value[0] + value[1] + value[2] + value[3];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        result += __shfl_down_sync(0xffffffffu, result, offset);
    }
    return result;
}

template <bool Upper>
__global__ void compact_pair_stats_kernel(
    const at::BFloat16* __restrict__ descriptor_left,
    const at::BFloat16* __restrict__ descriptor_right,
    const at::BFloat16* __restrict__ local_u,
    const at::BFloat16* __restrict__ local_h,
    float* __restrict__ descriptor_j,
    float* __restrict__ descriptor_d,
    float* __restrict__ gram_j,
    float* __restrict__ gram_d,
    int pairs) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int pair = blockIdx.x * kCompactWarps + warp;
    if (pair >= pairs) return;
    const int panel = pair / (kChunk * kChunk);
    const int local_pair = pair % (kChunk * kChunk);
    const int target = local_pair / kChunk;
    const int source = local_pair % kChunk;
    float source_u[4];
    float source_h[4];
    float target_u[4];
    float target_h[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int coordinate = lane * 4 + item;
        source_u[item] = static_cast<float>(
            local_u[(panel * kChunk + source) * kRank + coordinate]);
        source_h[item] = static_cast<float>(
            local_h[(panel * kChunk + source) * kRank + coordinate]);
        target_u[item] = static_cast<float>(
            local_u[(panel * kChunk + target) * kRank + coordinate]);
        target_h[item] = static_cast<float>(
            local_h[(panel * kChunk + target) * kRank + coordinate]);
    }
    float descriptor_j_value[4] = {};
    float descriptor_d_value[4] = {};
#pragma unroll
    for (int route = 0; route < kCompactRoutes; ++route) {
        float left[4];
        float right_u[4];
        float right_h[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int coordinate = lane * 4 + item;
            const int index =
                ((panel * kChunk + target) * kCompactRoutes + route)
                    * kRank + coordinate;
            const float descriptor_l =
                static_cast<float>(descriptor_left[index]);
            const float descriptor_r =
                static_cast<float>(descriptor_right[index]);
            left[item] = descriptor_l * source_u[item];
            right_u[item] = descriptor_r * source_u[item];
            right_h[item] = descriptor_r * source_h[item];
        }
        float strict_u[4];
        float strict_h[4];
        warp_strict4<Upper>(right_u, strict_u);
        warp_strict4<Upper>(right_h, strict_h);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            descriptor_j_value[item] = fmaf(
                left[item], strict_u[item], descriptor_j_value[item]);
            descriptor_d_value[item] = fmaf(
                left[item], strict_h[item], descriptor_d_value[item]);
        }
    }
    float right_j[4];
    float right_d[4];
    float left_local[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        left_local[item] = target_u[item] * source_u[item];
        right_j[item] = target_u[item] * source_u[item];
        right_d[item] = target_h[item] * source_h[item];
    }
    float strict_j[4];
    float strict_d[4];
    warp_strict4<Upper>(right_j, strict_j);
    warp_strict4<Upper>(right_d, strict_d);
    float gram_j_value[4];
    float gram_d_value[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        gram_j_value[item] = left_local[item] * strict_j[item];
        gram_d_value[item] = left_local[item] * strict_d[item];
    }
    const float result_descriptor_j = warp_sum4(descriptor_j_value);
    const float result_descriptor_d = warp_sum4(descriptor_d_value);
    const float result_gram_j = warp_sum4(gram_j_value);
    const float result_gram_d = warp_sum4(gram_d_value);
    if (lane == 0) {
        descriptor_j[pair] = result_descriptor_j;
        descriptor_d[pair] = result_descriptor_d;
        gram_j[pair] = result_gram_j;
        gram_d[pair] = result_gram_d;
    }
}

template <bool Upper>
__global__ void compact_leaf_kernel(
    const at::BFloat16* __restrict__ descriptor_left,
    const at::BFloat16* __restrict__ descriptor_right,
    const float* __restrict__ direct_j,
    const float* __restrict__ direct_d,
    const float* __restrict__ mix_j,
    const float* __restrict__ mix_d,
    const at::BFloat16* __restrict__ local_u,
    const at::BFloat16* __restrict__ local_h,
    const float* __restrict__ boundary_j_forward,
    const float* __restrict__ boundary_j_transpose,
    const float* __restrict__ boundary_d_forward,
    const float* __restrict__ boundary_d_transpose,
    float* __restrict__ grad_u,
    float* __restrict__ grad_h,
    int64_t forward_stride_0,
    int64_t forward_stride_1,
    int64_t forward_stride_2,
    int64_t transpose_stride_0,
    int64_t transpose_stride_1,
    int64_t transpose_stride_2,
    int items) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int target_item = blockIdx.x * kCompactWarps + warp;
    if (target_item >= items) return;
    const int panel = target_item / kChunk;
    const int target = target_item % kChunk;
    float target_u[4];
    float target_h[4];
    float output_u[4];
    float output_h[4];
    const int mix_boundary =
        (panel * (kChunk + 1) + target + 1) * (kChunk + 1);
    const float boundary_scale_j = mix_j[mix_boundary];
    const float boundary_scale_d = mix_d[mix_boundary];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int coordinate = lane * 4 + item;
        const int vector = target_item * kRank + coordinate;
        const int64_t forward =
            panel * forward_stride_0
            + target * forward_stride_1
            + coordinate * forward_stride_2;
        const int64_t transpose =
            panel * transpose_stride_0
            + target * transpose_stride_1
            + coordinate * transpose_stride_2;
        target_u[item] = static_cast<float>(local_u[vector]);
        target_h[item] = static_cast<float>(local_h[vector]);
        output_u[item] = boundary_scale_j
                * (boundary_j_forward[forward]
                   + boundary_j_transpose[transpose])
            + boundary_scale_d * boundary_d_forward[forward];
        output_h[item] =
            boundary_scale_d * boundary_d_transpose[transpose];
    }
#pragma unroll 1
    for (int source = 0; source < kChunk; ++source) {
        float descriptor_scale_j = 0.0f;
        float descriptor_scale_d = 0.0f;
        float local_scale_j = 0.0f;
        float local_scale_d = 0.0f;
        if (lane == 0) {
            const int direct =
                (panel * (kChunk + 1) + target + 1) * kChunk + source;
            const int mix =
                (panel * (kChunk + 1) + target + 1) * (kChunk + 1)
                + source + 1;
            descriptor_scale_j = direct_j[direct];
            descriptor_scale_d = direct_d[direct];
            local_scale_j = mix_j[mix];
            local_scale_d = mix_d[mix];
        }
        descriptor_scale_j = __shfl_sync(
            0xffffffffu, descriptor_scale_j, 0);
        descriptor_scale_d = __shfl_sync(
            0xffffffffu, descriptor_scale_d, 0);
        local_scale_j = __shfl_sync(0xffffffffu, local_scale_j, 0);
        local_scale_d = __shfl_sync(0xffffffffu, local_scale_d, 0);
#pragma unroll
        for (int route = 0; route < kCompactRoutes; ++route) {
            float descriptor_l[4];
            float descriptor_r[4];
            float right_u[4];
            float right_h[4];
            float left_u[4];
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int coordinate = lane * 4 + item;
                const int index =
                    ((panel * kChunk + source) * kCompactRoutes + route)
                        * kRank + coordinate;
                descriptor_l[item] =
                    static_cast<float>(descriptor_left[index]);
                descriptor_r[item] =
                    static_cast<float>(descriptor_right[index]);
                right_u[item] = descriptor_r[item] * target_u[item];
                right_h[item] = descriptor_r[item] * target_h[item];
                left_u[item] = descriptor_l[item] * target_u[item];
            }
            float strict_right_u[4];
            float strict_right_h[4];
            float strict_left_u[4];
            warp_strict4<Upper>(right_u, strict_right_u);
            warp_strict4<Upper>(right_h, strict_right_h);
            warp_strict4<!Upper>(left_u, strict_left_u);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                output_u[item] = fmaf(
                    descriptor_scale_j * descriptor_l[item],
                    strict_right_u[item],
                    output_u[item]);
                output_u[item] = fmaf(
                    descriptor_scale_j * descriptor_r[item],
                    strict_left_u[item],
                    output_u[item]);
                output_u[item] = fmaf(
                    descriptor_scale_d * descriptor_l[item],
                    strict_right_h[item],
                    output_u[item]);
                output_h[item] = fmaf(
                    descriptor_scale_d * descriptor_r[item],
                    strict_left_u[item],
                    output_h[item]);
            }
        }
        float source_u[4];
        float source_h[4];
        float right_u[4];
        float right_h[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int coordinate = lane * 4 + item;
            source_u[item] = static_cast<float>(
                local_u[(panel * kChunk + source) * kRank + coordinate]);
            source_h[item] = static_cast<float>(
                local_h[(panel * kChunk + source) * kRank + coordinate]);
            right_u[item] = source_u[item] * target_u[item];
            right_h[item] = source_h[item] * target_h[item];
        }
        float strict_right_u[4];
        float strict_right_h[4];
        float strict_left_u[4];
        warp_strict4<Upper>(right_u, strict_right_u);
        warp_strict4<Upper>(right_h, strict_right_h);
        warp_strict4<!Upper>(right_u, strict_left_u);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            output_u[item] = fmaf(
                local_scale_j * source_u[item],
                strict_right_u[item] + strict_left_u[item],
                output_u[item]);
            output_u[item] = fmaf(
                local_scale_d * source_u[item],
                strict_right_h[item],
                output_u[item]);
            output_h[item] = fmaf(
                local_scale_d * source_h[item],
                strict_left_u[item],
                output_h[item]);
        }
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int coordinate = lane * 4 + item;
        const int vector = target_item * kRank + coordinate;
        grad_u[vector] = output_u[item];
        grad_h[vector] = output_h[item];
    }
}

}  // namespace

using CompactPairResult = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

CompactPairResult c32_frame_compact_pair_cuda(
    const at::Tensor& descriptor_left,
    const at::Tensor& descriptor_right,
    const at::Tensor& local_u,
    const at::Tensor& local_h,
    bool upper) {
    TORCH_CHECK(
        descriptor_left.is_cuda()
            && descriptor_left.scalar_type() == at::kBFloat16
            && descriptor_left.is_contiguous(),
        "descriptor_left must be contiguous BF16 CUDA");
    TORCH_CHECK(
        descriptor_right.sizes() == descriptor_left.sizes()
            && descriptor_right.scalar_type() == at::kBFloat16
            && descriptor_right.is_contiguous(),
        "descriptor_right mismatch");
    TORCH_CHECK(
        descriptor_left.dim() == 4
            && descriptor_left.size(1) == kChunk
            && descriptor_left.size(2) == kCompactRoutes
            && descriptor_left.size(3) == kRank,
        "descriptor tensors must be [P,32,3,128]");
    const int64_t panels = descriptor_left.size(0);
    TORCH_CHECK(
        local_u.sizes() == at::IntArrayRef({panels, kChunk, kRank})
            && local_h.sizes() == local_u.sizes()
            && local_u.scalar_type() == at::kBFloat16
            && local_h.scalar_type() == at::kBFloat16
            && local_u.is_contiguous()
            && local_h.is_contiguous(),
        "local_u/h must be contiguous BF16 [P,32,128]");
    const auto device = descriptor_left.device();
    for (const auto& tensor : {
             &descriptor_right, &local_u, &local_h}) {
        TORCH_CHECK(
            tensor->is_cuda() && tensor->device() == device,
            "compact pair inputs must share one CUDA device");
    }
    c10::cuda::CUDAGuard guard(device);
    auto options = descriptor_left.options().dtype(at::kFloat);
    auto descriptor_j = at::empty({panels, kChunk, kChunk}, options);
    auto descriptor_d = at::empty_like(descriptor_j);
    auto gram_j = at::empty_like(descriptor_j);
    auto gram_d = at::empty_like(descriptor_j);
    const int pairs = static_cast<int>(panels * kChunk * kChunk);
    const int blocks = (pairs + kCompactWarps - 1) / kCompactWarps;
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (upper) {
        compact_pair_stats_kernel<true><<<blocks, kCompactWarps * 32, 0, stream>>>(
            descriptor_left.data_ptr<at::BFloat16>(),
            descriptor_right.data_ptr<at::BFloat16>(),
            local_u.data_ptr<at::BFloat16>(),
            local_h.data_ptr<at::BFloat16>(),
            descriptor_j.data_ptr<float>(),
            descriptor_d.data_ptr<float>(),
            gram_j.data_ptr<float>(),
            gram_d.data_ptr<float>(),
            pairs);
    } else {
        compact_pair_stats_kernel<false><<<blocks, kCompactWarps * 32, 0, stream>>>(
            descriptor_left.data_ptr<at::BFloat16>(),
            descriptor_right.data_ptr<at::BFloat16>(),
            local_u.data_ptr<at::BFloat16>(),
            local_h.data_ptr<at::BFloat16>(),
            descriptor_j.data_ptr<float>(),
            descriptor_d.data_ptr<float>(),
            gram_j.data_ptr<float>(),
            gram_d.data_ptr<float>(),
            pairs);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {descriptor_j, descriptor_d, gram_j, gram_d};
}

using CompactLeafResult = std::tuple<at::Tensor, at::Tensor>;

CompactLeafResult c32_frame_compact_leaf_cuda(
    const at::Tensor& descriptor_left,
    const at::Tensor& descriptor_right,
    const at::Tensor& direct_j,
    const at::Tensor& direct_d,
    const at::Tensor& mix_j,
    const at::Tensor& mix_d,
    const at::Tensor& local_u,
    const at::Tensor& local_h,
    const at::Tensor& boundary_j_forward,
    const at::Tensor& boundary_j_transpose,
    const at::Tensor& boundary_d_forward,
    const at::Tensor& boundary_d_transpose,
    bool upper) {
    const int64_t panels = descriptor_left.size(0);
    TORCH_CHECK(
        descriptor_left.is_cuda()
            && descriptor_left.scalar_type() == at::kBFloat16
            && descriptor_left.is_contiguous()
            && descriptor_right.sizes() == descriptor_left.sizes()
            && descriptor_right.scalar_type() == at::kBFloat16
            && descriptor_right.is_contiguous(),
        "compact leaf descriptors must be contiguous BF16");
    for (const auto& tensor : {&direct_j, &direct_d}) {
        TORCH_CHECK(
            tensor->sizes()
                    == at::IntArrayRef({panels, kChunk + 1, kChunk})
                && tensor->scalar_type() == at::kFloat
                && tensor->is_contiguous(),
            "compact leaf direct weights must be contiguous FP32 [P,33,32]");
    }
    for (const auto& tensor : {&mix_j, &mix_d}) {
        TORCH_CHECK(
            tensor->sizes()
                    == at::IntArrayRef({panels, kChunk + 1, kChunk + 1})
                && tensor->scalar_type() == at::kFloat
                && tensor->is_contiguous(),
            "compact leaf mix weights must be contiguous FP32 [P,33,33]");
    }
    for (const auto& tensor : {
             &boundary_j_forward,
             &boundary_j_transpose,
             &boundary_d_forward,
             &boundary_d_transpose}) {
        TORCH_CHECK(
            tensor->sizes() == at::IntArrayRef({panels, kChunk, kRank})
                && tensor->scalar_type() == at::kFloat,
            "compact leaf boundary actions must be FP32 [P,32,128]");
    }
    TORCH_CHECK(
        boundary_j_forward.strides() == boundary_d_forward.strides()
            && boundary_j_transpose.strides()
                == boundary_d_transpose.strides(),
        "J/D boundary action strides must match");
    TORCH_CHECK(
        local_u.sizes() == at::IntArrayRef({panels, kChunk, kRank})
            && local_h.sizes() == local_u.sizes()
            && local_u.scalar_type() == at::kBFloat16
            && local_h.scalar_type() == at::kBFloat16
            && local_u.is_contiguous()
            && local_h.is_contiguous(),
        "compact leaf local tensors must be contiguous BF16");
    const auto device = descriptor_left.device();
    for (const auto& tensor : {
             &descriptor_right,
             &direct_j,
             &direct_d,
             &mix_j,
             &mix_d,
             &local_u,
             &local_h,
             &boundary_j_forward,
             &boundary_j_transpose,
             &boundary_d_forward,
             &boundary_d_transpose}) {
        TORCH_CHECK(
            tensor->is_cuda() && tensor->device() == device,
            "compact leaf inputs must share one CUDA device");
    }
    c10::cuda::CUDAGuard guard(device);
    auto grad_u = at::empty(
        {panels, kChunk, kRank}, direct_j.options());
    auto grad_h = at::empty_like(grad_u);
    const int items = static_cast<int>(panels * kChunk);
    const int blocks = (items + kCompactWarps - 1) / kCompactWarps;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t forward_stride_0 = boundary_j_forward.stride(0);
    const int64_t forward_stride_1 = boundary_j_forward.stride(1);
    const int64_t forward_stride_2 = boundary_j_forward.stride(2);
    const int64_t transpose_stride_0 = boundary_j_transpose.stride(0);
    const int64_t transpose_stride_1 = boundary_j_transpose.stride(1);
    const int64_t transpose_stride_2 = boundary_j_transpose.stride(2);
    if (upper) {
        compact_leaf_kernel<true><<<blocks, kCompactWarps * 32, 0, stream>>>(
            descriptor_left.data_ptr<at::BFloat16>(),
            descriptor_right.data_ptr<at::BFloat16>(),
            direct_j.data_ptr<float>(), direct_d.data_ptr<float>(),
            mix_j.data_ptr<float>(), mix_d.data_ptr<float>(),
            local_u.data_ptr<at::BFloat16>(), local_h.data_ptr<at::BFloat16>(),
            boundary_j_forward.data_ptr<float>(),
            boundary_j_transpose.data_ptr<float>(),
            boundary_d_forward.data_ptr<float>(),
            boundary_d_transpose.data_ptr<float>(),
            grad_u.data_ptr<float>(), grad_h.data_ptr<float>(),
            forward_stride_0, forward_stride_1, forward_stride_2,
            transpose_stride_0, transpose_stride_1, transpose_stride_2,
            items);
    } else {
        compact_leaf_kernel<false><<<blocks, kCompactWarps * 32, 0, stream>>>(
            descriptor_left.data_ptr<at::BFloat16>(),
            descriptor_right.data_ptr<at::BFloat16>(),
            direct_j.data_ptr<float>(), direct_d.data_ptr<float>(),
            mix_j.data_ptr<float>(), mix_d.data_ptr<float>(),
            local_u.data_ptr<at::BFloat16>(), local_h.data_ptr<at::BFloat16>(),
            boundary_j_forward.data_ptr<float>(),
            boundary_j_transpose.data_ptr<float>(),
            boundary_d_forward.data_ptr<float>(),
            boundary_d_transpose.data_ptr<float>(),
            grad_u.data_ptr<float>(), grad_h.data_ptr<float>(),
            forward_stride_0, forward_stride_1, forward_stride_2,
            transpose_stride_0, transpose_stride_1, transpose_stride_2,
            items);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_u, grad_h};
}


namespace {

constexpr int kCoefficientThreads = 256;
constexpr int kCoefficientBasis = kChunk + 1;
constexpr int kDescriptorRoutes = 3;

struct CompactCoefficientShared {
    float temporal[kChunk * kCoefficientBasis];
    float projection[kChunk * kCoefficientBasis];
    float gram[kCoefficientBasis * kCoefficientBasis];
    float p[kChunk];
    float rho[kChunk];
    float reduction[kCoefficientThreads];
};

static_assert(sizeof(CompactCoefficientShared) == 14084);

__device__ __forceinline__ float coefficient_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <bool Upper>
__global__ __launch_bounds__(kCoefficientThreads, 2)
void compact_coefficient_kernel(
    const at::BFloat16* __restrict__ descriptor_left,
    const at::BFloat16* __restrict__ local_u,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ action_j,
    const float* __restrict__ action_d,
    int64_t action_stride_0,
    int64_t action_stride_1,
    int64_t action_stride_2,
    const float* __restrict__ descriptor_j,
    const float* __restrict__ descriptor_d,
    const float* __restrict__ gram_j,
    const float* __restrict__ gram_d,
    const float* __restrict__ temporal,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ radial_scale,
    const float* __restrict__ radial_q2,
    const float* __restrict__ strength,
    float* __restrict__ direct_j,
    float* __restrict__ direct_d,
    float* __restrict__ mix_j,
    float* __restrict__ mix_d,
    float* __restrict__ grad_temporal,
    float* __restrict__ grad_strength,
    int panels) {
    __shared__ CompactCoefficientShared shared;
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int temporal_base = panel * kChunk * kCoefficientBasis;
    for (int index = tid;
         index < kChunk * kCoefficientBasis;
         index += blockDim.x) {
        shared.temporal[index] = temporal[temporal_base + index];
        grad_temporal[temporal_base + index] = 0.0f;
    }
    if (tid == 0) grad_strength[panel] = 0.0f;
    __syncthreads();

#pragma unroll
    for (int channel = 0; channel < 2; ++channel) {
        const float* boundary = channel == 0 ? boundary_j : boundary_d;
        const float* action = channel == 0 ? action_j : action_d;
        const float* descriptor = channel == 0 ? descriptor_j : descriptor_d;
        const float* local_gram = channel == 0 ? gram_j : gram_d;
        const int component = (Upper ? 2 : 0) + channel;
        float* direct = channel == 0 ? direct_j : direct_d;
        float* mix = channel == 0 ? mix_j : mix_d;
        const int matrix_base = panel * kRank * kRank;

        float norm = 0.0f;
        for (int entry = tid; entry < kRank * kRank; entry += blockDim.x) {
            const int row = entry / kRank;
            const int column = entry % kRank;
            const bool active = Upper ? column > row : row > column;
            const float value = active ? boundary[matrix_base + entry] : 0.0f;
            norm = fmaf(value, value, norm);
        }
        shared.reduction[tid] = norm;
        __syncthreads();
#pragma unroll
        for (int offset = kCoefficientThreads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                shared.reduction[tid] += shared.reduction[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) shared.gram[0] = shared.reduction[0];

#pragma unroll
        for (int group = 0; group < kChunk / 8; ++group) {
            const int target = group * 8 + warp;
            float descriptor_inner = 0.0f;
            float local_inner = 0.0f;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int coordinate = lane * 4 + item;
#pragma unroll
                for (int route = 0; route < kDescriptorRoutes; ++route) {
                    const int descriptor_index =
                        ((panel * kChunk + target) * kDescriptorRoutes + route)
                            * kRank + coordinate;
                    const int action_vector = target * kDescriptorRoutes + route;
                    const int64_t action_index =
                        panel * action_stride_0
                        + action_vector * action_stride_1
                        + coordinate * action_stride_2;
                    descriptor_inner = fmaf(
                        static_cast<float>(descriptor_left[descriptor_index]),
                        action[action_index],
                        descriptor_inner);
                }
                const int local_index =
                    (panel * kChunk + target) * kRank + coordinate;
                const int64_t action_index =
                    panel * action_stride_0
                    + (kChunk * kDescriptorRoutes + target) * action_stride_1
                    + coordinate * action_stride_2;
                local_inner = fmaf(
                    static_cast<float>(local_u[local_index]),
                    action[action_index],
                    local_inner);
            }
            descriptor_inner = coefficient_warp_sum(descriptor_inner);
            local_inner = coefficient_warp_sum(local_inner);
            if (lane == 0) {
                shared.projection[target * kCoefficientBasis] =
                    descriptor_inner;
                shared.gram[target + 1] = local_inner;
                shared.gram[(target + 1) * kCoefficientBasis] = local_inner;
            }
        }
        __syncthreads();

        for (int index = tid; index < kChunk * kChunk; index += blockDim.x) {
            const int target = index / kChunk;
            const int source = index % kChunk;
            shared.projection[
                target * kCoefficientBasis + source + 1] =
                descriptor[panel * kChunk * kChunk + index];
            shared.gram[
                (target + 1) * kCoefficientBasis + source + 1] =
                local_gram[panel * kChunk * kChunk + index];
        }
        __syncthreads();

        if (tid < kChunk) {
            const int target = tid;
            float zeta = 0.0f;
#pragma unroll
            for (int basis = 0; basis < kCoefficientBasis; ++basis) {
                zeta = fmaf(
                    shared.temporal[target * kCoefficientBasis + basis],
                    shared.projection[target * kCoefficientBasis + basis],
                    zeta);
            }
            const int scalar_index =
                (panel * kChunk + target) * kComponents + component;
            const float local_scale = radial_scale[scalar_index];
            const float local_q2 = radial_q2[scalar_index];
            const float panel_strength = strength[panel];
            shared.p[target] =
                -local_scale * panel_strength * panel_strength / local_q2 * zeta;
            shared.reduction[target] =
                kRadius * kRadius * kRadius * zeta
                / powf(local_q2, 1.5f);
        }
        __syncthreads();
        if (tid == 0) {
            float next_rho = 0.0f;
            for (int reverse = 0; reverse < kChunk; ++reverse) {
                const int target = kChunk - 1 - reverse;
                if (target + 1 < kChunk) {
                    const float beta =
                        1.0f - inverse_mass[panel * kChunk + target + 1];
                    next_rho = shared.p[target] + beta * beta * next_rho;
                } else {
                    next_rho = shared.p[target];
                }
                shared.rho[target] = next_rho;
            }
            float strength_sum = 0.0f;
#pragma unroll
            for (int target = 0; target < kChunk; ++target) {
                strength_sum += shared.reduction[target];
            }
            grad_strength[panel] += strength_sum;
        }
        __syncthreads();

        for (int index = tid;
             index < kCoefficientBasis * kChunk;
             index += blockDim.x) {
            const int basis = index / kChunk;
            const int target = index % kChunk;
            direct[panel * kCoefficientBasis * kChunk + index] =
                shared.temporal[target * kCoefficientBasis + basis]
                * radial_scale[
                    (panel * kChunk + target) * kComponents + component];
        }
        for (int index = tid;
             index < kCoefficientBasis * kCoefficientBasis;
             index += blockDim.x) {
            const int left = index / kCoefficientBasis;
            const int right = index % kCoefficientBasis;
            float value;
            if (left == 0 && right == 0) {
                const float theta0 = shared.temporal[0];
                value = theta0 * theta0 * shared.rho[0];
            } else if (left == 0 || right == 0) {
                const int local = (left == 0 ? right : left) - 1;
                value =
                    shared.temporal[local * kCoefficientBasis]
                    * inverse_mass[panel * kChunk + local]
                    * shared.rho[local];
            } else {
                const int left_local = left - 1;
                const int right_local = right - 1;
                const int later = max(left_local, right_local);
                const int earlier = min(left_local, right_local);
                value =
                    shared.temporal[
                        later * kCoefficientBasis + earlier + 1]
                    * inverse_mass[panel * kChunk + later]
                    * shared.rho[later];
            }
            mix[panel * kCoefficientBasis * kCoefficientBasis + index] = value;
        }
        for (int index = tid;
             index < kChunk * kCoefficientBasis;
             index += blockDim.x) {
            const int target = index / kCoefficientBasis;
            const int basis = index % kCoefficientBasis;
            float basis_action = 0.0f;
#pragma unroll
            for (int source = 0; source < kCoefficientBasis; ++source) {
                basis_action = fmaf(
                    shared.temporal[target * kCoefficientBasis + source],
                    shared.gram[source * kCoefficientBasis + basis],
                    basis_action);
            }
            const float value =
                radial_scale[
                    (panel * kChunk + target) * kComponents + component]
                    * shared.projection[target * kCoefficientBasis + basis]
                + shared.p[target] * basis_action;
            grad_temporal[temporal_base + index] += value;
        }
        __syncthreads();
    }
}

}  // namespace


using CompactCoefficientResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

CompactCoefficientResult c32_frame_compact_coefficients_cuda(
    const at::Tensor& descriptor_left,
    const at::Tensor& local_u,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& action_j,
    const at::Tensor& action_d,
    const at::Tensor& descriptor_j,
    const at::Tensor& descriptor_d,
    const at::Tensor& gram_j,
    const at::Tensor& gram_d,
    const at::Tensor& temporal,
    const at::Tensor& inverse_mass,
    const at::Tensor& radial_scale,
    const at::Tensor& radial_q2,
    const at::Tensor& strength,
    bool upper) {
    TORCH_CHECK(
        descriptor_left.is_cuda()
            && descriptor_left.scalar_type() == at::kBFloat16
            && descriptor_left.is_contiguous()
            && descriptor_left.dim() == 4
            && descriptor_left.size(1) == kChunk
            && descriptor_left.size(2) == kDescriptorRoutes
            && descriptor_left.size(3) == kRank,
        "descriptor_left must be contiguous BF16 [P,32,3,128]");
    const int64_t panels = descriptor_left.size(0);
    TORCH_CHECK(
        local_u.sizes() == at::IntArrayRef({panels, kChunk, kRank})
            && local_u.scalar_type() == at::kBFloat16
            && local_u.is_contiguous(),
        "local_u must be contiguous BF16 [P,32,128]");
    for (const auto& tensor : {&boundary_j, &boundary_d}) {
        TORCH_CHECK(
            tensor->sizes() == at::IntArrayRef({panels, kRank, kRank})
                && tensor->scalar_type() == at::kFloat
                && tensor->is_contiguous(),
            "coefficient boundaries must be contiguous FP32 [P,128,128]");
    }
    for (const auto& tensor : {&action_j, &action_d}) {
        TORCH_CHECK(
            tensor->sizes()
                    == at::IntArrayRef({panels, kChunk * kDescriptorRoutes + kChunk, kRank})
                && tensor->scalar_type() == at::kFloat,
            "boundary actions must be FP32 [P,128,128]");
    }
    TORCH_CHECK(
        action_j.strides() == action_d.strides(),
        "J/D boundary action strides must match");
    for (const auto& tensor : {
             &descriptor_j, &descriptor_d, &gram_j, &gram_d}) {
        TORCH_CHECK(
            tensor->sizes() == at::IntArrayRef({panels, kChunk, kChunk})
                && tensor->scalar_type() == at::kFloat
                && tensor->is_contiguous(),
            "compact pair statistics must be contiguous FP32 [P,32,32]");
    }
    TORCH_CHECK(
        temporal.sizes()
                == at::IntArrayRef({panels, kChunk, kCoefficientBasis})
            && temporal.scalar_type() == at::kFloat
            && temporal.is_contiguous(),
        "temporal must be contiguous FP32 [P,32,33]");
    TORCH_CHECK(
        inverse_mass.sizes() == at::IntArrayRef({panels, kChunk})
            && inverse_mass.scalar_type() == at::kFloat
            && inverse_mass.is_contiguous(),
        "inverse_mass must be contiguous FP32 [P,32]");
    for (const auto& tensor : {&radial_scale, &radial_q2}) {
        TORCH_CHECK(
            tensor->sizes()
                    == at::IntArrayRef({panels, kChunk, kComponents})
                && tensor->scalar_type() == at::kFloat
                && tensor->is_contiguous(),
            "radial scalars must be contiguous FP32 [P,32,4]");
    }
    TORCH_CHECK(
        strength.sizes() == at::IntArrayRef({panels})
            && strength.scalar_type() == at::kFloat
            && strength.is_contiguous(),
        "strength must be contiguous FP32 [P]");
    const auto device = descriptor_left.device();
    for (const auto& tensor : {
             &local_u,
             &boundary_j,
             &boundary_d,
             &action_j,
             &action_d,
             &descriptor_j,
             &descriptor_d,
             &gram_j,
             &gram_d,
             &temporal,
             &inverse_mass,
             &radial_scale,
             &radial_q2,
             &strength}) {
        TORCH_CHECK(
            tensor->is_cuda() && tensor->device() == device,
            "compact coefficient inputs must share one CUDA device");
    }
    TORCH_CHECK(
        panels <= std::numeric_limits<int>::max(),
        "panel count must fit int32");

    c10::cuda::CUDAGuard guard(device);
    auto options = temporal.options();
    auto direct_j = at::empty({panels, kCoefficientBasis, kChunk}, options);
    auto direct_d = at::empty_like(direct_j);
    auto mix_j = at::empty(
        {panels, kCoefficientBasis, kCoefficientBasis}, options);
    auto mix_d = at::empty_like(mix_j);
    auto grad_temporal = at::empty_like(temporal);
    auto grad_strength = at::empty({panels}, options);
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (upper) {
        compact_coefficient_kernel<true><<<
            static_cast<int>(panels), kCoefficientThreads, 0, stream>>>(
            descriptor_left.data_ptr<at::BFloat16>(),
            local_u.data_ptr<at::BFloat16>(),
            boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
            action_j.data_ptr<float>(), action_d.data_ptr<float>(),
            action_j.stride(0), action_j.stride(1), action_j.stride(2),
            descriptor_j.data_ptr<float>(), descriptor_d.data_ptr<float>(),
            gram_j.data_ptr<float>(), gram_d.data_ptr<float>(),
            temporal.data_ptr<float>(), inverse_mass.data_ptr<float>(),
            radial_scale.data_ptr<float>(), radial_q2.data_ptr<float>(),
            strength.data_ptr<float>(),
            direct_j.data_ptr<float>(), direct_d.data_ptr<float>(),
            mix_j.data_ptr<float>(), mix_d.data_ptr<float>(),
            grad_temporal.data_ptr<float>(), grad_strength.data_ptr<float>(),
            static_cast<int>(panels));
    } else {
        compact_coefficient_kernel<false><<<
            static_cast<int>(panels), kCoefficientThreads, 0, stream>>>(
            descriptor_left.data_ptr<at::BFloat16>(),
            local_u.data_ptr<at::BFloat16>(),
            boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
            action_j.data_ptr<float>(), action_d.data_ptr<float>(),
            action_j.stride(0), action_j.stride(1), action_j.stride(2),
            descriptor_j.data_ptr<float>(), descriptor_d.data_ptr<float>(),
            gram_j.data_ptr<float>(), gram_d.data_ptr<float>(),
            temporal.data_ptr<float>(), inverse_mass.data_ptr<float>(),
            radial_scale.data_ptr<float>(), radial_q2.data_ptr<float>(),
            strength.data_ptr<float>(),
            direct_j.data_ptr<float>(), direct_d.data_ptr<float>(),
            mix_j.data_ptr<float>(), mix_d.data_ptr<float>(),
            grad_temporal.data_ptr<float>(), grad_strength.data_ptr<float>(),
            static_cast<int>(panels));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {direct_j, direct_d, mix_j, mix_d, grad_temporal, grad_strength};
}


TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def("c32_frame_compact_pair(Tensor descriptor_left, Tensor descriptor_right, Tensor local_u, Tensor local_h, bool upper) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("c32_frame_compact_leaf(Tensor descriptor_left, Tensor descriptor_right, Tensor direct_J, Tensor direct_D, Tensor mix_J, Tensor mix_D, Tensor local_u, Tensor local_h, Tensor boundary_J_forward, Tensor boundary_J_transpose, Tensor boundary_D_forward, Tensor boundary_D_transpose, bool upper) -> (Tensor, Tensor)");
    m.def("c32_frame_compact_coefficients(Tensor descriptor_left, Tensor local_u, Tensor boundary_J, Tensor boundary_D, Tensor action_J, Tensor action_D, Tensor descriptor_J, Tensor descriptor_D, Tensor gram_J, Tensor gram_D, Tensor temporal, Tensor inverse_mass, Tensor radial_scale, Tensor radial_q2, Tensor strength, bool upper) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}


TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("c32_frame_compact_pair", &c32_frame_compact_pair_cuda);
    m.impl("c32_frame_compact_leaf", &c32_frame_compact_leaf_cuda);
    m.impl("c32_frame_compact_coefficients", &c32_frame_compact_coefficients_cuda);
}
