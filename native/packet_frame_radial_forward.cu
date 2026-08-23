#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <limits>
#include <tuple>

namespace {

constexpr int kRank = 128;
constexpr int kChunk = 16;
constexpr int kComponents = 4;
constexpr int kThreads = 512;
constexpr int kGroup = 32;
constexpr int kMatrixTile = 512;
constexpr double kRadius = 1.0 / 8.0;

struct Accumulator {
    float hi = 0.0f;
    float lo = 0.0f;

    __device__ __forceinline__ void add(float value) {
        const float next = __fadd_rn(hi, value);
        const float bridge = __fsub_rn(next, hi);
        const float error = __fadd_rn(
            __fsub_rn(hi, __fsub_rn(next, bridge)),
            __fsub_rn(value, bridge));
        hi = next;
        lo = __fadd_rn(lo, error);
    }

    __device__ __forceinline__ void add_product(float left, float right) {
        const float product = __fmul_rn(left, right);
        const float product_error = __fmaf_rn(left, right, -product);
        const float next = __fadd_rn(hi, product);
        const float bridge = __fsub_rn(next, hi);
        const float sum_error = __fadd_rn(
            __fsub_rn(hi, __fsub_rn(next, bridge)),
            __fsub_rn(product, bridge));
        hi = next;
        lo = __fadd_rn(lo, __fadd_rn(product_error, sum_error));
    }

    __device__ __forceinline__ void add_triple(
        float first, float second, float third) {
        const float product = __fmul_rn(first, second);
        const float product_error = __fmaf_rn(first, second, -product);
        add_product(product, third);
        lo = __fmaf_rn(product_error, third, lo);
    }

    __device__ __forceinline__ void add_accumulator(
        const Accumulator& other) {
        add(other.hi);
        lo = __fadd_rn(lo, other.lo);
    }

    __device__ __forceinline__ void add_accumulator_product(
        const Accumulator& left, const Accumulator& right) {
        add_product(left.hi, right.hi);
        lo = __fmaf_rn(left.hi, right.lo, lo);
        lo = __fmaf_rn(left.lo, right.hi, lo);
        lo = __fmaf_rn(left.lo, right.lo, lo);
    }

    __device__ __forceinline__ double value() const {
        return static_cast<double>(hi) + static_cast<double>(lo);
    }
};

__device__ __forceinline__ void reduce_group(Accumulator& value) {
#pragma unroll
    for (int offset = kGroup / 2; offset > 0; offset >>= 1) {
        Accumulator other;
        other.hi = __shfl_down_sync(0xffffffffu, value.hi, offset, kGroup);
        other.lo = __shfl_down_sync(0xffffffffu, value.lo, offset, kGroup);
        value.add_accumulator(other);
    }
}

__launch_bounds__(kThreads, 1)
__global__ void boundary_statistics_kernel(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    double* __restrict__ statistics,
    const int panels) {
    __shared__ float vectors[2][kChunk * kRank];
    __shared__ float matrix_j[kMatrixTile];
    __shared__ float matrix_d[kMatrixTile];

    const int panel = blockIdx.x;
    if (panel >= panels) return;
    const int source = threadIdx.x / kGroup;
    const int lane = threadIdx.x % kGroup;
    const int vector_base = panel * kChunk * kRank;
    const int matrix_base = panel * kRank * kRank;

    for (int index = threadIdx.x; index < kChunk * kRank; index += kThreads) {
        vectors[0][index] = u[vector_base + index];
        vectors[1][index] = h[vector_base + index];
    }
    __syncthreads();

    Accumulator cross_j_lower;
    Accumulator cross_d_lower;
    Accumulator cross_j_upper;
    Accumulator cross_d_upper;
    Accumulator norm_j_lower;
    Accumulator norm_d_lower;
    Accumulator norm_j_upper;
    Accumulator norm_d_upper;

    for (int base = 0; base < kRank * kRank; base += kMatrixTile) {
        for (int offset = threadIdx.x; offset < kMatrixTile; offset += kThreads) {
            const int entry = base + offset;
            matrix_j[offset] = boundary_j[matrix_base + entry];
            matrix_d[offset] = boundary_d[matrix_base + entry];
        }
        __syncthreads();

#pragma unroll
        for (int offset = lane; offset < kMatrixTile; offset += kGroup) {
            const int entry = base + offset;
            const int row = entry / kRank;
            const int col = entry % kRank;
            const float j = matrix_j[offset];
            const float d = matrix_d[offset];
            const float u_row = vectors[0][source * kRank + row];
            const float u_col = vectors[0][source * kRank + col];
            const float h_col = vectors[1][source * kRank + col];
            if (row > col) {
                cross_j_lower.add_triple(j, u_row, u_col);
                cross_d_lower.add_triple(d, u_row, h_col);
                if (source == 0) {
                    norm_j_lower.add_product(j, j);
                    norm_d_lower.add_product(d, d);
                }
            } else if (row < col) {
                cross_j_upper.add_triple(j, u_row, u_col);
                cross_d_upper.add_triple(d, u_row, h_col);
                if (source == 0) {
                    norm_j_upper.add_product(j, j);
                    norm_d_upper.add_product(d, d);
                }
            }
        }
        __syncthreads();
    }

    reduce_group(cross_j_lower);
    reduce_group(cross_d_lower);
    reduce_group(cross_j_upper);
    reduce_group(cross_d_upper);
    if (source == 0) {
        reduce_group(norm_j_lower);
        reduce_group(norm_d_lower);
        reduce_group(norm_j_upper);
        reduce_group(norm_d_upper);
    }
    if (lane == 0) {
        const int base = panel * (kComponents + kChunk * kComponents);
        statistics[base + kComponents + source * kComponents + 0] =
            cross_j_lower.value();
        statistics[base + kComponents + source * kComponents + 1] =
            cross_d_lower.value();
        statistics[base + kComponents + source * kComponents + 2] =
            cross_j_upper.value();
        statistics[base + kComponents + source * kComponents + 3] =
            cross_d_upper.value();
        if (source == 0) {
            statistics[base + 0] = norm_j_lower.value();
            statistics[base + 1] = norm_d_lower.value();
            statistics[base + 2] = norm_j_upper.value();
            statistics[base + 3] = norm_d_upper.value();
        }
    }
}

__launch_bounds__(256, 2)
__global__ void local_statistics_kernel(
    const float* __restrict__ u,
    const float* __restrict__ h,
    double* __restrict__ statistics,
    const int panels) {
    __shared__ float vectors[2][kChunk * kRank];
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    const int pair = threadIdx.x;
    const int source = pair / kChunk;
    const int other = pair % kChunk;
    const int vector_base = panel * kChunk * kRank;
    for (int index = threadIdx.x; index < 2 * kChunk * kRank; index += 256) {
        const int which = index / (kChunk * kRank);
        const int remainder = index % (kChunk * kRank);
        vectors[which][remainder] = which == 0
            ? u[vector_base + remainder]
            : h[vector_base + remainder];
    }
    __syncthreads();

    Accumulator sum_u;
    Accumulator sum_h;
    Accumulator diagonal_j;
    Accumulator diagonal_d;
    Accumulator gram_j_lower;
    Accumulator gram_d_lower;

#pragma unroll 1
    for (int coordinate = 0; coordinate < kRank; ++coordinate) {
        Accumulator product_u;
        product_u.add_product(
            vectors[0][source * kRank + coordinate],
            vectors[0][other * kRank + coordinate]);
        Accumulator product_h;
        product_h.add_product(
            vectors[1][source * kRank + coordinate],
            vectors[1][other * kRank + coordinate]);
        gram_j_lower.add_accumulator_product(product_u, sum_u);
        gram_d_lower.add_accumulator_product(product_u, sum_h);
        diagonal_j.add_accumulator_product(product_u, product_u);
        diagonal_d.add_accumulator_product(product_u, product_h);
        sum_u.add_accumulator(product_u);
        sum_h.add_accumulator(product_h);
    }

    const int output = (panel * kChunk * kChunk + pair) * kComponents;
    const double su = sum_u.value();
    const double sh = sum_h.value();
    const double j_lower = gram_j_lower.value();
    const double d_lower = gram_d_lower.value();
    statistics[output + 0] = j_lower;
    statistics[output + 1] = d_lower;
    statistics[output + 2] = su * su - diagonal_j.value() - j_lower;
    statistics[output + 3] =
        su * sh - diagonal_d.value() - d_lower;
}

__launch_bounds__(32, 4)
__global__ void norm_map_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ weights,
    const float* __restrict__ strength,
    const double* __restrict__ boundary_statistics,
    const double* __restrict__ local_statistics,
    float* __restrict__ coefficient,
    float* __restrict__ norm_sq,
    const int heads,
    const int chunks,
    const int length,
    const int panels) {
    const int panel = blockIdx.x;
    const int target = threadIdx.x;
    if (panel >= panels || target >= kChunk) return;
    const int chunk = panel % chunks;
    const int head = (panel / chunks) % heads;
    const bool valid = chunk * kChunk + target < length;
    const int boundary_base =
        panel * (kComponents + kChunk * kComponents);
    const int local_base = panel * kChunk * kChunk * kComponents;
    const double a = static_cast<double>(alpha[panel * kChunk + target]);
    double result[kComponents];
#pragma unroll
    for (int component = 0; component < kComponents; ++component) {
        result[component] =
            a * a * boundary_statistics[boundary_base + component];
    }
#pragma unroll
    for (int source = 0; source < kChunk; ++source) {
        const double w = static_cast<double>(
            weights[(panel * kChunk + source) * kChunk + target]);
#pragma unroll
        for (int component = 0; component < kComponents; ++component) {
            result[component] += 2.0 * a * w
                * boundary_statistics[
                    boundary_base + kComponents
                    + source * kComponents + component];
        }
#pragma unroll
        for (int other = 0; other < kChunk; ++other) {
            const double wo = static_cast<double>(
                weights[(panel * kChunk + other) * kChunk + target]);
#pragma unroll
            for (int component = 0; component < kComponents; ++component) {
                result[component] += w * wo
                    * local_statistics[
                        local_base
                        + (source * kChunk + other) * kComponents
                        + component];
            }
        }
    }
    const double g = static_cast<double>(strength[head]);
    const double g2 = g * g;
    const int output = (panel * kChunk + target) * kComponents;
#pragma unroll
    for (int component = 0; component < kComponents; ++component) {
        const double scaled = g2 * result[component];
        norm_sq[output + component] =
            valid ? static_cast<float>(scaled) : 0.0f;
        coefficient[output + component] = valid
            ? static_cast<float>(
                g * kRadius / sqrt(kRadius * kRadius + scaled))
            : 0.0f;
    }
}

__device__ __forceinline__ float diagonal_value(
    const float boundary,
    const float* __restrict__ source_a,
    const float* __restrict__ source_b,
    const float* __restrict__ coefficients,
    const int target,
    const int coordinate) {
    Accumulator value;
    value.add_product(coefficients[kChunk * kChunk + target], boundary);
#pragma unroll
    for (int source = 0; source < kChunk; ++source) {
        value.add_triple(
            coefficients[source * kChunk + target],
            source_a[source * kRank + coordinate],
            source_b[source * kRank + coordinate]);
    }
    return static_cast<float>(value.value());
}

__launch_bounds__(256, 2)
__global__ void diagonal_map_kernel(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ alpha,
    const float* __restrict__ weights,
    const float* __restrict__ strength,
    float* __restrict__ diagonal,
    float* __restrict__ base_diagonal_h,
    float* __restrict__ base_diagonal_r,
    const int heads,
    const int chunks,
    const int length,
    const int panels) {
    __shared__ float vectors[2][kChunk * kRank];
    __shared__ float coefficients[kChunk * kChunk + kChunk];
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    const int chunk = panel % chunks;
    const int head = (panel / chunks) % heads;
    const int vector_base = panel * kChunk * kRank;
    const int matrix_base = panel * kRank * kRank;
    for (int index = threadIdx.x; index < 2 * kChunk * kRank; index += 256) {
        const int which = index / (kChunk * kRank);
        const int remainder = index % (kChunk * kRank);
        vectors[which][remainder] = which == 0
            ? u[vector_base + remainder]
            : h[vector_base + remainder];
    }
    for (int index = threadIdx.x; index < kChunk * kChunk; index += 256) {
        coefficients[index] = weights[panel * kChunk * kChunk + index];
    }
    if (threadIdx.x < kChunk) {
        coefficients[kChunk * kChunk + threadIdx.x] =
            alpha[panel * kChunk + threadIdx.x];
    }
    __syncthreads();

    const float g = strength[head];
    constexpr float radius = static_cast<float>(kRadius);
    for (int item = threadIdx.x; item < kChunk * kRank; item += 256) {
        const int target = item / kRank;
        const int coordinate = item % kRank;
        const bool valid = chunk * kChunk + target < length;
        const float j = diagonal_value(
            boundary_j[matrix_base + coordinate * (kRank + 1)],
            vectors[0], vectors[0], coefficients, target, coordinate);
        const float d = diagonal_value(
            boundary_d[matrix_base + coordinate * (kRank + 1)],
            vectors[0], vectors[1], coefficients, target, coordinate);
        const float base_h = j - 1.0f / static_cast<float>(kRank);
        const float base_r = d;
        base_diagonal_h[vector_base + item] = valid ? base_h : 0.0f;
        base_diagonal_r[vector_base + item] = valid ? base_r : 0.0f;
        diagonal[vector_base + item] = valid
            ? expf(
                radius * tanhf(g * base_h / radius)
                + radius * tanhf(g * base_r / radius))
            : 1.0f;
    }
}

void check_tensor(
    const at::Tensor& value,
    const at::Tensor& reference,
    const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(value.scalar_type() == at::kFloat, name, " must be FP32");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.device() == reference.device(), name, " must share a device");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
radial_forward_cuda(
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& alpha,
    const at::Tensor& weights,
    const at::Tensor& strength,
    const int64_t heads_value,
    const int64_t chunks_value,
    const int64_t length_value) {
    TORCH_CHECK(boundary_j.is_cuda(), "boundary_j must be a CUDA tensor");
    TORCH_CHECK(boundary_j.scalar_type() == at::kFloat, "boundary_j must be FP32");
    TORCH_CHECK(boundary_j.is_contiguous(), "boundary_j must be contiguous");
    check_tensor(boundary_d, boundary_j, "boundary_d");
    check_tensor(u, boundary_j, "u");
    check_tensor(h, boundary_j, "h");
    check_tensor(alpha, boundary_j, "alpha");
    check_tensor(weights, boundary_j, "weights");
    check_tensor(strength, boundary_j, "strength");

    const int64_t panels_value = boundary_j.size(0);
    TORCH_CHECK(
        boundary_j.dim() == 3 && boundary_j.size(1) == kRank
        && boundary_j.size(2) == kRank,
        "boundary_j must have shape [P,128,128]");
    TORCH_CHECK(boundary_d.sizes() == boundary_j.sizes(), "boundary_d shape mismatch");
    TORCH_CHECK(
        u.dim() == 3 && u.size(0) == panels_value && u.size(1) == kChunk
        && u.size(2) == kRank,
        "u must have shape [P,16,128]");
    TORCH_CHECK(h.sizes() == u.sizes(), "h shape mismatch");
    TORCH_CHECK(
        alpha.dim() == 2 && alpha.size(0) == panels_value
        && alpha.size(1) == kChunk,
        "alpha must have shape [P,16]");
    TORCH_CHECK(
        weights.dim() == 3 && weights.size(0) == panels_value
        && weights.size(1) == kChunk && weights.size(2) == kChunk,
        "weights must have shape [P,16,16]");
    TORCH_CHECK(heads_value > 0 && chunks_value > 0, "heads and chunks must be positive");
    TORCH_CHECK(length_value > 0, "length must be positive");
    TORCH_CHECK(
        chunks_value == (length_value + kChunk - 1) / kChunk,
        "chunks must equal ceil(length/16)");
    TORCH_CHECK(
        panels_value > 0 && panels_value % (heads_value * chunks_value) == 0,
        "panel count must be divisible by heads * chunks");
    TORCH_CHECK(strength.numel() == heads_value, "strength must have one value per head");
    TORCH_CHECK(
        panels_value <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "too many panels");

    const int panels = static_cast<int>(panels_value);
    const int heads = static_cast<int>(heads_value);
    const int chunks = static_cast<int>(chunks_value);
    const int length = static_cast<int>(length_value);
    auto stats_options = boundary_j.options().dtype(at::kDouble);
    auto boundary_statistics = at::empty(
        {panels, kComponents + kChunk * kComponents}, stats_options);
    auto local_statistics = at::empty(
        {panels, kChunk, kChunk, kComponents}, stats_options);
    auto coefficient = at::empty(
        {panels, kChunk, kComponents}, boundary_j.options());
    auto norm_sq = at::empty_like(coefficient);
    auto diagonal = at::empty({panels, kChunk, kRank}, boundary_j.options());
    auto base_diagonal_h = at::empty_like(diagonal);
    auto base_diagonal_r = at::empty_like(diagonal);

    c10::cuda::CUDAGuard guard(boundary_j.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    boundary_statistics_kernel<<<panels, kThreads, 0, stream>>>(
        boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(),
        boundary_statistics.data_ptr<double>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    local_statistics_kernel<<<panels, 256, 0, stream>>>(
        u.data_ptr<float>(), h.data_ptr<float>(),
        local_statistics.data_ptr<double>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    norm_map_kernel<<<panels, 32, 0, stream>>>(
        alpha.data_ptr<float>(), weights.data_ptr<float>(),
        strength.data_ptr<float>(), boundary_statistics.data_ptr<double>(),
        local_statistics.data_ptr<double>(), coefficient.data_ptr<float>(),
        norm_sq.data_ptr<float>(), heads, chunks, length, panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    diagonal_map_kernel<<<panels, 256, 0, stream>>>(
        boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), alpha.data_ptr<float>(),
        weights.data_ptr<float>(), strength.data_ptr<float>(),
        diagonal.data_ptr<float>(), base_diagonal_h.data_ptr<float>(),
        base_diagonal_r.data_ptr<float>(), heads, chunks, length, panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {coefficient, diagonal, norm_sq, base_diagonal_h, base_diagonal_r};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "packet_frame_radial_forward128(Tensor boundary_j, Tensor boundary_d, "
        "Tensor u, Tensor h, Tensor alpha, Tensor weights, Tensor strength, "
        "int heads, int chunks, int length) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("packet_frame_radial_forward128", &radial_forward_cuda);
}
