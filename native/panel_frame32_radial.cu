#include "panel_frame32_radial.cuh"

#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

namespace {

constexpr int kRank = 128;
constexpr int kChunk = 32;
constexpr int kComponents = 4;
constexpr int kThreads = 512;
constexpr int kEntriesPerThread = 16;
constexpr int kStrictEntries = kRank * (kRank - 1) / 2;
constexpr double kRadius = 1.0 / 8.0;

struct Accumulator {
    float hi = 0.0f;
    float lo = 0.0f;

    __device__ __forceinline__ void add_product(float left, float right) {
        hi = __fmaf_rn(left, right, hi);
    }

    __device__ __forceinline__ void add_accumulator(
        const Accumulator& other) {
        hi = __fadd_rn(hi, other.hi);
    }

    __device__ __forceinline__ void add_accumulator_product(
        const Accumulator& left, const Accumulator& right) {
        hi = __fmaf_rn(left.hi, right.hi, hi);
    }
};

__device__ __forceinline__ Accumulator scaled(
    const Accumulator& value, float factor) {
    Accumulator result;
    result.hi = __fmul_rn(value.hi, factor);
    return result;
}

__device__ __forceinline__ void decode_lower_entry(
    int entry, int& row, int& column) {
    row = static_cast<int>(
        (1.0f + sqrtf(1.0f + 8.0f * static_cast<float>(entry))) * 0.5f);
    while (row * (row - 1) / 2 > entry) --row;
    while ((row + 1) * row / 2 <= entry) ++row;
    column = entry - row * (row - 1) / 2;
}

__device__ __forceinline__ void reduce_warp(Accumulator& value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value.hi = __fadd_rn(
            value.hi,
            __shfl_down_sync(0xffffffffu, value.hi, offset));
    }
}

__launch_bounds__(kThreads, 1)
__global__ void recurrent_parameters_kernel(
    const float* __restrict__ boundary_m,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ log_decay,
    const float* __restrict__ strength,
    float* __restrict__ alpha0,
    float* __restrict__ inverse_mass_out,
    float* __restrict__ coefficient,
    float* __restrict__ norm_sq,
    float* __restrict__ diagonal,
    int heads,
    int chunks,
    int length,
    int panels) {
    __shared__ float vectors[2][kChunk * kRank];
    __shared__ float decay[kChunk];
    __shared__ float inverse_mass[kChunk];
    __shared__ float reduction[2][16];
    const int panel = blockIdx.x;
    const int side = blockIdx.y;
    if (panel >= panels) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int chunk = panel % chunks;
    const int head = (panel / chunks) % heads;
    const int vector_base = panel * kChunk * kRank;
    const int matrix_base = panel * kRank * kRank;

    for (int index = threadIdx.x; index < 2 * kChunk * kRank;
         index += kThreads) {
        const int which = index / (kChunk * kRank);
        const int remainder = index % (kChunk * kRank);
        vectors[which][remainder] = which == 0
            ? u[vector_base + remainder]
            : h[vector_base + remainder];
    }
    if (threadIdx.x == 0) {
        float mass = boundary_m[panel];
        float first_alpha = 0.0f;
#pragma unroll
        for (int target = 0; target < kChunk; ++target) {
            const bool valid = chunk * kChunk + target < length;
            const float factor = valid
                ? expf(log_decay[panel * kChunk + target]) : 0.0f;
            mass = valid ? fmaf(factor, mass, 1.0f) : mass;
            const float weight = valid ? 1.0f / mass : 0.0f;
            decay[target] = factor;
            inverse_mass[target] = weight;
            if (target == 0) first_alpha = valid ? factor * weight : 0.0f;
            if (side == 0) {
                inverse_mass_out[panel * kChunk + target] = weight;
            }
        }
        if (side == 0) alpha0[panel] = first_alpha;
    }
    __syncthreads();

    int rows[kEntriesPerThread];
    int columns[kEntriesPerThread];
    bool active[kEntriesPerThread];
    Accumulator moment_j[kEntriesPerThread];
    Accumulator moment_d[kEntriesPerThread];
#pragma unroll
    for (int slot = 0; slot < kEntriesPerThread; ++slot) {
        const int entry = threadIdx.x + slot * kThreads;
        active[slot] = entry < kStrictEntries;
        int lower_row = 1;
        int lower_column = 0;
        if (active[slot]) decode_lower_entry(entry, lower_row, lower_column);
        rows[slot] = side == 0 ? lower_row : lower_column;
        columns[slot] = side == 0 ? lower_column : lower_row;
        if (active[slot]) {
            moment_j[slot].hi = boundary_j[
                matrix_base + rows[slot] * kRank + columns[slot]];
            moment_d[slot].hi = boundary_d[
                matrix_base + rows[slot] * kRank + columns[slot]];
        }
    }

    Accumulator diagonal_j;
    Accumulator diagonal_d;
    const bool owns_diagonal = side == 0 && threadIdx.x < kRank;
    if (owns_diagonal) {
        const int coordinate = threadIdx.x;
        diagonal_j.hi = boundary_j[
            matrix_base + coordinate * (kRank + 1)];
        diagonal_d.hi = boundary_d[
            matrix_base + coordinate * (kRank + 1)];
    }

    const float g = strength[head];
    constexpr float radius = static_cast<float>(kRadius);
#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        const bool valid = chunk * kChunk + target < length;
        Accumulator sum_j;
        Accumulator sum_d;
        if (valid) {
            const float factor = decay[target];
            const float weight = inverse_mass[target];
#pragma unroll
            for (int slot = 0; slot < kEntriesPerThread; ++slot) {
                if (active[slot]) {
                    Accumulator next_j = scaled(moment_j[slot], factor);
                    Accumulator next_d = scaled(moment_d[slot], factor);
                    const float ur =
                        vectors[0][target * kRank + rows[slot]];
                    next_j.add_product(
                        ur, vectors[0][target * kRank + columns[slot]]);
                    next_d.add_product(
                        ur, vectors[1][target * kRank + columns[slot]]);
                    moment_j[slot] = next_j;
                    moment_d[slot] = next_d;
                    const Accumulator normalized_j = scaled(next_j, weight);
                    const Accumulator normalized_d = scaled(next_d, weight);
                    sum_j.add_accumulator_product(normalized_j, normalized_j);
                    sum_d.add_accumulator_product(normalized_d, normalized_d);
                }
            }
            if (owns_diagonal) {
                const int coordinate = threadIdx.x;
                Accumulator next_j = scaled(diagonal_j, factor);
                Accumulator next_d = scaled(diagonal_d, factor);
                const float uv =
                    vectors[0][target * kRank + coordinate];
                next_j.add_product(uv, uv);
                next_d.add_product(
                    uv, vectors[1][target * kRank + coordinate]);
                diagonal_j = next_j;
                diagonal_d = next_d;
                const float base_h = next_j.hi * weight
                    - 1.0f / static_cast<float>(kRank);
                const float base_r = next_d.hi * weight;
                diagonal[vector_base + target * kRank + coordinate] = expf(
                    radius * tanhf(g * base_h / radius)
                    + radius * tanhf(g * base_r / radius));
            }
        } else if (owns_diagonal) {
            diagonal[vector_base + target * kRank + threadIdx.x] = 1.0f;
        }

        reduce_warp(sum_j);
        reduce_warp(sum_d);
        if (lane == 0) {
            reduction[0][warp] = sum_j.hi;
            reduction[1][warp] = sum_d.hi;
        }
        __syncthreads();
        if (warp == 0) {
            Accumulator block_j;
            Accumulator block_d;
            if (lane < 16) {
                block_j.hi = reduction[0][lane];
                block_d.hi = reduction[1][lane];
            }
            reduce_warp(block_j);
            reduce_warp(block_d);
            if (lane == 0) {
                const int component = side * 2;
                const int output =
                    (panel * kChunk + target) * kComponents + component;
                const double g2 =
                    static_cast<double>(g) * static_cast<double>(g);
                const double j_value = valid
                    ? g2 * static_cast<double>(block_j.hi) : 0.0;
                const double d_value = valid
                    ? g2 * static_cast<double>(block_d.hi) : 0.0;
                norm_sq[output] = static_cast<float>(j_value);
                norm_sq[output + 1] = static_cast<float>(d_value);
                coefficient[output] = valid
                    ? static_cast<float>(
                        static_cast<double>(g) * kRadius
                        / sqrt(kRadius * kRadius + j_value))
                    : 0.0f;
                coefficient[output + 1] = valid
                    ? static_cast<float>(
                        static_cast<double>(g) * kRadius
                        / sqrt(kRadius * kRadius + d_value))
                    : 0.0f;
            }
        }
        __syncthreads();
    }
}

void check_tensor(
    const at::Tensor& value,
    const at::Tensor& reference,
    const char* name) {
    TORCH_CHECK(
        value.is_cuda() && value.scalar_type() == at::kFloat
            && value.is_contiguous() && value.device() == reference.device(),
        name, " must be contiguous CUDA FP32 on the same device");
}

}  // namespace

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
panel_frame32_parameters_cuda(
    const at::Tensor& boundary_m,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& log_decay,
    const at::Tensor& strength,
    int64_t heads_value,
    int64_t chunks_value,
    int64_t length_value) {
    TORCH_CHECK(
        boundary_j.is_cuda() && boundary_j.scalar_type() == at::kFloat
            && boundary_j.is_contiguous() && boundary_j.dim() == 3
            && boundary_j.size(1) == kRank && boundary_j.size(2) == kRank,
        "boundary_j must be contiguous CUDA FP32 [P,128,128]");
    for (const auto& item : {
             std::pair<const at::Tensor*, const char*>{&boundary_m, "boundary_m"},
             {&boundary_d, "boundary_d"}, {&u, "u"}, {&h, "h"},
             {&log_decay, "log_decay"}, {&strength, "strength"}}) {
        check_tensor(*item.first, boundary_j, item.second);
    }
    const int64_t panels_value = boundary_j.size(0);
    TORCH_CHECK(
        boundary_j.sizes()
            == at::IntArrayRef({panels_value, kRank, kRank}),
        "boundary_j must be [P,128,128]");
    TORCH_CHECK(
        boundary_d.sizes() == boundary_j.sizes(),
        "boundary_d shape mismatch");
    TORCH_CHECK(
        boundary_m.sizes() == at::IntArrayRef({panels_value}),
        "boundary_m must be [P]");
    TORCH_CHECK(
        u.sizes() == at::IntArrayRef({panels_value, kChunk, kRank}),
        "u must be [P,32,128]");
    TORCH_CHECK(h.sizes() == u.sizes(), "h must match u");
    TORCH_CHECK(
        log_decay.sizes() == at::IntArrayRef({panels_value, kChunk}),
        "log_decay must be [P,32]");
    TORCH_CHECK(
        heads_value > 0 && chunks_value > 0 && length_value > 0,
        "heads, chunks, and length must be positive");
    TORCH_CHECK(
        chunks_value == (length_value + kChunk - 1) / kChunk,
        "chunks must equal ceil(length/32)");
    TORCH_CHECK(
        panels_value % (heads_value * chunks_value) == 0,
        "P must be divisible by heads*chunks");
    TORCH_CHECK(
        strength.numel() == heads_value,
        "strength must have one value per head");
    TORCH_CHECK(
        panels_value <= std::numeric_limits<int>::max(),
        "too many panels");

    const int panels = static_cast<int>(panels_value);
    const int heads = static_cast<int>(heads_value);
    const int chunks = static_cast<int>(chunks_value);
    const int length = static_cast<int>(length_value);
    auto alpha0 = at::empty({panels}, boundary_j.options());
    auto inverse_mass = at::empty({panels, kChunk}, boundary_j.options());
    auto coefficient = at::empty(
        {panels, kChunk, kComponents}, boundary_j.options());
    auto diagonal = at::empty(
        {panels, kChunk, kRank}, boundary_j.options());
    auto norm_sq = at::empty_like(coefficient);

    c10::cuda::CUDAGuard guard(boundary_j.device());
    recurrent_parameters_kernel<<<
        dim3(panels, 2), kThreads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        boundary_m.data_ptr<float>(), boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(), u.data_ptr<float>(), h.data_ptr<float>(),
        log_decay.data_ptr<float>(), strength.data_ptr<float>(),
        alpha0.data_ptr<float>(), inverse_mass.data_ptr<float>(),
        coefficient.data_ptr<float>(), norm_sq.data_ptr<float>(),
        diagonal.data_ptr<float>(), heads, chunks, length, panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {alpha0, inverse_mass, coefficient, diagonal, norm_sq};
}

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "panel_frame32_parameters128(Tensor boundary_m, Tensor boundary_j, "
        "Tensor boundary_d, Tensor u, Tensor h, Tensor log_decay, "
        "Tensor strength, int heads, int chunks, int length) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("panel_frame32_parameters128", &panel_frame32_parameters_cuda);
}
