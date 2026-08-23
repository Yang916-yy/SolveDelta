#include "solvedelta_c32.h"

#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <tuple>


namespace {

constexpr int kRank = 128;
constexpr int kChunk = 32;
constexpr int kTile = 16;
constexpr int kTiles = kRank / kTile;
constexpr int kComponents = 4;
constexpr int kDualRhs = 2;
constexpr int kThreads = kChunk * kTile;
constexpr int kEntriesPerThread = 16;
constexpr int kStrictEntries = kRank * (kRank - 1) / 2;
constexpr double kRadius = 1.0 / 8.0;

struct FactorShared {
    float primal[kChunk * kRank];
    float u_column[kChunk * kTile];
    float h_column[kChunk * kTile];
    float u_row[kChunk * kTile];
    float boundary_j[kTile * kTile];
    float boundary_d[kTile * kTile];
    float inverse_mass[kChunk];
    float coefficient[kChunk * kComponents];
    float factor[kChunk * kTile * kTile];
};

struct FrameShared {
    FactorShared factor;
    float dual[kDualRhs * kChunk * kRank];
    float dual_accumulator[kDualRhs * kChunk * kTile];
};

static_assert(sizeof(FrameShared) == 94848);

__device__ __forceinline__ int vector_index(
    int batch_index,
    int token,
    int head,
    int coordinate,
    int length,
    int heads) {
    return ((batch_index * length + token) * heads + head) * kRank
        + coordinate;
}

__device__ __forceinline__ void decode_panel(
    int panel,
    int heads,
    int chunks,
    int& batch_index,
    int& head,
    int& chunk) {
    chunk = panel % chunks;
    const int head_batch = panel / chunks;
    head = head_batch % heads;
    batch_index = head_batch / heads;
}

__device__ __forceinline__ void decode_lower_entry(
    int entry,
    int& row,
    int& column) {
    row = static_cast<int>(
        (1.0f + sqrtf(1.0f + 8.0f * static_cast<float>(entry))) * 0.5f);
    while (row * (row - 1) / 2 > entry) --row;
    while ((row + 1) * row / 2 <= entry) ++row;
    column = entry - row * (row - 1) / 2;
}

__device__ __forceinline__ void reduce_warp(float& value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
}

__global__ __launch_bounds__(kThreads, 1) void radial_kernel(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ geometry_log_decay,
    const float* __restrict__ geometry_strength,
    const float* __restrict__ boundary_m,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    float* __restrict__ alpha0,
    float* __restrict__ inverse_mass_output,
    float* __restrict__ coefficient,
    float* __restrict__ diagonal,
    int length,
    int heads,
    int chunks,
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
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    const int matrix_base = panel * kRank * kRank;

    for (int index = threadIdx.x;
         index < 2 * kChunk * kRank;
         index += blockDim.x) {
        const int which = index / (kChunk * kRank);
        const int local = index % (kChunk * kRank);
        const int target = local / kRank;
        const int coordinate = local % kRank;
        float value = 0.0f;
        if (target < valid_count) {
            const int source_index = vector_index(
                batch_index,
                token_start + target,
                head,
                coordinate,
                length,
                heads);
            value = which == 0 ? u[source_index] : h[source_index];
        }
        vectors[which][local] = value;
    }

    if (threadIdx.x == 0) {
        float mass = boundary_m[panel];
        float first_alpha = 0.0f;
#pragma unroll
        for (int target = 0; target < kChunk; ++target) {
            const bool valid = target < valid_count;
            float factor = 1.0f;
            if (valid) {
                const int scalar_index =
                    (batch_index * length + token_start + target) * heads + head;
                factor = expf(geometry_log_decay[scalar_index]);
                mass = fmaf(factor, mass, 1.0f);
            }
            const float weight = valid ? 1.0f / mass : 0.0f;
            decay[target] = factor;
            inverse_mass[target] = weight;
            if (target == 0) first_alpha = factor * weight;
            if (side == 0) {
                inverse_mass_output[panel * kChunk + target] = weight;
            }
        }
        if (side == 0) alpha0[panel] = first_alpha;
    }
    __syncthreads();

    int rows[kEntriesPerThread];
    int columns[kEntriesPerThread];
    bool active[kEntriesPerThread];
    float moment_j[kEntriesPerThread];
    float moment_d[kEntriesPerThread];
#pragma unroll
    for (int slot = 0; slot < kEntriesPerThread; ++slot) {
        const int entry = threadIdx.x + slot * blockDim.x;
        active[slot] = entry < kStrictEntries;
        int lower_row = 1;
        int lower_column = 0;
        if (active[slot]) decode_lower_entry(entry, lower_row, lower_column);
        rows[slot] = side == 0 ? lower_row : lower_column;
        columns[slot] = side == 0 ? lower_column : lower_row;
        moment_j[slot] = active[slot]
            ? boundary_j[matrix_base + rows[slot] * kRank + columns[slot]]
            : 0.0f;
        moment_d[slot] = active[slot]
            ? boundary_d[matrix_base + rows[slot] * kRank + columns[slot]]
            : 0.0f;
    }

    float diagonal_j = 0.0f;
    float diagonal_d = 0.0f;
    const bool owns_diagonal = side == 0 && threadIdx.x < kRank;
    if (owns_diagonal) {
        diagonal_j = boundary_j[
            matrix_base + threadIdx.x * (kRank + 1)];
        diagonal_d = boundary_d[
            matrix_base + threadIdx.x * (kRank + 1)];
    }

    const float strength = geometry_strength[head];
    constexpr float radius = static_cast<float>(kRadius);
#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        const bool valid = target < valid_count;
        float norm_j = 0.0f;
        float norm_d = 0.0f;
        if (valid) {
            const float factor = decay[target];
            const float weight = inverse_mass[target];
#pragma unroll
            for (int slot = 0; slot < kEntriesPerThread; ++slot) {
                if (active[slot]) {
                    const float row_u =
                        vectors[0][target * kRank + rows[slot]];
                    moment_j[slot] = fmaf(
                        factor,
                        moment_j[slot],
                        row_u * vectors[0][target * kRank + columns[slot]]);
                    moment_d[slot] = fmaf(
                        factor,
                        moment_d[slot],
                        row_u * vectors[1][target * kRank + columns[slot]]);
                    const float normalized_j = moment_j[slot] * weight;
                    const float normalized_d = moment_d[slot] * weight;
                    norm_j = fmaf(normalized_j, normalized_j, norm_j);
                    norm_d = fmaf(normalized_d, normalized_d, norm_d);
                }
            }
            if (owns_diagonal) {
                const int coordinate = threadIdx.x;
                const float token_u =
                    vectors[0][target * kRank + coordinate];
                diagonal_j = fmaf(factor, diagonal_j, token_u * token_u);
                diagonal_d = fmaf(
                    factor,
                    diagonal_d,
                    token_u * vectors[1][target * kRank + coordinate]);
                const float x_j = strength * (
                    diagonal_j * weight
                    - 1.0f / static_cast<float>(kRank));
                const float x_d = strength * diagonal_d * weight;
                diagonal[(panel * kChunk + target) * kRank + coordinate] =
                    expf(
                        radius * tanhf(x_j / radius)
                        + radius * tanhf(x_d / radius));
            }
        } else if (owns_diagonal) {
            diagonal[(panel * kChunk + target) * kRank + threadIdx.x] = 1.0f;
        }

        reduce_warp(norm_j);
        reduce_warp(norm_d);
        if (lane == 0) {
            reduction[0][warp] = norm_j;
            reduction[1][warp] = norm_d;
        }
        __syncthreads();
        if (warp == 0) {
            float block_j = lane < 16 ? reduction[0][lane] : 0.0f;
            float block_d = lane < 16 ? reduction[1][lane] : 0.0f;
            reduce_warp(block_j);
            reduce_warp(block_d);
            if (lane == 0) {
                const int output =
                    (panel * kChunk + target) * kComponents + side * 2;
                const double strength_squared =
                    static_cast<double>(strength)
                    * static_cast<double>(strength);
                const double scaled_j = valid
                    ? strength_squared * static_cast<double>(block_j)
                    : 0.0;
                const double scaled_d = valid
                    ? strength_squared * static_cast<double>(block_d)
                    : 0.0;
                coefficient[output] = valid
                    ? static_cast<float>(
                        static_cast<double>(strength) * kRadius
                        / sqrt(kRadius * kRadius + scaled_j))
                    : 0.0f;
                coefficient[output + 1] = valid
                    ? static_cast<float>(
                        static_cast<double>(strength) * kRadius
                        / sqrt(kRadius * kRadius + scaled_d))
                    : 0.0f;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float load_chunk_vector(
    const float* source,
    int batch_index,
    int token_start,
    int target,
    int head,
    int coordinate,
    int valid_count,
    int length,
    int heads) {
    if (target >= valid_count) return 0.0f;
    return source[vector_index(
        batch_index,
        token_start + target,
        head,
        coordinate,
        length,
        heads)];
}

template <bool Upper>
__device__ __forceinline__ void reconstruct_factor(
    FactorShared& shared,
    const float* __restrict__ alpha0,
    int panel) {
    constexpr int component = Upper ? 2 : 0;
    const int tid = threadIdx.x;
    if (tid < kTile * kTile) {
        const int row = tid / kTile;
        const int column = tid % kTile;
        const float weight0 = shared.inverse_mass[0];
        float moment_j = fmaf(
            alpha0[panel],
            shared.boundary_j[tid],
            weight0 * shared.u_row[row] * shared.u_column[column]);
        float moment_d = fmaf(
            alpha0[panel],
            shared.boundary_d[tid],
            weight0 * shared.u_row[row] * shared.h_column[column]);
        shared.factor[tid] = fmaf(
            shared.coefficient[component],
            moment_j,
            shared.coefficient[component + 1] * moment_d);
#pragma unroll 1
        for (int target = 1; target < kChunk; ++target) {
            const float weight = shared.inverse_mass[target];
            const float retain = 1.0f - weight;
            moment_j = fmaf(
                retain,
                moment_j,
                weight
                    * shared.u_row[target * kTile + row]
                    * shared.u_column[target * kTile + column]);
            moment_d = fmaf(
                retain,
                moment_d,
                weight
                    * shared.u_row[target * kTile + row]
                    * shared.h_column[target * kTile + column]);
            shared.factor[target * kTile * kTile + tid] = fmaf(
                shared.coefficient[target * kComponents + component],
                moment_j,
                shared.coefficient[target * kComponents + component + 1]
                    * moment_d);
        }
    }
    __syncthreads();
}

template <bool Upper, bool DiagonalBlock>
__device__ __forceinline__ void accumulate_dual(
    FrameShared& shared,
    int row_start) {
    const int target = threadIdx.x / kTile;
    const int column = threadIdx.x % kTile;
#pragma unroll
    for (int rhs = 0; rhs < kDualRhs; ++rhs) {
        float action = 0.0f;
#pragma unroll
        for (int row = 0; row < kTile; ++row) {
            bool active = true;
            if constexpr (DiagonalBlock) {
                active = Upper ? row < column : row > column;
            }
            if (active) {
                action = fmaf(
                    shared.factor.factor[
                        target * kTile * kTile + row * kTile + column],
                    shared.dual[
                        (rhs * kChunk + target) * kRank + row_start + row],
                    action);
            }
        }
        shared.dual_accumulator[
            (rhs * kChunk + target) * kTile + column] += action;
    }
}

template <bool Upper>
__device__ __forceinline__ void frame_step(
    FrameShared& shared,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ alpha0,
    int panel,
    int tile,
    int batch_index,
    int head,
    int token_start,
    int valid_count,
    int length,
    int heads) {
    auto& factor = shared.factor;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int matrix_base = panel * kRank * kRank;
    const int column_start = tile * kTile;
    const int target = tid / kTile;
    const int local_coordinate = tid % kTile;
    const int column_coordinate = column_start + local_coordinate;

    factor.u_column[tid] = load_chunk_vector(
        u, batch_index, token_start, target, head, column_coordinate,
        valid_count, length, heads);
    factor.h_column[tid] = load_chunk_vector(
        h, batch_index, token_start, target, head, column_coordinate,
        valid_count, length, heads);
    factor.u_row[tid] = factor.u_column[tid];
    if (tid < kTile * kTile) {
        const int row = column_start + tid / kTile;
        const int column = column_start + tid % kTile;
        factor.boundary_j[tid] =
            boundary_j[matrix_base + row * kRank + column];
        factor.boundary_d[tid] =
            boundary_d[matrix_base + row * kRank + column];
    }
    for (int index = tid;
         index < kDualRhs * kChunk * kTile;
         index += blockDim.x) {
        const int rhs = index / (kChunk * kTile);
        const int local = index % (kChunk * kTile);
        const int local_target = local / kTile;
        const int coordinate = local % kTile;
        shared.dual_accumulator[index] = shared.dual[
            (rhs * kChunk + local_target) * kRank
            + column_start + coordinate];
    }
    __syncthreads();
    reconstruct_factor<Upper>(factor, alpha0, panel);

#pragma unroll
    for (int target_group = 0; target_group < 2; ++target_group) {
        const int local_target = target_group * 16 + warp;
        float residual = lane < kTile
            ? factor.primal[
                local_target * kRank + column_start + lane]
            : 0.0f;
#pragma unroll
        for (int step = 0; step < kTile; ++step) {
            const int pivot = Upper ? kTile - 1 - step : step;
            const float solved = __shfl_sync(0xffffffffu, residual, pivot);
            const bool active = Upper
                ? lane < pivot
                : lane > pivot && lane < kTile;
            if (active) {
                residual = fmaf(
                    -factor.factor[
                        local_target * kTile * kTile
                        + lane * kTile + pivot],
                    solved,
                    residual);
            }
        }
        if (lane < kTile) {
            factor.primal[
                local_target * kRank + column_start + lane] = residual;
        }
    }
    accumulate_dual<Upper, true>(shared, column_start);
    __syncthreads();

    const int row_begin = Upper ? 0 : tile + 1;
    const int row_end = Upper ? tile : kTiles;
    for (int row_tile = row_begin; row_tile < row_end; ++row_tile) {
        const int row_start = row_tile * kTile;
        factor.u_row[tid] = load_chunk_vector(
            u,
            batch_index,
            token_start,
            target,
            head,
            row_start + local_coordinate,
            valid_count,
            length,
            heads);
        if (tid < kTile * kTile) {
            const int row = row_start + tid / kTile;
            const int column = column_start + tid % kTile;
            factor.boundary_j[tid] =
                boundary_j[matrix_base + row * kRank + column];
            factor.boundary_d[tid] =
                boundary_d[matrix_base + row * kRank + column];
        }
        __syncthreads();
        reconstruct_factor<Upper>(factor, alpha0, panel);

        float action = 0.0f;
#pragma unroll
        for (int column = 0; column < kTile; ++column) {
            action = fmaf(
                factor.factor[
                    target * kTile * kTile
                    + local_coordinate * kTile + column],
                factor.primal[
                    target * kRank + column_start + column],
                action);
        }
        factor.primal[target * kRank + row_start + local_coordinate] -= action;
        accumulate_dual<Upper, false>(shared, row_start);
        __syncthreads();
    }

    for (int index = tid;
         index < kDualRhs * kChunk * kTile;
         index += blockDim.x) {
        const int rhs = index / (kChunk * kTile);
        const int local = index % (kChunk * kTile);
        const int local_target = local / kTile;
        const int coordinate = local % kTile;
        shared.dual[
            (rhs * kChunk + local_target) * kRank
            + column_start + coordinate] = shared.dual_accumulator[index];
    }
    __syncthreads();
}

__global__ __launch_bounds__(kThreads, 1) void frame_kernel(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ key,
    const float* __restrict__ erase,
    const float* __restrict__ query,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ alpha0,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    float* __restrict__ write_direction,
    float* __restrict__ erase_direction,
    float* __restrict__ solved_query,
    float* __restrict__ lower_primal,
    float* __restrict__ lower_dual_scaled,
    int length,
    int heads,
    int chunks,
    int panels) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<FrameShared*>(storage);
    auto& factor = shared.factor;
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);

    for (int index = threadIdx.x;
         index < kChunk * kRank;
         index += blockDim.x) {
        const int target = index / kRank;
        const int coordinate = index % kRank;
        const float key_value = load_chunk_vector(
            key,
            batch_index,
            token_start,
            target,
            head,
            coordinate,
            valid_count,
            length,
            heads);
        const float erase_value = load_chunk_vector(
            erase,
            batch_index,
            token_start,
            target,
            head,
            coordinate,
            valid_count,
            length,
            heads);
        factor.primal[index] = key_value;
        shared.dual[index] = erase_value * key_value;
        shared.dual[kChunk * kRank + index] = load_chunk_vector(
            query,
            batch_index,
            token_start,
            target,
            head,
            coordinate,
            valid_count,
            length,
            heads);
    }
    if (threadIdx.x < kChunk) {
        factor.inverse_mass[threadIdx.x] =
            inverse_mass[panel * kChunk + threadIdx.x];
    }
    if (threadIdx.x < kChunk * kComponents) {
        factor.coefficient[threadIdx.x] =
            coefficient[panel * kChunk * kComponents + threadIdx.x];
    }
    __syncthreads();

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        frame_step<false>(
            shared,
            boundary_j,
            boundary_d,
            u,
            h,
            alpha0,
            panel,
            tile,
            batch_index,
            head,
            token_start,
            valid_count,
            length,
            heads);
    }

    for (int index = threadIdx.x;
         index < kChunk * kRank;
         index += blockDim.x) {
        const int target = index / kRank;
        const int coordinate = index % kRank;
        const float scale = diagonal[(panel * kChunk + target) * kRank + coordinate];
        if (target < valid_count) {
            const int output_index = vector_index(
                batch_index,
                token_start + target,
                head,
                coordinate,
                length,
                heads);
            lower_primal[output_index] = factor.primal[index];
        }
        factor.primal[index] /= scale;
        shared.dual[index] *= scale;
        shared.dual[kChunk * kRank + index] *= scale;
        if (target < valid_count) {
            const int dual_base = (
                ((batch_index * length + token_start + target) * heads + head)
                    * kDualRhs) * kRank + coordinate;
            lower_dual_scaled[dual_base] = shared.dual[index];
            lower_dual_scaled[dual_base + kRank] =
                shared.dual[kChunk * kRank + index];
        }
    }
    __syncthreads();

#pragma unroll 1
    for (int tile = kTiles - 1; tile >= 0; --tile) {
        frame_step<true>(
            shared,
            boundary_j,
            boundary_d,
            u,
            h,
            alpha0,
            panel,
            tile,
            batch_index,
            head,
            token_start,
            valid_count,
            length,
            heads);
    }

    for (int index = threadIdx.x;
         index < kChunk * kRank;
         index += blockDim.x) {
        const int target = index / kRank;
        if (target >= valid_count) continue;
        const int coordinate = index % kRank;
        const int output_index = vector_index(
            batch_index,
            token_start + target,
            head,
            coordinate,
            length,
            heads);
        write_direction[output_index] = factor.primal[index];
        erase_direction[output_index] = shared.dual[index];
        solved_query[output_index] = shared.dual[kChunk * kRank + index];
    }
}

void check_fp32_cuda_contiguous(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(
        tensor.get_device() == reference.get_device(),
        name,
        " must share one CUDA device");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be FP32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

C32ForwardResult c32_frame_forward_cuda(
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& geometry_log_decay,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& geometry_strength,
    const at::Tensor& boundary_m,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d) {
    TORCH_CHECK(u.is_cuda(), "u must be CUDA");
    TORCH_CHECK(
        u.scalar_type() == at::kFloat && u.is_contiguous(),
        "u must be contiguous FP32");
    TORCH_CHECK(u.dim() == 4 && u.size(3) == kRank, "u must be [B,T,H,128]");
    const int64_t batch = u.size(0);
    const int64_t length = u.size(1);
    const int64_t heads = u.size(2);
    TORCH_CHECK(batch > 0 && length > 0 && heads > 0, "B, T, and H must be positive");
    const int64_t chunks = (length - 1) / kChunk + 1;
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{&h, "h"},
             {&geometry_log_decay, "geometry_log_decay"},
             {&key, "key"},
             {&erase, "erase"},
             {&query, "query"},
             {&geometry_strength, "geometry_strength"},
             {&boundary_m, "boundary_m"},
             {&boundary_j, "boundary_J"},
             {&boundary_d, "boundary_D"}}) {
        check_fp32_cuda_contiguous(*named.first, u, named.second);
    }
    TORCH_CHECK(h.sizes() == u.sizes() && query.sizes() == u.sizes(), "h/query shape mismatch");
    TORCH_CHECK(
        geometry_log_decay.sizes() == at::IntArrayRef({batch, length, heads}),
        "geometry_log_decay must be [B,T,H]");
    TORCH_CHECK(
        key.sizes() == at::IntArrayRef({batch, length, heads, 1, kRank})
            && erase.sizes() == key.sizes(),
        "key/erase must be [B,T,H,1,128]");
    TORCH_CHECK(
        geometry_strength.sizes() == at::IntArrayRef({heads}),
        "geometry_strength must be [H]");
    TORCH_CHECK(
        boundary_m.sizes() == at::IntArrayRef({batch, heads, chunks}),
        "boundary_m must be [B,H,N]");
    TORCH_CHECK(
        boundary_j.sizes() == at::IntArrayRef({batch, heads, chunks, kRank, kRank})
            && boundary_d.sizes() == boundary_j.sizes(),
        "boundary_J/D must be [B,H,N,128,128]");
    constexpr int64_t max_index = std::numeric_limits<int>::max();
    TORCH_CHECK(
        length <= max_index && heads <= max_index && chunks <= max_index,
        "length, heads, and chunks must fit the native int32 launch ABI");
    TORCH_CHECK(
        boundary_m.numel() <= max_index,
        "panel count must fit the native int32 index space");
    TORCH_CHECK(
        u.numel() <= max_index / kDualRhs,
        "vector tensors exceed the native int32 index space");
    TORCH_CHECK(
        boundary_j.numel() <= max_index,
        "boundary matrices exceed the native int32 index space");

    c10::cuda::CUDAGuard guard(u.device());
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, u.get_device()));
    TORCH_CHECK(
        properties.major == 12 && properties.minor == 0,
        "the native C32 frame contains only the SM120 specialization; got SM",
        properties.major,
        properties.minor);
    auto write_direction = at::empty_like(key);
    auto erase_direction = at::empty_like(key);
    auto solved_query = at::empty_like(query);
    auto inverse_mass = at::empty(
        {batch, heads, chunks, kChunk}, u.options());
    auto lower_primal = at::empty_like(key);
    auto lower_dual_scaled = at::empty(
        {batch, length, heads, kDualRhs, kRank}, u.options());

    const int panels = static_cast<int>(batch * heads * chunks);
    auto alpha0 = at::empty({panels}, u.options());
    auto coefficient = at::empty(
        {panels, kChunk, kComponents}, u.options());
    auto diagonal = at::empty({panels, kChunk, kRank}, u.options());
    const auto stream = at::cuda::getCurrentCUDAStream();

    radial_kernel<<<dim3(panels, 2), kThreads, 0, stream>>>(
        u.data_ptr<float>(),
        h.data_ptr<float>(),
        geometry_log_decay.data_ptr<float>(),
        geometry_strength.data_ptr<float>(),
        boundary_m.data_ptr<float>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        alpha0.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        coefficient.data_ptr<float>(),
        diagonal.data_ptr<float>(),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        frame_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sizeof(FrameShared))));
    frame_kernel<<<panels, kThreads, sizeof(FrameShared), stream>>>(
        u.data_ptr<float>(),
        h.data_ptr<float>(),
        key.data_ptr<float>(),
        erase.data_ptr<float>(),
        query.data_ptr<float>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        alpha0.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        coefficient.data_ptr<float>(),
        diagonal.data_ptr<float>(),
        write_direction.data_ptr<float>(),
        erase_direction.data_ptr<float>(),
        solved_query.data_ptr<float>(),
        lower_primal.data_ptr<float>(),
        lower_dual_scaled.data_ptr<float>(),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        write_direction,
        erase_direction,
        solved_query,
        lower_primal,
        lower_dual_scaled};
}


TORCH_LIBRARY(causallsso, m) {
    m.def("c32_frame_forward(Tensor u, Tensor h, Tensor geometry_log_decay, Tensor key, Tensor erase, Tensor query, Tensor geometry_strength, Tensor boundary_m, Tensor boundary_J, Tensor boundary_D) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("c32_frame_forward", &c32_frame_forward_cuda);
}
