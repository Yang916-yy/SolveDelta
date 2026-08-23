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
constexpr int kTile = 16;
constexpr int kTiles = kRank / kTile;
constexpr int kMatrixTiles = kTiles * kTiles;
constexpr int kComponents = 4;
constexpr int kDualRhs = 2;
constexpr int kActionThreads = kChunk * kTile;
constexpr int kFactorGroup = 16;
constexpr int kEntryThreads = kTile * kTile;
constexpr float kRadius = 1.0f / 8.0f;


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


__device__ __forceinline__ int dual_index(
    int batch_index,
    int token,
    int head,
    int rhs,
    int coordinate,
    int length,
    int heads) {
    return ((((batch_index * length + token) * heads + head) * kDualRhs
             + rhs) * kRank) + coordinate;
}


__device__ __forceinline__ void decode_panel(
    int panel,
    int heads,
    int chunks,
    int& batch_index,
    int& head,
    int& chunk) {
    chunk = panel % chunks;
    const int batch_head = panel / chunks;
    head = batch_head % heads;
    batch_index = batch_head / heads;
}


__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}


__device__ __forceinline__ void block_sum4(
    float& value0,
    float& value1,
    float& value2,
    float& value3,
    float* shared) {
    constexpr int warps = kEntryThreads / 32;
    // This result region must not alias partials written by the next token.
    constexpr int result = kComponents * warps;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value0 = warp_sum(value0);
    value1 = warp_sum(value1);
    value2 = warp_sum(value2);
    value3 = warp_sum(value3);
    if (lane == 0) {
        shared[0 * warps + warp] = value0;
        shared[1 * warps + warp] = value1;
        shared[2 * warps + warp] = value2;
        shared[3 * warps + warp] = value3;
    }
    __syncthreads();
    if (warp == 0) {
        value0 = lane < warps ? shared[0 * warps + lane] : 0.0f;
        value1 = lane < warps ? shared[1 * warps + lane] : 0.0f;
        value2 = lane < warps ? shared[2 * warps + lane] : 0.0f;
        value3 = lane < warps ? shared[3 * warps + lane] : 0.0f;
        value0 = warp_sum(value0);
        value1 = warp_sum(value1);
        value2 = warp_sum(value2);
        value3 = warp_sum(value3);
        __syncwarp();
        if (lane == 0) {
            shared[result + 0] = value0;
            shared[result + 1] = value1;
            shared[result + 2] = value2;
            shared[result + 3] = value3;
        }
    }
    __syncthreads();
    value0 = shared[result + 0];
    value1 = shared[result + 1];
    value2 = shared[result + 2];
    value3 = shared[result + 3];
}


__device__ __forceinline__ float block_sum128(
    float value,
    float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < 4 ? shared[lane] : 0.0f;
        value = warp_sum(value);
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}


__global__ void prefix_scalar_kernel(
    const float* __restrict__ geometry_log_decay,
    const float* __restrict__ boundary_m,
    float* __restrict__ inverse_mass,
    int length,
    int heads,
    int chunks,
    int panels) {
    const int panel = blockIdx.x;
    if (panel >= panels || threadIdx.x != 0) return;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    float mass = boundary_m[panel];
#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        float value = 0.0f;
        if (target < valid_count) {
            const int scalar =
                (batch_index * length + token_start + target) * heads + head;
            const float lambda = expf(geometry_log_decay[scalar]);
            mass = fmaf(lambda, mass, 1.0f);
            value = 1.0f / mass;
        }
        inverse_mass[panel * kChunk + target] = value;
    }
}


template <bool DescriptorInner>
__global__ __launch_bounds__(kEntryThreads, 2) void radial_tile_stat_kernel(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ geometry_log_decay,
    const float* __restrict__ key,
    const float* __restrict__ erase,
    const float* __restrict__ query,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ lower_primal,
    const float* __restrict__ lower_dual_scaled,
    const float* __restrict__ write_direction,
    const float* __restrict__ grad_write_direction,
    const float* __restrict__ grad_erase_direction,
    const float* __restrict__ grad_solved_query,
    const float* __restrict__ primal_upper_left,
    const float* __restrict__ primal_lower_left,
    const float* __restrict__ dual_lower_right0,
    const float* __restrict__ dual_lower_right1,
    float* __restrict__ tile_partial,
    int length,
    int heads,
    int chunks,
    int panels) {
    __shared__ float reduction[kComponents * (kEntryThreads / 32 + 1)];
    const int panel = blockIdx.x;
    const int tile = blockIdx.y;
    if (panel >= panels || tile >= kMatrixTiles) return;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    const int row = (tile / kTiles) * kTile + threadIdx.x / kTile;
    const int column = (tile % kTiles) * kTile + threadIdx.x % kTile;
    const int matrix_index = panel * kRank * kRank + row * kRank + column;
    float state_j = boundary_j[matrix_index];
    float state_d = boundary_d[matrix_index];

#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        float partial2 = 0.0f;
        float partial3 = 0.0f;
        if (target < valid_count) {
            const int token = token_start + target;
            const int row_index = vector_index(
                batch_index, token, head, row, length, heads);
            const int column_index = vector_index(
                batch_index, token, head, column, length, heads);
            const float weight = inverse_mass[panel * kChunk + target];
            const float outer_j = u[row_index] * u[column_index];
            const float outer_d = u[row_index] * h[column_index];
            if (target == 0) {
                const int scalar =
                    (batch_index * length + token) * heads + head;
                const float lambda = expf(geometry_log_decay[scalar]);
                state_j = fmaf(lambda * weight, state_j, weight * outer_j);
                state_d = fmaf(lambda * weight, state_d, weight * outer_d);
            } else {
                const float retain = 1.0f - weight;
                state_j = fmaf(retain, state_j, weight * outer_j);
                state_d = fmaf(retain, state_d, weight * outer_d);
            }

            float value = 1.0f;
            if constexpr (DescriptorInner) {
                if (row > column) {
                    const float b_row = erase[row_index] * key[row_index];
                    value = fmaf(
                        -primal_lower_left[row_index],
                        lower_primal[column_index],
                        fmaf(
                            b_row, dual_lower_right0[column_index],
                            query[row_index]
                                * dual_lower_right1[column_index]));
                } else if (row < column) {
                    value = fmaf(
                        -primal_upper_left[row_index],
                        write_direction[column_index],
                        fmaf(
                            lower_dual_scaled[dual_index(
                                batch_index, token, head, 0, row,
                                length, heads)],
                            grad_erase_direction[column_index],
                            lower_dual_scaled[dual_index(
                                batch_index, token, head, 1, row,
                                length, heads)]
                                * grad_solved_query[column_index]));
                } else {
                    value = 0.0f;
                }
            }
            if (row > column) {
                partial0 = DescriptorInner ? value * state_j : state_j * state_j;
                partial1 = DescriptorInner ? value * state_d : state_d * state_d;
            } else if (row < column) {
                partial2 = DescriptorInner ? value * state_j : state_j * state_j;
                partial3 = DescriptorInner ? value * state_d : state_d * state_d;
            }
        }
        block_sum4(partial0, partial1, partial2, partial3, reduction);
        if (threadIdx.x == 0) {
            const int output =
                (((panel * kMatrixTiles + tile) * kChunk + target)
                 * kComponents);
            tile_partial[output + 0] = partial0;
            tile_partial[output + 1] = partial1;
            tile_partial[output + 2] = partial2;
            tile_partial[output + 3] = partial3;
        }
    }
}


__global__ void radial_norm_reduce_kernel(
    const float* __restrict__ tile_partial,
    const float* __restrict__ geometry_strength,
    float* __restrict__ radial_a,
    float* __restrict__ radial_q2,
    int length,
    int heads,
    int chunks,
    int items) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= items) return;
    const int component = item % kComponents;
    const int token_panel = item / kComponents;
    const int target = token_panel % kChunk;
    const int panel = token_panel / kChunk;
    const int head = (panel / chunks) % heads;
    const int target_chunk = panel % chunks;
    const int valid_count = min(kChunk, length - target_chunk * kChunk);
    float norm_sq = 0.0f;
#pragma unroll
    for (int tile = 0; tile < kMatrixTiles; ++tile) {
        const int index =
            (((panel * kMatrixTiles + tile) * kChunk + target) * kComponents)
            + component;
        norm_sq += tile_partial[index];
    }
    const double strength = target < valid_count
        ? static_cast<double>(geometry_strength[head])
        : 0.0;
    const double q2 = static_cast<double>(kRadius * kRadius)
        + strength * strength * static_cast<double>(norm_sq);
    radial_q2[item] = static_cast<float>(q2);
    radial_a[item] = static_cast<float>(
        static_cast<double>(kRadius) * strength / sqrt(q2));
}


struct __align__(32) ActionShared {
    float solve[kChunk * kRank];
    float factor[kFactorGroup * kTile * kTile];
};


template <int DirectRhs>
__device__ __forceinline__ float direct_value(
    const float* __restrict__ direct0,
    const float* __restrict__ direct1,
    int batch_index,
    int token,
    int head,
    int coordinate,
    int item,
    int length,
    int heads) {
    if constexpr (DirectRhs == 1) {
        return direct0[vector_index(
            batch_index, token, head, coordinate, length, heads)];
    } else {
        const float* source = item == 0 ? direct0 : direct1;
        return source[vector_index(
            batch_index, token, head, coordinate, length, heads)];
    }
}


template <bool Upper, bool Diagonal, bool DoSolve, int DirectRhs>
__device__ __forceinline__ void apply_factor_group(
    int group,
    int row_start,
    int column_start,
    const float* __restrict__ direct0,
    const float* __restrict__ direct1,
    float& direct_accumulator0,
    float& direct_accumulator1,
    ActionShared& shared,
    int batch_index,
    int head,
    int token_start,
    int valid_count,
    int length,
    int heads) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int solve_target = group * kFactorGroup + warp;
    const int factor_base = warp * kTile * kTile;

    if constexpr (DoSolve && Diagonal) {
        float residual = solve_target < valid_count && lane < kTile
            ? shared.solve[solve_target * kRank + row_start + lane]
            : 0.0f;
#pragma unroll
        for (int step = 0; step < kTile; ++step) {
            const int pivot = Upper ? step : kTile - 1 - step;
            const float solved = __shfl_sync(0xffffffffu, residual, pivot);
            const bool active = Upper
                ? lane > pivot && lane < kTile
                : lane < pivot;
            if (active) {
                residual = fmaf(
                    -shared.factor[factor_base + pivot * kTile + lane],
                    solved,
                    residual);
            }
        }
        if (solve_target < valid_count && lane < kTile) {
            shared.solve[solve_target * kRank + row_start + lane] = residual;
        }
    }

    const int group_begin = group * (kActionThreads / 2);
    const bool owns_group =
        tid >= group_begin && tid < group_begin + kActionThreads / 2;
    if (!owns_group) return;
    const int local = tid - group_begin;
    const int target = group * kFactorGroup + local / kTile;
    const int coordinate = local % kTile;
    const int local_factor_base =
        (target - group * kFactorGroup) * kTile * kTile;

    if constexpr (DoSolve && !Diagonal) {
        if (target < valid_count) {
            float transpose_action = 0.0f;
#pragma unroll
            for (int inner = 0; inner < kTile; ++inner) {
                transpose_action = fmaf(
                    shared.factor[
                        local_factor_base + inner * kTile + coordinate],
                    shared.solve[target * kRank + row_start + inner],
                    transpose_action);
            }
            shared.solve[target * kRank + column_start + coordinate]
                -= transpose_action;
        }
    }

    if constexpr (DirectRhs > 0) {
        if (target < valid_count) {
            const int token = token_start + target;
            float action0 = 0.0f;
            float action1 = 0.0f;
#pragma unroll
            for (int inner = 0; inner < kTile; ++inner) {
                const bool strict = Upper
                    ? inner > coordinate
                    : inner < coordinate;
                if constexpr (!Diagonal) {
                    const float factor = shared.factor[
                        local_factor_base + coordinate * kTile + inner];
                    action0 = fmaf(
                        factor,
                        direct_value<DirectRhs>(
                            direct0,
                            direct1,
                            batch_index,
                            token,
                            head,
                            column_start + inner,
                            0,
                            length,
                            heads),
                        action0);
                    if constexpr (DirectRhs == 2) {
                        action1 = fmaf(
                            factor,
                            direct_value<DirectRhs>(
                                direct0,
                                direct1,
                                batch_index,
                                token,
                                head,
                                column_start + inner,
                                1,
                                length,
                                heads),
                            action1);
                    }
                } else if (strict) {
                    const float factor = shared.factor[
                        local_factor_base + coordinate * kTile + inner];
                    action0 = fmaf(
                        factor,
                        direct_value<DirectRhs>(
                            direct0,
                            direct1,
                            batch_index,
                            token,
                            head,
                            row_start + inner,
                            0,
                            length,
                            heads),
                        action0);
                    if constexpr (DirectRhs == 2) {
                        action1 = fmaf(
                            factor,
                            direct_value<DirectRhs>(
                                direct0,
                                direct1,
                                batch_index,
                                token,
                                head,
                                row_start + inner,
                                1,
                                length,
                                heads),
                            action1);
                    }
                }
            }
            direct_accumulator0 += action0;
            if constexpr (DirectRhs == 2) direct_accumulator1 += action1;
        }
    }
}


template <bool Upper, bool Diagonal, bool DoSolve, int DirectRhs>
__device__ __forceinline__ void process_factor_block(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ geometry_log_decay,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ radial_a,
    const float* __restrict__ direct0,
    const float* __restrict__ direct1,
    float& direct_accumulator0,
    float& direct_accumulator1,
    ActionShared& shared,
    int panel,
    int row_start,
    int column_start,
    int batch_index,
    int head,
    int token_start,
    int valid_count,
    int length,
    int heads) {
    const int tid = threadIdx.x;
    constexpr int component = Upper ? 2 : 0;
    float state_j = 0.0f;
    float state_d = 0.0f;
    int local_row = 0;
    int local_column = 0;
    if (tid < kTile * kTile) {
        local_row = tid / kTile;
        local_column = tid % kTile;
        const int row = row_start + local_row;
        const int column = column_start + local_column;
        const int matrix = panel * kRank * kRank + row * kRank + column;
        state_j = boundary_j[matrix];
        state_d = boundary_d[matrix];
    }

#pragma unroll
    for (int group = 0; group < 2; ++group) {
#pragma unroll
        for (int relative = 0; relative < kFactorGroup; ++relative) {
            const int target = group * kFactorGroup + relative;
            if (tid < kTile * kTile && target < valid_count) {
                const int token = token_start + target;
                const int row_index = vector_index(
                    batch_index,
                    token,
                    head,
                    row_start + local_row,
                    length,
                    heads);
                const int column_index = vector_index(
                    batch_index,
                    token,
                    head,
                    column_start + local_column,
                    length,
                    heads);
                const float weight = inverse_mass[panel * kChunk + target];
                const float outer_j = u[row_index] * u[column_index];
                const float outer_d = u[row_index] * h[column_index];
                if (target == 0) {
                    const int scalar =
                        (batch_index * length + token) * heads + head;
                    const float lambda = expf(geometry_log_decay[scalar]);
                    state_j = fmaf(
                        lambda * weight, state_j, weight * outer_j);
                    state_d = fmaf(
                        lambda * weight, state_d, weight * outer_d);
                } else {
                    const float retain = 1.0f - weight;
                    state_j = fmaf(retain, state_j, weight * outer_j);
                    state_d = fmaf(retain, state_d, weight * outer_d);
                }
                const int scalar =
                    (panel * kChunk + target) * kComponents + component;
                shared.factor[relative * kTile * kTile + tid] = fmaf(
                    radial_a[scalar],
                    state_j,
                    radial_a[scalar + 1] * state_d);
            } else if (tid < kTile * kTile) {
                shared.factor[relative * kTile * kTile + tid] = 0.0f;
            }
        }
        __syncthreads();
        apply_factor_group<Upper, Diagonal, DoSolve, DirectRhs>(
            group,
            row_start,
            column_start,
            direct0,
            direct1,
            direct_accumulator0,
            direct_accumulator1,
            shared,
            batch_index,
            head,
            token_start,
            valid_count,
            length,
            heads);
        __syncthreads();
    }
}


template <bool Upper, bool DoSolve, int DirectRhs>
__global__ __launch_bounds__(kActionThreads, 2) void action_kernel(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ geometry_log_decay,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ radial_a,
    const float* __restrict__ solve_rhs,
    const float* __restrict__ direct0,
    const float* __restrict__ direct1,
    float* __restrict__ solve_output,
    float* __restrict__ direct_output0,
    float* __restrict__ direct_output1,
    int length,
    int heads,
    int chunks,
    int panels) {
    __shared__ ActionShared shared;
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    const int tid = threadIdx.x;

    if constexpr (DoSolve) {
        for (int index = tid; index < kChunk * kRank;
             index += kActionThreads) {
            const int target = index / kRank;
            const int coordinate = index % kRank;
            shared.solve[index] = target < valid_count
                ? solve_rhs[vector_index(
                    batch_index,
                    token_start + target,
                    head,
                    coordinate,
                    length,
                    heads)]
                : 0.0f;
        }
        __syncthreads();
    }

#pragma unroll 1
    for (int block_step = 0; block_step < kTiles; ++block_step) {
        const int block = Upper ? block_step : kTiles - 1 - block_step;
        const int row_start = block * kTile;
        const int target = tid / kTile;
        const int coordinate = tid % kTile;
        float accumulator0 = 0.0f;
        float accumulator1 = 0.0f;
        if constexpr (DirectRhs > 0) {
            if (target < valid_count) {
                const int token = token_start + target;
                accumulator0 = direct_value<DirectRhs>(
                    direct0,
                    direct1,
                    batch_index,
                    token,
                    head,
                    row_start + coordinate,
                    0,
                    length,
                    heads);
                if constexpr (DirectRhs == 2) {
                    accumulator1 = direct_value<DirectRhs>(
                        direct0,
                        direct1,
                        batch_index,
                        token,
                        head,
                        row_start + coordinate,
                        1,
                        length,
                        heads);
                }
            }
        }

        process_factor_block<Upper, true, DoSolve, DirectRhs>(
            boundary_j,
            boundary_d,
            u,
            h,
            geometry_log_decay,
            inverse_mass,
            radial_a,
            direct0,
            direct1,
            accumulator0,
            accumulator1,
            shared,
            panel,
            row_start,
            row_start,
            batch_index,
            head,
            token_start,
            valid_count,
            length,
            heads);

        const int remaining = Upper ? kTiles - block - 1 : block;
#pragma unroll 1
        for (int relative = 0; relative < remaining; ++relative) {
            const int other = Upper ? block + 1 + relative : relative;
            process_factor_block<Upper, false, DoSolve, DirectRhs>(
                boundary_j,
                boundary_d,
                u,
                h,
                geometry_log_decay,
                inverse_mass,
                radial_a,
                direct0,
                direct1,
                accumulator0,
                accumulator1,
                shared,
                panel,
                row_start,
                other * kTile,
                batch_index,
                head,
                token_start,
                valid_count,
                length,
                heads);
        }
        if constexpr (DirectRhs > 0) {
            if (target < valid_count) {
                const int output = vector_index(
                    batch_index,
                    token_start + target,
                    head,
                    row_start + coordinate,
                    length,
                    heads);
                direct_output0[output] = accumulator0;
                if constexpr (DirectRhs == 2) {
                    direct_output1[output] = accumulator1;
                }
            }
        }
        __syncthreads();
    }

    if constexpr (DoSolve) {
        for (int index = tid; index < kChunk * kRank;
             index += kActionThreads) {
            const int target = index / kRank;
            if (target >= valid_count) continue;
            const int coordinate = index % kRank;
            solve_output[vector_index(
                batch_index,
                token_start + target,
                head,
                coordinate,
                length,
                heads)] = shared.solve[index];
        }
    }
}


__device__ __forceinline__ float replay_diagonal_scale(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ geometry_log_decay,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ inverse_mass,
    float strength,
    int panel,
    int batch_index,
    int head,
    int token_start,
    int target,
    int coordinate,
    int length,
    int heads) {
    const int matrix =
        panel * kRank * kRank + coordinate * kRank + coordinate;
    float state_j = boundary_j[matrix];
    float state_d = boundary_d[matrix];
#pragma unroll
    for (int source = 0; source < kChunk; ++source) {
        if (source > target) break;
        const int token = token_start + source;
        const int index = vector_index(
            batch_index, token, head, coordinate, length, heads);
        const float weight = inverse_mass[panel * kChunk + source];
        const float outer_j = u[index] * u[index];
        const float outer_d = u[index] * h[index];
        if (source == 0) {
            const int scalar = (batch_index * length + token) * heads + head;
            const float lambda = expf(geometry_log_decay[scalar]);
            state_j = fmaf(lambda * weight, state_j, weight * outer_j);
            state_d = fmaf(lambda * weight, state_d, weight * outer_d);
        } else {
            const float retain = 1.0f - weight;
            state_j = fmaf(retain, state_j, weight * outer_j);
            state_d = fmaf(retain, state_d, weight * outer_d);
        }
    }
    const float x_h = strength * (
        state_j - 1.0f / static_cast<float>(kRank));
    const float x_d = strength * state_d;
    return expf(
        kRadius * tanhf(x_h / kRadius)
        + kRadius * tanhf(x_d / kRadius));
}


__global__ void scale_action_kernel(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ geometry_log_decay,
    const float* __restrict__ geometry_strength,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ primal_upper_left,
    float* __restrict__ primal_lower_rhs,
    float* __restrict__ dual_lower_right0,
    float* __restrict__ dual_lower_right1,
    int length,
    int heads,
    int chunks,
    int vector_items) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= vector_items) return;
    const int coordinate = item % kRank;
    const int scalar = item / kRank;
    const int head = scalar % heads;
    const int batch_token = scalar / heads;
    const int token = batch_token % length;
    const int batch_index = batch_token / length;
    const int chunk = token / kChunk;
    const int target = token % kChunk;
    const int panel = (batch_index * heads + head) * chunks + chunk;
    const float diagonal = replay_diagonal_scale(
        u,
        h,
        geometry_log_decay,
        boundary_j,
        boundary_d,
        inverse_mass,
        geometry_strength[head],
        panel,
        batch_index,
        head,
        chunk * kChunk,
        target,
        coordinate,
        length,
        heads);
    primal_lower_rhs[item] = primal_upper_left[item] / diagonal;
    dual_lower_right0[item] *= diagonal;
    dual_lower_right1[item] *= diagonal;
}


__global__ __launch_bounds__(128, 2) void radial_inner_reduce_kernel(
    const float* __restrict__ tile_partial,
    const float* __restrict__ geometry_strength,
    const float* __restrict__ radial_a,
    float* __restrict__ radial_q2_to_p,
    float* __restrict__ radial_strength,
    int length,
    int heads,
    int chunks,
    int panels) {
    __shared__ float reduction[4];
    const int panel = blockIdx.x;
    const int item = threadIdx.x;
    float strength_value = 0.0f;
    if (panel < panels && item < kChunk * kComponents) {
        const int target = item / kComponents;
        const int component = item % kComponents;
        const int valid_count = min(kChunk, length - (panel % chunks) * kChunk);
        float inner = 0.0f;
#pragma unroll
        for (int tile = 0; tile < kMatrixTiles; ++tile) {
            inner += tile_partial[
                (((panel * kMatrixTiles + tile) * kChunk + target)
                 * kComponents) + component];
        }
        const int scalar = (panel * kChunk + target) * kComponents + component;
        if (target < valid_count) {
            const int head = (panel / chunks) % heads;
            const float strength = geometry_strength[head];
            const float q2 = radial_q2_to_p[scalar];
            const float a = radial_a[scalar];
            radial_q2_to_p[scalar] =
                -a * strength * strength / q2 * inner;
            strength_value = kRadius * kRadius * kRadius
                / (q2 * sqrtf(q2)) * inner;
        } else {
            radial_q2_to_p[scalar] = 0.0f;
        }
    }
    const float total = block_sum128(strength_value, reduction);
    if (threadIdx.x == 0 && panel < panels) radial_strength[panel] = total;
}


struct MomentContext {
    const float* u;
    const float* h;
    const float* geometry_log_decay;
    const float* key;
    const float* erase;
    const float* query;
    const float* boundary_j;
    const float* boundary_d;
    const float* inverse_mass;
    const float* lower_primal;
    const float* lower_dual_scaled;
    const float* write_direction;
    const float* grad_erase_direction;
    const float* grad_solved_query;
    const float* primal_upper_left;
    const float* primal_lower_left;
    const float* dual_lower_right0;
    const float* dual_lower_right1;
    const float* radial_a;
    const float* radial_p;
    float* grad_boundary_j;
    float* grad_boundary_d;
    int panel;
    int batch_index;
    int head;
    int token_start;
    int valid_count;
    int length;
    int heads;
    float strength;
};


__device__ __forceinline__ void chart_entry_vjp(
    const MomentContext& context,
    int target,
    int row,
    int column,
    float state_j,
    float state_d,
    float& state_bar_j,
    float& state_bar_d,
    float& diagonal_strength) {
    const int token = context.token_start + target;
    const int row_index = vector_index(
        context.batch_index,
        token,
        context.head,
        row,
        context.length,
        context.heads);
    const int column_index = vector_index(
        context.batch_index,
        token,
        context.head,
        column,
        context.length,
        context.heads);
    if (row > column) {
        const float factor_bar = fmaf(
            -context.primal_lower_left[row_index],
            context.lower_primal[column_index],
            fmaf(
                context.erase[row_index] * context.key[row_index],
                context.dual_lower_right0[column_index],
                context.query[row_index]
                    * context.dual_lower_right1[column_index]));
        const int scalar =
            (context.panel * kChunk + target) * kComponents;
        state_bar_j = fmaf(
            context.radial_a[scalar + 0],
            factor_bar,
            context.radial_p[scalar + 0] * state_j);
        state_bar_d = fmaf(
            context.radial_a[scalar + 1],
            factor_bar,
            context.radial_p[scalar + 1] * state_d);
    } else if (row < column) {
        const float factor_bar = fmaf(
            -context.primal_upper_left[row_index],
            context.write_direction[column_index],
            fmaf(
                context.lower_dual_scaled[dual_index(
                    context.batch_index,
                    token,
                    context.head,
                    0,
                    row,
                    context.length,
                    context.heads)],
                context.grad_erase_direction[column_index],
                context.lower_dual_scaled[dual_index(
                    context.batch_index,
                    token,
                    context.head,
                    1,
                    row,
                    context.length,
                    context.heads)]
                    * context.grad_solved_query[column_index]));
        const int scalar =
            (context.panel * kChunk + target) * kComponents;
        state_bar_j = fmaf(
            context.radial_a[scalar + 2],
            factor_bar,
            context.radial_p[scalar + 2] * state_j);
        state_bar_d = fmaf(
            context.radial_a[scalar + 3],
            factor_bar,
            context.radial_p[scalar + 3] * state_d);
    } else {
        const float base_h =
            state_j - 1.0f / static_cast<float>(kRank);
        const float tanh_h = tanhf(context.strength * base_h / kRadius);
        const float tanh_d = tanhf(context.strength * state_d / kRadius);
        const float diagonal = expf(
            kRadius * tanh_h + kRadius * tanh_d);
        const float inverse_diagonal = 1.0f / diagonal;
        const float grad_log_diagonal = fmaf(
            -context.primal_upper_left[row_index]
                * context.lower_primal[row_index],
            inverse_diagonal,
            fmaf(
                context.lower_dual_scaled[dual_index(
                    context.batch_index,
                    token,
                    context.head,
                    0,
                    row,
                    context.length,
                    context.heads)]
                    * context.dual_lower_right0[row_index],
                inverse_diagonal,
                context.lower_dual_scaled[dual_index(
                    context.batch_index,
                    token,
                    context.head,
                    1,
                    row,
                    context.length,
                    context.heads)]
                    * context.dual_lower_right1[row_index]
                    * inverse_diagonal));
        const float derivative_h = 1.0f - tanh_h * tanh_h;
        const float derivative_d = 1.0f - tanh_d * tanh_d;
        state_bar_j = context.strength * derivative_h * grad_log_diagonal;
        state_bar_d = context.strength * derivative_d * grad_log_diagonal;
        diagonal_strength = fmaf(
            grad_log_diagonal,
            derivative_h * base_h + derivative_d * state_d,
            diagonal_strength);
    }
}


constexpr int kEntryWarps = kEntryThreads / 32;
constexpr int kMomentPartial = kChunk + 2;
constexpr int kMomentPairs =
    kTiles + (kTiles - 1) * (kTiles / 2);


struct TilePairShared {
    float grad_u0[kChunk * kTile];
    float grad_h0[kChunk * kTile];
    float grad_u1[kChunk * kTile];
    float grad_h1[kChunk * kTile];
    float column_u[kChunk * kEntryWarps * kTile];
    float column_h[kChunk * kEntryWarps * kTile];
    float inverse_mass_warp[kChunk * kEntryWarps];
    float direct_lambda_warp[kEntryWarps];
    float diagonal_strength_warp[kEntryWarps];
};


__device__ __forceinline__ float half_warp_sum(float value) {
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset, 16);
    }
    return value;
}


__device__ __forceinline__ void replay_tile_entry(
    const MomentContext& context,
    int row_tile,
    int column_tile,
    float* row_grad_u,
    float* column_grad_u,
    float* column_grad_h,
    TilePairShared& shared) {
    const int local_row = threadIdx.x / kTile;
    const int local_column = threadIdx.x % kTile;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = row_tile * kTile + local_row;
    const int column = column_tile * kTile + local_column;
    float state_j[kChunk];
    float state_d[kChunk];
    const int matrix =
        context.panel * kRank * kRank + row * kRank + column;
    const float boundary_value_j = context.boundary_j[matrix];
    const float boundary_value_d = context.boundary_d[matrix];
    float current_j = boundary_value_j;
    float current_d = boundary_value_d;
#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        if (target >= context.valid_count) break;
        const int token = context.token_start + target;
        const int row_index = vector_index(
            context.batch_index,
            token,
            context.head,
            row,
            context.length,
            context.heads);
        const int column_index = vector_index(
            context.batch_index,
            token,
            context.head,
            column,
            context.length,
            context.heads);
        const float weight =
            context.inverse_mass[context.panel * kChunk + target];
        const float outer_j =
            context.u[row_index] * context.u[column_index];
        const float outer_d =
            context.u[row_index] * context.h[column_index];
        if (target == 0) {
            const int scalar =
                (context.batch_index * context.length + token)
                * context.heads + context.head;
            const float lambda = expf(context.geometry_log_decay[scalar]);
            current_j = fmaf(
                lambda * weight, current_j, weight * outer_j);
            current_d = fmaf(
                lambda * weight, current_d, weight * outer_d);
        } else {
            const float retain = 1.0f - weight;
            current_j = fmaf(retain, current_j, weight * outer_j);
            current_d = fmaf(retain, current_d, weight * outer_d);
        }
        state_j[target] = current_j;
        state_d[target] = current_d;
    }

    float carry_j = 0.0f;
    float carry_d = 0.0f;
    float boundary_bar_j = 0.0f;
    float boundary_bar_d = 0.0f;
    float direct_lambda_bar = 0.0f;
    float diagonal_strength = 0.0f;
#pragma unroll
    for (int target = kChunk - 1; target >= 0; --target) {
        if (target >= context.valid_count) continue;
        const int token = context.token_start + target;
        const int row_index = vector_index(
            context.batch_index,
            token,
            context.head,
            row,
            context.length,
            context.heads);
        const int column_index = vector_index(
            context.batch_index,
            token,
            context.head,
            column,
            context.length,
            context.heads);
        float chart_bar_j = 0.0f;
        float chart_bar_d = 0.0f;
        chart_entry_vjp(
            context,
            target,
            row,
            column,
            state_j[target],
            state_d[target],
            chart_bar_j,
            chart_bar_d,
            diagonal_strength);
        const float adjoint_j = chart_bar_j + carry_j;
        const float adjoint_d = chart_bar_d + carry_d;
        const float weight =
            context.inverse_mass[context.panel * kChunk + target];
        const float outer_bar_j = weight * adjoint_j;
        const float outer_bar_d = weight * adjoint_d;
        float row_u_value = fmaf(
            outer_bar_j,
            context.u[column_index],
            outer_bar_d * context.h[column_index]);
        row_u_value = half_warp_sum(row_u_value);
        if (local_column == 0) {
            const int output = target * kTile + local_row;
            row_grad_u[output] += row_u_value;
        }
        const float column_u_value =
            outer_bar_j * context.u[row_index];
        const float column_h_value =
            outer_bar_d * context.u[row_index];
        const float column_u_pair = column_u_value
            + __shfl_down_sync(
                0xffffffffu, column_u_value, kTile, 32);
        const float column_h_pair = column_h_value
            + __shfl_down_sync(
                0xffffffffu, column_h_value, kTile, 32);
        if (lane < kTile) {
            const int output =
                (target * kEntryWarps + warp) * kTile + local_column;
            shared.column_u[output] = column_u_pair;
            shared.column_h[output] = column_h_pair;
        }

        float inverse_mass_value = 0.0f;
        if (target == 0) {
            const int scalar =
                (context.batch_index * context.length + token)
                * context.heads + context.head;
            const float lambda = expf(context.geometry_log_decay[scalar]);
            inverse_mass_value = fmaf(
                adjoint_j,
                fmaf(
                    lambda,
                    boundary_value_j,
                    context.u[row_index] * context.u[column_index]),
                inverse_mass_value);
            inverse_mass_value = fmaf(
                adjoint_d,
                fmaf(
                    lambda,
                    boundary_value_d,
                    context.u[row_index] * context.h[column_index]),
                inverse_mass_value);
            direct_lambda_bar = fmaf(
                weight * adjoint_j,
                boundary_value_j,
                direct_lambda_bar);
            direct_lambda_bar = fmaf(
                weight * adjoint_d,
                boundary_value_d,
                direct_lambda_bar);
            boundary_bar_j = lambda * weight * adjoint_j;
            boundary_bar_d = lambda * weight * adjoint_d;
        } else {
            const float previous_j = state_j[target - 1];
            const float previous_d = state_d[target - 1];
            inverse_mass_value = fmaf(
                adjoint_j,
                context.u[row_index] * context.u[column_index]
                    - previous_j,
                inverse_mass_value);
            inverse_mass_value = fmaf(
                adjoint_d,
                context.u[row_index] * context.h[column_index]
                    - previous_d,
                inverse_mass_value);
            const float retain = 1.0f - weight;
            carry_j = retain * adjoint_j;
            carry_d = retain * adjoint_d;
        }
        inverse_mass_value = warp_sum(inverse_mass_value);
        if (lane == 0) {
            shared.inverse_mass_warp[
                target * kEntryWarps + warp] += inverse_mass_value;
        }
    }

    context.grad_boundary_j[matrix] = boundary_bar_j;
    context.grad_boundary_d[matrix] = boundary_bar_d;
    direct_lambda_bar = warp_sum(direct_lambda_bar);
    diagonal_strength = warp_sum(diagonal_strength);
    if (lane == 0) {
        shared.direct_lambda_warp[warp] += direct_lambda_bar;
        shared.diagonal_strength_warp[warp] += diagonal_strength;
    }
    __syncthreads();

    for (int item = threadIdx.x;
         item < kChunk * kTile;
         item += blockDim.x) {
        const int target = item / kTile;
        if (target < context.valid_count) {
            float total_u = 0.0f;
            float total_h = 0.0f;
#pragma unroll
            for (int source_warp = 0;
                 source_warp < kEntryWarps;
                 ++source_warp) {
                const int source =
                    (target * kEntryWarps + source_warp) * kTile
                    + item % kTile;
                total_u += shared.column_u[source];
                total_h += shared.column_h[source];
            }
            column_grad_u[item] += total_u;
            column_grad_h[item] += total_h;
        }
    }
    __syncthreads();
}


__global__ __launch_bounds__(kEntryThreads, 1) void moment_tile_pair_kernel(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ geometry_log_decay,
    const float* __restrict__ key,
    const float* __restrict__ erase,
    const float* __restrict__ query,
    const float* __restrict__ geometry_strength,
    const float* __restrict__ boundary_m,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ lower_primal,
    const float* __restrict__ lower_dual_scaled,
    const float* __restrict__ write_direction,
    const float* __restrict__ grad_erase_direction,
    const float* __restrict__ grad_solved_query,
    const float* __restrict__ primal_upper_left,
    const float* __restrict__ primal_lower_left,
    const float* __restrict__ dual_lower_right0,
    const float* __restrict__ dual_lower_right1,
    const float* __restrict__ radial_a,
    const float* __restrict__ radial_p,
    float* __restrict__ staged_grad_u,
    float* __restrict__ staged_grad_h,
    float* __restrict__ grad_boundary_j,
    float* __restrict__ grad_boundary_d,
    float* __restrict__ moment_partial,
    int length,
    int heads,
    int chunks,
    int panels,
    int phase) {
    __shared__ TilePairShared shared;
    const int panel = blockIdx.x;
    const int pair_index = blockIdx.y;
    const bool diagonal_phase = phase == 0;
    if (panel >= panels
        || (diagonal_phase && pair_index >= kTiles)
        || (!diagonal_phase && pair_index >= kTiles / 2)) return;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    MomentContext context{
        u,
        h,
        geometry_log_decay,
        key,
        erase,
        query,
        boundary_j,
        boundary_d,
        inverse_mass,
        lower_primal,
        lower_dual_scaled,
        write_direction,
        grad_erase_direction,
        grad_solved_query,
        primal_upper_left,
        primal_lower_left,
        dual_lower_right0,
        dual_lower_right1,
        radial_a,
        radial_p,
        grad_boundary_j,
        grad_boundary_d,
        panel,
        batch_index,
        head,
        token_start,
        valid_count,
        length,
        heads,
        geometry_strength[head]};

    for (int item = threadIdx.x;
         item < kChunk * kTile;
         item += blockDim.x) {
        shared.grad_u0[item] = 0.0f;
        shared.grad_h0[item] = 0.0f;
        shared.grad_u1[item] = 0.0f;
        shared.grad_h1[item] = 0.0f;
    }
    for (int item = threadIdx.x;
         item < kChunk * kEntryWarps;
         item += blockDim.x) {
        shared.inverse_mass_warp[item] = 0.0f;
    }
    if (threadIdx.x < kEntryWarps) {
        shared.direct_lambda_warp[threadIdx.x] = 0.0f;
        shared.diagonal_strength_warp[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int tile0;
    int tile1;
    int partial_slot;
    if (diagonal_phase) {
        tile0 = pair_index;
        tile1 = pair_index;
        partial_slot = pair_index;
        replay_tile_entry(
            context,
            tile0,
            tile0,
            shared.grad_u0,
            shared.grad_u0,
            shared.grad_h0,
            shared);
    } else {
        const int round = phase - 1;
        if (pair_index == 0) {
            tile0 = kTiles - 1;
            tile1 = round;
        } else {
            tile0 = (round + pair_index) % (kTiles - 1);
            tile1 = (round - pair_index + kTiles - 1) % (kTiles - 1);
        }
        partial_slot =
            kTiles + round * (kTiles / 2) + pair_index;
        replay_tile_entry(
            context,
            tile0,
            tile1,
            shared.grad_u0,
            shared.grad_u1,
            shared.grad_h1,
            shared);
        replay_tile_entry(
            context,
            tile1,
            tile0,
            shared.grad_u1,
            shared.grad_u0,
            shared.grad_h0,
            shared);
    }
    __syncthreads();

    if (threadIdx.x < kMomentPartial) {
        float total = 0.0f;
        if (threadIdx.x < kChunk) {
#pragma unroll
            for (int warp = 0; warp < kEntryWarps; ++warp) {
                total += shared.inverse_mass_warp[
                    threadIdx.x * kEntryWarps + warp];
            }
        } else if (threadIdx.x == kChunk) {
#pragma unroll
            for (int warp = 0; warp < kEntryWarps; ++warp) {
                total += shared.direct_lambda_warp[warp];
            }
        } else {
#pragma unroll
            for (int warp = 0; warp < kEntryWarps; ++warp) {
                total += shared.diagonal_strength_warp[warp];
            }
        }
        moment_partial[
            (panel * kMomentPairs + partial_slot) * kMomentPartial
                + threadIdx.x]
            = total;
    }

    for (int item = threadIdx.x;
         item < kChunk * kTile;
         item += blockDim.x) {
        const int target = item / kTile;
        if (target >= valid_count) continue;
        const int index0 = vector_index(
            batch_index,
            token_start + target,
            head,
            tile0 * kTile + item % kTile,
            length,
            heads);
        staged_grad_u[index0] += shared.grad_u0[item];
        staged_grad_h[index0] += shared.grad_h0[item];
        if (!diagonal_phase) {
            const int index1 = vector_index(
                batch_index,
                token_start + target,
                head,
                tile1 * kTile + item % kTile,
                length,
                heads);
            staged_grad_u[index1] += shared.grad_u1[item];
            staged_grad_h[index1] += shared.grad_h1[item];
        }
    }
}


__global__ void moment_scalar_reduce_kernel(
    const float* __restrict__ geometry_log_decay,
    const float* __restrict__ boundary_m,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ radial_strength,
    const float* __restrict__ moment_partial,
    float* __restrict__ grad_geometry_log_decay,
    float* __restrict__ grad_boundary_m,
    float* __restrict__ panel_strength,
    int length,
    int heads,
    int chunks,
    int panels) {
    const int panel = blockIdx.x;
    if (panel >= panels || threadIdx.x != 0) return;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    float direct_lambda_total = 0.0f;
    float diagonal_strength_total = 0.0f;
#pragma unroll
    for (int pair = 0; pair < kMomentPairs; ++pair) {
        const int base =
            (panel * kMomentPairs + pair) * kMomentPartial;
        direct_lambda_total += moment_partial[base + kChunk];
        diagonal_strength_total += moment_partial[base + kChunk + 1];
    }

    double mass_bar = 0.0;
    for (int target = valid_count - 1; target >= 0; --target) {
        float inverse_mass_bar = 0.0f;
#pragma unroll
        for (int pair = 0; pair < kMomentPairs; ++pair) {
            inverse_mass_bar += moment_partial[
                (panel * kMomentPairs + pair) * kMomentPartial + target];
        }
        const double weight = static_cast<double>(
            inverse_mass[panel * kChunk + target]);
        mass_bar -= static_cast<double>(inverse_mass_bar) * weight * weight;
        const double previous_mass = target == 0
            ? static_cast<double>(boundary_m[panel])
            : 1.0 / static_cast<double>(
                inverse_mass[panel * kChunk + target - 1]);
        double lambda_bar = mass_bar * previous_mass;
        if (target == 0) {
            lambda_bar += static_cast<double>(direct_lambda_total);
        }
        const int token = token_start + target;
        const int scalar =
            (batch_index * length + token) * heads + head;
        const double lambda = exp(
            static_cast<double>(geometry_log_decay[scalar]));
        grad_geometry_log_decay[scalar] = static_cast<float>(
            lambda * lambda_bar);
        mass_bar *= lambda;
    }
    grad_boundary_m[panel] = static_cast<float>(mass_bar);
    panel_strength[panel] =
        radial_strength[panel] + diagonal_strength_total;
}


__global__ void finalize_edit_inputs_kernel(
    const float* __restrict__ key,
    const float* __restrict__ erase,
    const float* __restrict__ staged_grad_u,
    const float* __restrict__ staged_grad_h,
    float* __restrict__ grad_u,
    float* __restrict__ grad_h,
    float* __restrict__ grad_key,
    float* __restrict__ grad_erase,
    int items) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= items) return;
    const float grad_b = grad_erase[index];
    grad_u[index] = staged_grad_u[index];
    grad_h[index] = staged_grad_h[index];
    grad_key[index] = fmaf(erase[index], grad_b, grad_key[index]);
    grad_erase[index] = key[index] * grad_b;
}


__global__ void reduce_strength_kernel(
    const float* __restrict__ panel_strength,
    float* __restrict__ grad_strength,
    int batch,
    int heads,
    int chunks) {
    const int head = blockIdx.x;
    if (head >= heads || threadIdx.x != 0) return;
    float total = 0.0f;
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        for (int chunk = 0; chunk < chunks; ++chunk) {
            total += panel_strength[
                (batch_index * heads + head) * chunks + chunk];
        }
    }
    grad_strength[head] = total;
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


C32BackwardResult c32_frame_backward_cuda(
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& geometry_log_decay,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& geometry_strength,
    const at::Tensor& boundary_m,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& lower_primal,
    const at::Tensor& lower_dual_scaled,
    const at::Tensor& write_direction,
    const at::Tensor& grad_write_direction,
    const at::Tensor& grad_erase_direction,
    const at::Tensor& grad_solved_query) {
    TORCH_CHECK(
        u.is_cuda() && u.scalar_type() == at::kFloat && u.is_contiguous(),
        "u must be contiguous FP32 CUDA");
    TORCH_CHECK(
        u.dim() == 4 && u.size(3) == kRank,
        "u must be [B,T,H,128]");
    const int64_t batch = u.size(0);
    const int64_t length = u.size(1);
    const int64_t heads = u.size(2);
    TORCH_CHECK(
        batch > 0 && length > 0 && heads > 0,
        "B, T, and H must be positive");
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
             {&boundary_d, "boundary_D"},
             {&lower_primal, "lower_primal"},
             {&lower_dual_scaled, "lower_dual_scaled"},
             {&write_direction, "write_direction"},
             {&grad_write_direction, "grad_write_direction"},
             {&grad_erase_direction, "grad_erase_direction"},
             {&grad_solved_query, "grad_solved_query"}}) {
        check_fp32_cuda_contiguous(*named.first, u, named.second);
    }
    TORCH_CHECK(
        h.sizes() == u.sizes() && query.sizes() == u.sizes(),
        "h/query shape mismatch");
    TORCH_CHECK(
        geometry_log_decay.sizes()
            == at::IntArrayRef({batch, length, heads}),
        "geometry_log_decay must be [B,T,H]");
    TORCH_CHECK(
        key.sizes()
            == at::IntArrayRef({batch, length, heads, 1, kRank})
            && erase.sizes() == key.sizes(),
        "key/erase must be [B,T,H,1,128]");
    TORCH_CHECK(
        geometry_strength.sizes() == at::IntArrayRef({heads}),
        "geometry_strength must be [H]");
    TORCH_CHECK(
        boundary_m.sizes() == at::IntArrayRef({batch, heads, chunks}),
        "boundary_m must be [B,H,N]");
    TORCH_CHECK(
        boundary_j.sizes()
            == at::IntArrayRef({batch, heads, chunks, kRank, kRank})
            && boundary_d.sizes() == boundary_j.sizes(),
        "boundary_J/D must be [B,H,N,128,128]");
    TORCH_CHECK(
        lower_primal.sizes() == key.sizes()
            && write_direction.sizes() == key.sizes()
            && grad_write_direction.sizes() == key.sizes()
            && grad_erase_direction.sizes() == key.sizes(),
        "primal/edit tensors must be [B,T,H,1,128]");
    TORCH_CHECK(
        lower_dual_scaled.sizes()
            == at::IntArrayRef({batch, length, heads, kDualRhs, kRank}),
        "lower_dual_scaled must be [B,T,H,2,128]");
    TORCH_CHECK(
        grad_solved_query.sizes() == query.sizes(),
        "grad_solved_query shape mismatch");

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
        "the C32 backward contains only the SM120 specialization; got SM",
        properties.major,
        properties.minor);

    auto grad_u = at::empty_like(u);
    auto grad_h = at::empty_like(h);
    auto grad_geometry_log_decay = at::empty_like(geometry_log_decay);
    auto grad_key = at::empty_like(key);
    auto grad_erase_input = at::empty_like(erase);
    auto grad_query_input = at::empty_like(query);
    auto grad_geometry_strength = at::empty_like(geometry_strength);
    auto grad_boundary_m = at::empty_like(boundary_m);
    auto grad_boundary_j = at::empty_like(boundary_j);
    auto grad_boundary_d = at::empty_like(boundary_d);

    const int panels = static_cast<int>(boundary_m.numel());
    const int vector_items = static_cast<int>(u.numel());
    auto inverse_mass = at::empty({panels, kChunk}, u.options());
    auto radial_a = at::empty({panels, kChunk, kComponents}, u.options());
    auto radial_q2 = at::empty_like(radial_a);
    auto tile_partial = at::empty(
        {panels, kMatrixTiles, kChunk, kComponents}, u.options());
    auto radial_strength = at::empty({panels}, u.options());
    auto panel_strength = at::empty({panels}, u.options());
    auto moment_partial = at::empty(
        {panels, kMomentPairs, kMomentPartial}, u.options());
    TORCH_CHECK(
        tile_partial.numel() >= 2 * u.numel(),
        "the C32 moment workspace is smaller than its two staged vectors");

    const int length_i = static_cast<int>(length);
    const int heads_i = static_cast<int>(heads);
    const int chunks_i = static_cast<int>(chunks);
    const auto stream = at::cuda::getCurrentCUDAStream();
    prefix_scalar_kernel<<<panels, 1, 0, stream>>>(
        geometry_log_decay.data_ptr<float>(),
        boundary_m.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        length_i,
        heads_i,
        chunks_i,
        panels);
    radial_tile_stat_kernel<false><<<
        dim3(panels, kMatrixTiles), kEntryThreads, 0, stream>>>(
        u.data_ptr<float>(),
        h.data_ptr<float>(),
        geometry_log_decay.data_ptr<float>(),
        key.data_ptr<float>(),
        erase.data_ptr<float>(),
        query.data_ptr<float>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        lower_primal.data_ptr<float>(),
        lower_dual_scaled.data_ptr<float>(),
        write_direction.data_ptr<float>(),
        grad_write_direction.data_ptr<float>(),
        grad_erase_direction.data_ptr<float>(),
        grad_solved_query.data_ptr<float>(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        tile_partial.data_ptr<float>(),
        length_i,
        heads_i,
        chunks_i,
        panels);
    constexpr int linear_threads = 256;
    const int radial_items = panels * kChunk * kComponents;
    radial_norm_reduce_kernel<<<
        (radial_items + linear_threads - 1) / linear_threads,
        linear_threads,
        0,
        stream>>>(
        tile_partial.data_ptr<float>(),
        geometry_strength.data_ptr<float>(),
        radial_a.data_ptr<float>(),
        radial_q2.data_ptr<float>(),
        length_i,
        heads_i,
        chunks_i,
        radial_items);

    action_kernel<true, true, 2><<<panels, kActionThreads, 0, stream>>>(
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        u.data_ptr<float>(),
        h.data_ptr<float>(),
        geometry_log_decay.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        radial_a.data_ptr<float>(),
        grad_write_direction.data_ptr<float>(),
        grad_erase_direction.data_ptr<float>(),
        grad_solved_query.data_ptr<float>(),
        grad_u.data_ptr<float>(),
        grad_h.data_ptr<float>(),
        grad_query_input.data_ptr<float>(),
        length_i,
        heads_i,
        chunks_i,
        panels);
    scale_action_kernel<<<
        (vector_items + linear_threads - 1) / linear_threads,
        linear_threads,
        0,
        stream>>>(
        u.data_ptr<float>(),
        h.data_ptr<float>(),
        geometry_log_decay.data_ptr<float>(),
        geometry_strength.data_ptr<float>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        grad_u.data_ptr<float>(),
        grad_key.data_ptr<float>(),
        grad_h.data_ptr<float>(),
        grad_query_input.data_ptr<float>(),
        length_i,
        heads_i,
        chunks_i,
        vector_items);
    action_kernel<false, true, 1><<<panels, kActionThreads, 0, stream>>>(
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        u.data_ptr<float>(),
        h.data_ptr<float>(),
        geometry_log_decay.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        radial_a.data_ptr<float>(),
        grad_key.data_ptr<float>(),
        grad_h.data_ptr<float>(),
        nullptr,
        grad_key.data_ptr<float>(),
        grad_erase_input.data_ptr<float>(),
        nullptr,
        length_i,
        heads_i,
        chunks_i,
        panels);

    radial_tile_stat_kernel<true><<<
        dim3(panels, kMatrixTiles), kEntryThreads, 0, stream>>>(
        u.data_ptr<float>(),
        h.data_ptr<float>(),
        geometry_log_decay.data_ptr<float>(),
        key.data_ptr<float>(),
        erase.data_ptr<float>(),
        query.data_ptr<float>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        lower_primal.data_ptr<float>(),
        lower_dual_scaled.data_ptr<float>(),
        write_direction.data_ptr<float>(),
        grad_write_direction.data_ptr<float>(),
        grad_erase_direction.data_ptr<float>(),
        grad_solved_query.data_ptr<float>(),
        grad_u.data_ptr<float>(),
        grad_key.data_ptr<float>(),
        grad_h.data_ptr<float>(),
        grad_query_input.data_ptr<float>(),
        tile_partial.data_ptr<float>(),
        length_i,
        heads_i,
        chunks_i,
        panels);
    radial_inner_reduce_kernel<<<panels, 128, 0, stream>>>(
        tile_partial.data_ptr<float>(),
        geometry_strength.data_ptr<float>(),
        radial_a.data_ptr<float>(),
        radial_q2.data_ptr<float>(),
        radial_strength.data_ptr<float>(),
        length_i,
        heads_i,
        chunks_i,
        panels);
    float* staged_grad_u = tile_partial.data_ptr<float>();
    float* staged_grad_h = staged_grad_u + vector_items;
    C10_CUDA_CHECK(cudaMemsetAsync(
        staged_grad_u,
        0,
        static_cast<size_t>(2) * vector_items * sizeof(float),
        stream));
    for (int phase = 0; phase < kTiles; ++phase) {
        const int pairs = phase == 0 ? kTiles : kTiles / 2;
        moment_tile_pair_kernel<<<
            dim3(panels, pairs), kEntryThreads, 0, stream>>>(
            u.data_ptr<float>(),
            h.data_ptr<float>(),
            geometry_log_decay.data_ptr<float>(),
            key.data_ptr<float>(),
            erase.data_ptr<float>(),
            query.data_ptr<float>(),
            geometry_strength.data_ptr<float>(),
            boundary_m.data_ptr<float>(),
            boundary_j.data_ptr<float>(),
            boundary_d.data_ptr<float>(),
            inverse_mass.data_ptr<float>(),
            lower_primal.data_ptr<float>(),
            lower_dual_scaled.data_ptr<float>(),
            write_direction.data_ptr<float>(),
            grad_erase_direction.data_ptr<float>(),
            grad_solved_query.data_ptr<float>(),
            grad_u.data_ptr<float>(),
            grad_key.data_ptr<float>(),
            grad_h.data_ptr<float>(),
            grad_query_input.data_ptr<float>(),
            radial_a.data_ptr<float>(),
            radial_q2.data_ptr<float>(),
            staged_grad_u,
            staged_grad_h,
            grad_boundary_j.data_ptr<float>(),
            grad_boundary_d.data_ptr<float>(),
            moment_partial.data_ptr<float>(),
            length_i,
            heads_i,
            chunks_i,
            panels,
            phase);
    }
    moment_scalar_reduce_kernel<<<panels, 1, 0, stream>>>(
        geometry_log_decay.data_ptr<float>(),
        boundary_m.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        radial_strength.data_ptr<float>(),
        moment_partial.data_ptr<float>(),
        grad_geometry_log_decay.data_ptr<float>(),
        grad_boundary_m.data_ptr<float>(),
        panel_strength.data_ptr<float>(),
        length_i,
        heads_i,
        chunks_i,
        panels);

    action_kernel<false, false, 1><<<panels, kActionThreads, 0, stream>>>(
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        u.data_ptr<float>(),
        h.data_ptr<float>(),
        geometry_log_decay.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        radial_a.data_ptr<float>(),
        nullptr,
        grad_query_input.data_ptr<float>(),
        nullptr,
        nullptr,
        grad_query_input.data_ptr<float>(),
        nullptr,
        length_i,
        heads_i,
        chunks_i,
        panels);
    finalize_edit_inputs_kernel<<<
        (vector_items + linear_threads - 1) / linear_threads,
        linear_threads,
        0,
        stream>>>(
        key.data_ptr<float>(),
        erase.data_ptr<float>(),
        staged_grad_u,
        staged_grad_h,
        grad_u.data_ptr<float>(),
        grad_h.data_ptr<float>(),
        grad_key.data_ptr<float>(),
        grad_erase_input.data_ptr<float>(),
        vector_items);
    reduce_strength_kernel<<<heads_i, 1, 0, stream>>>(
        panel_strength.data_ptr<float>(),
        grad_geometry_strength.data_ptr<float>(),
        static_cast<int>(batch),
        heads_i,
        chunks_i);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        grad_u,
        grad_h,
        grad_geometry_log_decay,
        grad_key,
        grad_erase_input,
        grad_query_input,
        grad_geometry_strength,
        grad_boundary_m,
        grad_boundary_j,
        grad_boundary_d};
}


TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "c32_frame_backward(Tensor u, Tensor h, Tensor geometry_log_decay, "
        "Tensor key, Tensor erase, Tensor query, Tensor geometry_strength, "
        "Tensor boundary_m, Tensor boundary_J, Tensor boundary_D, "
        "Tensor lower_primal, Tensor lower_dual_scaled, Tensor write_direction, "
        "Tensor grad_write_direction, Tensor grad_erase_direction, "
        "Tensor grad_solved_query) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}


TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("c32_frame_backward", &c32_frame_backward_cuda);
}
