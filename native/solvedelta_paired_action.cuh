#pragma once

#include <ATen/ATen.h>


namespace solvedelta::paired_action {

constexpr int kRank = 128;
constexpr int kChunk = 32;
constexpr int kTile = 16;
constexpr int kTiles = kRank / kTile;
constexpr int kRoutes = 2;
constexpr int kComponents = 4;

struct alignas(16) PairedActionShared {
    float primal[kChunk * kRank];
    at::Half factor[kChunk * kTile * kTile];
    at::Half u_left[kChunk * kTile];
    at::Half u_right[kChunk * kTile];
    at::BFloat16 h_right[kChunk * kTile];
    float boundary_j[kTile * kTile];
    float boundary_d[kTile * kTile];
    float inverse_mass[kChunk];
    float coefficient[kChunk * kComponents];
    float dual_accumulator[kRoutes * kChunk * kTile];
};

static_assert(sizeof(PairedActionShared) == 42624);

template <bool FactorUpper, bool TransposeFactor, class Geometry>
__device__ __forceinline__ void reconstruct_factor_tile(
    PairedActionShared& shared,
    const Geometry& geometry,
    int effective_row_start,
    int effective_column_start) {
    const int tid = threadIdx.x;
    const int original_row_start = TransposeFactor
        ? effective_column_start
        : effective_row_start;
    const int original_column_start = TransposeFactor
        ? effective_row_start
        : effective_column_start;

    if (tid < kChunk * kTile) {
        const int target = tid / kTile;
        const int coordinate = tid % kTile;
        shared.u_left[tid] = geometry.u(
            target, original_row_start + coordinate);
        shared.u_right[tid] = geometry.u(
            target, original_column_start + coordinate);
        shared.h_right[tid] = geometry.h(
            target, original_column_start + coordinate);
    }
    if (tid < kTile * kTile) {
        const int effective_row = tid / kTile;
        const int effective_column = tid % kTile;
        const int original_row = TransposeFactor
            ? original_row_start + effective_column
            : original_row_start + effective_row;
        const int original_column = TransposeFactor
            ? original_column_start + effective_row
            : original_column_start + effective_column;
        shared.boundary_j[tid] = geometry.boundary_j(
            original_row, original_column);
        shared.boundary_d[tid] = geometry.boundary_d(
            original_row, original_column);
    }
    __syncthreads();

    constexpr int component = FactorUpper ? 2 : 0;
    if (tid < kTile * kTile) {
        const int row = tid / kTile;
        const int column = tid % kTile;
        float moment_j = geometry.alpha0() * shared.boundary_j[tid];
        float moment_d = geometry.alpha0() * shared.boundary_d[tid];
#pragma unroll 1
        for (int target = 0; target < kChunk; ++target) {
            const float weight = shared.inverse_mass[target];
            if (target > 0) {
                const float retain = 1.0f - weight;
                moment_j *= retain;
                moment_d *= retain;
            }
            const int left_coordinate = TransposeFactor ? column : row;
            const int right_coordinate = TransposeFactor ? row : column;
            const float row_u = static_cast<float>(
                shared.u_left[target * kTile + left_coordinate]);
            moment_j = fmaf(
                weight * row_u,
                static_cast<float>(
                    shared.u_right[target * kTile + right_coordinate]),
                moment_j);
            moment_d = fmaf(
                weight * row_u,
                static_cast<float>(
                    shared.h_right[target * kTile + right_coordinate]),
                moment_d);
            const float value = fmaf(
                shared.coefficient[target * kComponents + component],
                moment_j,
                shared.coefficient[
                    target * kComponents + component + 1] * moment_d);
            shared.factor[target * kTile * kTile + tid] =
                target < geometry.valid_count()
                ? at::Half(value)
                : at::Half(0.0f);
        }
    }
    __syncthreads();
}

template <bool EffectiveUpper>
__device__ __forceinline__ void solve_diagonal_tile(
    PairedActionShared& shared,
    int coordinate_start) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int group = 0; group < 2; ++group) {
        const int target = group * 16 + warp;
        float value = lane < kTile
            ? shared.primal[target * kRank + coordinate_start + lane]
            : 0.0f;
#pragma unroll
        for (int step = 0; step < kTile; ++step) {
            const int pivot = EffectiveUpper ? kTile - 1 - step : step;
            const float solved = __shfl_sync(0xffffffffu, value, pivot);
            const bool active = EffectiveUpper
                ? lane < pivot
                : lane > pivot && lane < kTile;
            if (active) {
                value = fmaf(
                    -static_cast<float>(shared.factor[
                        target * kTile * kTile + lane * kTile + pivot]),
                    solved,
                    value);
            }
        }
        if (lane < kTile) {
            shared.primal[target * kRank + coordinate_start + lane] = value;
        }
    }
    __syncthreads();
}

template <bool EffectiveUpper, bool Diagonal, class DualInput>
__device__ __forceinline__ void accumulate_dual_tile(
    PairedActionShared& shared,
    const DualInput& dual_input,
    int source_start) {
    const int tid = threadIdx.x;
    const int target = tid / kTile;
    const int column = tid % kTile;
#pragma unroll
    for (int route = 0; route < kRoutes; ++route) {
        float value = 0.0f;
#pragma unroll
        for (int row = 0; row < kTile; ++row) {
            bool active = true;
            if constexpr (Diagonal) {
                active = EffectiveUpper ? row < column : row > column;
            }
            if (active) {
                const float rhs = dual_input.load(
                    target, route, source_start + row);
                value = fmaf(
                    static_cast<float>(shared.factor[
                        target * kTile * kTile + row * kTile + column]),
                    rhs,
                    value);
            }
        }
        shared.dual_accumulator[
            (route * kChunk + target) * kTile + column] += value;
    }
}

template <class DualInput>
__device__ __forceinline__ void initialize_dual_tile(
    PairedActionShared& shared,
    const DualInput& dual_input,
    int coordinate_start) {
    for (int index = threadIdx.x;
         index < kRoutes * kChunk * kTile;
         index += blockDim.x) {
        const int route = index / (kChunk * kTile);
        const int local = index % (kChunk * kTile);
        const int target = local / kTile;
        const int coordinate = local % kTile;
        shared.dual_accumulator[index] = dual_input.load(
            target, route, coordinate_start + coordinate);
    }
    __syncthreads();
}

template <class DualOutput>
__device__ __forceinline__ void store_dual_tile(
    PairedActionShared& shared,
    const DualOutput& dual_output,
    int coordinate_start,
    int valid_count) {
    for (int index = threadIdx.x;
         index < kRoutes * kChunk * kTile;
         index += blockDim.x) {
        const int route = index / (kChunk * kTile);
        const int local = index % (kChunk * kTile);
        const int target = local / kTile;
        const int coordinate = local % kTile;
        if (target < valid_count) {
            dual_output.store(
                target,
                route,
                coordinate_start + coordinate,
                shared.dual_accumulator[index]);
        }
    }
    __syncthreads();
}

template <bool EffectiveUpper>
__device__ __forceinline__ void update_primal_tile(
    PairedActionShared& shared,
    int row_start,
    int solved_start) {
    const int tid = threadIdx.x;
    const int target = tid / kTile;
    const int row = tid % kTile;
    float action = 0.0f;
#pragma unroll
    for (int column = 0; column < kTile; ++column) {
        action = fmaf(
            static_cast<float>(shared.factor[
                target * kTile * kTile + row * kTile + column]),
            shared.primal[target * kRank + solved_start + column],
            action);
    }
    shared.primal[target * kRank + row_start + row] -= action;
}

template <
    bool FactorUpper,
    bool TransposeFactor,
    class Geometry,
    class PrimalInput,
    class PrimalOutput,
    class DualInput,
    class DualOutput>
__device__ __forceinline__ void run_paired_action(
    PairedActionShared& shared,
    const Geometry& geometry,
    const PrimalInput& primal_input,
    const PrimalOutput& primal_output,
    const DualInput& dual_input,
    const DualOutput& dual_output) {
    constexpr bool effective_upper = FactorUpper != TransposeFactor;
    const int tid = threadIdx.x;

    if (tid < kChunk) {
        shared.inverse_mass[tid] = geometry.inverse_mass(tid);
        for (int component = 0; component < kComponents; ++component) {
            shared.coefficient[tid * kComponents + component] =
                geometry.coefficient(tid, component);
        }
    }
    for (int index = tid; index < kChunk * kRank; index += blockDim.x) {
        const int target = index / kRank;
        const int coordinate = index % kRank;
        shared.primal[index] = target < geometry.valid_count()
            ? primal_input.load(target, 0, coordinate)
            : 0.0f;
    }
    __syncthreads();

#pragma unroll 1
    for (int step = 0; step < kTiles; ++step) {
        const int tile = effective_upper ? kTiles - 1 - step : step;
        const int solved_start = tile * kTile;
        initialize_dual_tile(shared, dual_input, solved_start);
        reconstruct_factor_tile<FactorUpper, TransposeFactor>(
            shared, geometry, solved_start, solved_start);
        solve_diagonal_tile<effective_upper>(shared, solved_start);
        accumulate_dual_tile<effective_upper, true>(
            shared, dual_input, solved_start);
        __syncthreads();

        const int row_begin = effective_upper ? 0 : tile + 1;
        const int row_end = effective_upper ? tile : kTiles;
        for (int row_tile = row_begin; row_tile < row_end; ++row_tile) {
            const int row_start = row_tile * kTile;
            reconstruct_factor_tile<FactorUpper, TransposeFactor>(
                shared, geometry, row_start, solved_start);
            update_primal_tile<effective_upper>(
                shared, row_start, solved_start);
            accumulate_dual_tile<effective_upper, false>(
                shared, dual_input, row_start);
            __syncthreads();
        }
        store_dual_tile(
            shared, dual_output, solved_start, geometry.valid_count());
    }

    for (int index = tid; index < kChunk * kRank; index += blockDim.x) {
        const int target = index / kRank;
        const int coordinate = index % kRank;
        if (target < geometry.valid_count()) {
            primal_output.store(
                target, 0, coordinate, shared.primal[index]);
        }
    }
    __syncthreads();
}

}  // namespace solvedelta::paired_action
