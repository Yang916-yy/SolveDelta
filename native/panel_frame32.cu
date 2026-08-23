#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <limits>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

constexpr int kRank = 128;
constexpr int kChunk = 32;
constexpr int kTile = 16;
constexpr int kTiles = kRank / kTile;
constexpr int kThreads = kChunk * kTile;
constexpr int kDualRhs = 2;

struct FactorShared {
    float y[kChunk * kRank];
    float u_col[kChunk * kTile];
    float h_col[kChunk * kTile];
    float u_row[kChunk * kTile];
    float boundary_j[kTile * kTile];
    float boundary_d[kTile * kTile];
    float inv_mass[kChunk];
    float coefficient[kChunk * 4];
    float factor[kChunk * kTile * kTile];
};

struct FrameShared {
    FactorShared primal;
    float dual[kDualRhs * kChunk * kRank];
    float dual_accumulator[kDualRhs * kChunk * kTile];
};

template <bool Upper>
__device__ __forceinline__ void reconstruct_factor(
    FactorShared& shared,
    const float* __restrict__ alpha0,
    int panel) {
    const int tid = threadIdx.x;
    constexpr int component = Upper ? 2 : 0;
    if (tid < kTile * kTile) {
        const int row = tid / kTile;
        const int col = tid % kTile;
        const float weight0 = shared.inv_mass[0];
        float moment_j = fmaf(
            alpha0[panel], shared.boundary_j[tid],
            weight0 * shared.u_row[row] * shared.u_col[col]);
        float moment_d = fmaf(
            alpha0[panel], shared.boundary_d[tid],
            weight0 * shared.u_row[row] * shared.h_col[col]);
        shared.factor[tid] = fmaf(
            shared.coefficient[component], moment_j,
            shared.coefficient[component + 1] * moment_d);
#pragma unroll 1
        for (int target = 1; target < kChunk; ++target) {
            const float weight = shared.inv_mass[target];
            const float retain = 1.0f - weight;
            moment_j = fmaf(
                retain, moment_j,
                weight
                    * shared.u_row[target * kTile + row]
                    * shared.u_col[target * kTile + col]);
            moment_d = fmaf(
                retain, moment_d,
                weight
                    * shared.u_row[target * kTile + row]
                    * shared.h_col[target * kTile + col]);
            shared.factor[target * kTile * kTile + tid] = fmaf(
                shared.coefficient[target * 4 + component], moment_j,
                shared.coefficient[target * 4 + component + 1] * moment_d);
        }
    }
    __syncthreads();
}

template <bool Upper, bool DiagonalBlock>
__device__ __forceinline__ void accumulate_dual(
    FrameShared& shared,
    int row_start) {
    const int tid = threadIdx.x;
    const int target = tid / kTile;
    const int col = tid % kTile;
#pragma unroll
    for (int rhs = 0; rhs < kDualRhs; ++rhs) {
        float action = 0.0f;
#pragma unroll
        for (int row = 0; row < kTile; ++row) {
            bool active = true;
            if constexpr (DiagonalBlock) {
                active = Upper ? row < col : row > col;
            }
            if (active) {
                action = fmaf(
                    shared.primal.factor[
                        target * kTile * kTile + row * kTile + col],
                    shared.dual[
                        (rhs * kChunk + target) * kRank + row_start + row],
                    action);
            }
        }
        shared.dual_accumulator[
            (rhs * kChunk + target) * kTile + col] += action;
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
    int tile) {
    auto& primal = shared.primal;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int matrix_base = panel * kRank * kRank;
    const int vector_base = panel * kChunk * kRank;
    const int col_start = tile * kTile;
    const int source = tid / kTile;
    const int local_coordinate = tid % kTile;
    const int col_coordinate = col_start + local_coordinate;

    primal.u_col[tid] = u[
        vector_base + source * kRank + col_coordinate];
    primal.h_col[tid] = h[
        vector_base + source * kRank + col_coordinate];
    primal.u_row[tid] = primal.u_col[tid];
    if (tid < kTile * kTile) {
        const int row = col_start + tid / kTile;
        const int col = col_start + tid % kTile;
        primal.boundary_j[tid] = boundary_j[
            matrix_base + row * kRank + col];
        primal.boundary_d[tid] = boundary_d[
            matrix_base + row * kRank + col];
    }
    for (int index = tid; index < kDualRhs * kChunk * kTile;
         index += kThreads) {
        const int rhs = index / (kChunk * kTile);
        const int local = index % (kChunk * kTile);
        const int target = local / kTile;
        const int coordinate = local % kTile;
        shared.dual_accumulator[index] = shared.dual[
            (rhs * kChunk + target) * kRank + col_start + coordinate];
    }
    __syncthreads();
    reconstruct_factor<Upper>(primal, alpha0, panel);

#pragma unroll
    for (int target_group = 0; target_group < 2; ++target_group) {
        const int target = target_group * 16 + warp;
        float residual = lane < kTile
            ? primal.y[target * kRank + col_start + lane]
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
                    -primal.factor[
                        target * kTile * kTile + lane * kTile + pivot],
                    solved,
                    residual);
            }
        }
        if (lane < kTile) {
            primal.y[target * kRank + col_start + lane] = residual;
        }
    }
    accumulate_dual<Upper, true>(shared, col_start);
    __syncthreads();

    const int row_begin = Upper ? 0 : tile + 1;
    const int row_end = Upper ? tile : kTiles;
    for (int row_tile = row_begin; row_tile < row_end; ++row_tile) {
        const int row_start = row_tile * kTile;
        primal.u_row[tid] = u[
            vector_base + source * kRank + row_start + local_coordinate];
        if (tid < kTile * kTile) {
            const int row = row_start + tid / kTile;
            const int col = col_start + tid % kTile;
            primal.boundary_j[tid] = boundary_j[
                matrix_base + row * kRank + col];
            primal.boundary_d[tid] = boundary_d[
                matrix_base + row * kRank + col];
        }
        __syncthreads();
        reconstruct_factor<Upper>(primal, alpha0, panel);

        const int target = tid / kTile;
        const int row = tid % kTile;
        float action = 0.0f;
#pragma unroll
        for (int col = 0; col < kTile; ++col) {
            action = fmaf(
                primal.factor[
                    target * kTile * kTile + row * kTile + col],
                primal.y[target * kRank + col_start + col],
                action);
        }
        primal.y[target * kRank + row_start + row] -= action;
        accumulate_dual<Upper, false>(shared, row_start);
        __syncthreads();
    }

    for (int index = tid; index < kDualRhs * kChunk * kTile;
         index += kThreads) {
        const int rhs = index / (kChunk * kTile);
        const int local = index % (kChunk * kTile);
        const int target = local / kTile;
        const int coordinate = local % kTile;
        shared.dual[
            (rhs * kChunk + target) * kRank + col_start + coordinate]
            = shared.dual_accumulator[index];
    }
    __syncthreads();
}

__launch_bounds__(kThreads, 1)
__global__ void frame_kernel(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ alpha0,
    const float* __restrict__ inv_mass,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ key,
    const float* __restrict__ erase,
    const float* __restrict__ query,
    float* __restrict__ output_d,
    float* __restrict__ output_e,
    float* __restrict__ output_chi,
    float* __restrict__ lower_solved,
    float* __restrict__ dual_scaled,
    int panels) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<FrameShared*>(storage);
    auto& primal = shared.primal;
    const int panel = blockIdx.x;
    const int tid = threadIdx.x;
    if (panel >= panels) return;
    const int vector_base = panel * kChunk * kRank;
    const int scalar_base = panel * kChunk;

    for (int index = tid; index < kChunk * kRank; index += kThreads) {
        const float key_value = key[vector_base + index];
        primal.y[index] = key_value;
        shared.dual[index] = erase[vector_base + index] * key_value;
        shared.dual[kChunk * kRank + index] = query[vector_base + index];
    }
    if (tid < kChunk) {
        primal.inv_mass[tid] = inv_mass[scalar_base + tid];
    }
    if (tid < kChunk * 4) {
        primal.coefficient[tid] = coefficient[scalar_base * 4 + tid];
    }
    __syncthreads();

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        frame_step<false>(
            shared, boundary_j, boundary_d, u, h, alpha0, panel, tile);
    }
    for (int index = tid; index < kChunk * kRank; index += kThreads) {
        const float scale = diagonal[vector_base + index];
        const int target = index / kRank;
        const int coordinate = index % kRank;
        const int dual_base =
            ((panel * kChunk + target) * kDualRhs) * kRank + coordinate;
        lower_solved[vector_base + index] = primal.y[index];
        primal.y[index] /= scale;
        shared.dual[index] *= scale;
        shared.dual[kChunk * kRank + index] *= scale;
        dual_scaled[dual_base] = shared.dual[index];
        dual_scaled[dual_base + kRank] =
            shared.dual[kChunk * kRank + index];
    }
    __syncthreads();
#pragma unroll 1
    for (int tile = kTiles - 1; tile >= 0; --tile) {
        frame_step<true>(
            shared, boundary_j, boundary_d, u, h, alpha0, panel, tile);
    }

    for (int index = tid; index < kChunk * kRank; index += kThreads) {
        output_d[vector_base + index] = primal.y[index];
        output_e[vector_base + index] = shared.dual[index];
        output_chi[vector_base + index] =
            shared.dual[kChunk * kRank + index];
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

void check_inputs(
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& alpha0,
    const at::Tensor& inv_mass,
    const at::Tensor& coefficient,
    const at::Tensor& diagonal,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query) {
    TORCH_CHECK(
        boundary_j.is_cuda() && boundary_j.scalar_type() == at::kFloat
            && boundary_j.is_contiguous() && boundary_j.dim() == 3
            && boundary_j.size(1) == kRank && boundary_j.size(2) == kRank,
        "boundary_j must be contiguous CUDA FP32 [P,128,128]");
    const int64_t panels = boundary_j.size(0);
    TORCH_CHECK(
        panels > 0 && panels <= std::numeric_limits<int>::max(),
        "panel count must fit int32");
    for (const auto* item : {
             &boundary_d, &u, &h, &alpha0, &inv_mass, &coefficient,
             &diagonal, &key, &erase, &query}) {
        check_tensor(*item, boundary_j, "panel frame input");
    }
    TORCH_CHECK(boundary_d.sizes() == boundary_j.sizes(), "boundary_d shape");
    const std::vector<int64_t> vector_shape{panels, kChunk, kRank};
    TORCH_CHECK(u.sizes() == vector_shape, "u must be [P,32,128]");
    for (const auto* item : {&h, &diagonal, &key, &erase, &query}) {
        TORCH_CHECK(item->sizes() == vector_shape, "vector input shape");
    }
    TORCH_CHECK(alpha0.sizes() == at::IntArrayRef({panels}), "alpha0 shape");
    TORCH_CHECK(
        inv_mass.sizes() == at::IntArrayRef({panels, kChunk}),
        "inv_mass must be [P,32]");
    TORCH_CHECK(
        coefficient.sizes() == at::IntArrayRef({panels, kChunk, 4}),
        "coefficient must be [P,32,4]");
}

std::tuple<
    at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor>
panel_frame32_action_cuda(
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& alpha0,
    const at::Tensor& inv_mass,
    const at::Tensor& coefficient,
    const at::Tensor& diagonal,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query) {
    check_inputs(
        boundary_j, boundary_d, u, h, alpha0, inv_mass, coefficient,
        diagonal, key, erase, query);
    c10::cuda::CUDAGuard guard(boundary_j.device());
    auto output_d = at::empty_like(key);
    auto output_e = at::empty_like(key);
    auto output_chi = at::empty_like(key);
    auto lower_solved = at::empty_like(key);
    auto dual_scaled = at::empty(
        {key.size(0), kChunk, kDualRhs, kRank}, key.options());
    const auto stream = at::cuda::getCurrentCUDAStream();
    static std::once_flag configured;
    std::call_once(configured, [] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            frame_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(sizeof(FrameShared))));
    });
    frame_kernel<<<
        boundary_j.size(0), kThreads, sizeof(FrameShared), stream>>>(
        boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), alpha0.data_ptr<float>(),
        inv_mass.data_ptr<float>(), coefficient.data_ptr<float>(),
        diagonal.data_ptr<float>(), key.data_ptr<float>(),
        erase.data_ptr<float>(), query.data_ptr<float>(),
        output_d.data_ptr<float>(), output_e.data_ptr<float>(),
        output_chi.data_ptr<float>(), lower_solved.data_ptr<float>(),
        dual_scaled.data_ptr<float>(), static_cast<int>(boundary_j.size(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_d, output_e, output_chi, lower_solved, dual_scaled};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "panel_frame32_action128("
        "Tensor boundary_j, Tensor boundary_d, Tensor u, Tensor h, "
        "Tensor alpha0, Tensor inv_mass, Tensor coefficient, Tensor diagonal, "
        "Tensor key, Tensor erase, Tensor query) -> "
        "(Tensor d, Tensor e, Tensor chi, Tensor lower_solved, Tensor dual_scaled)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("panel_frame32_action128", &panel_frame32_action_cuda);
}
