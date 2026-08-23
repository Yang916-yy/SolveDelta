
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <limits>
#include <tuple>
#include <vector>

namespace {

constexpr int kRank = 128;
constexpr int kChunk = 32;
constexpr int kTile = 16;
constexpr int kTiles = kRank / kTile;
constexpr int kThreads = 512;
constexpr int kFactorGroup = 16;
constexpr int kDescriptorRank = 3;

struct __align__(32) ActionShared {
    float solve[kChunk * kRank];
    float factor[kFactorGroup * kTile * kTile];
};

template <bool Packed>
__device__ __forceinline__ float direct_value(
    const float* __restrict__ first,
    const float* __restrict__ second,
    int vector_base,
    int dual_base,
    int target,
    int coordinate,
    int item) {
    if constexpr (Packed) {
        return first[
            dual_base + target * 2 * kRank + item * kRank + coordinate];
    } else {
        const int index = vector_base + target * kRank + coordinate;
        return item == 0 ? first[index] : second[index];
    }
}

template <bool Upper, bool Diagonal, bool Packed>
__device__ __forceinline__ void apply_factor_group(
    int group,
    int row_start,
    int col_start,
    const float* __restrict__ direct_first,
    const float* __restrict__ direct_second,
    float& direct_accumulator0,
    float& direct_accumulator1,
    ActionShared& shared,
    int vector_base,
    int dual_base) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int solve_target = group * kFactorGroup + warp;
    const int factor_base = warp * kTile * kTile;

    if constexpr (Diagonal) {
        float residual = lane < kTile
            ? shared.solve[
                solve_target * kRank + row_start + lane]
            : 0.0f;
#pragma unroll
        for (int step = 0; step < kTile; ++step) {
            const int pivot = Upper ? step : kTile - 1 - step;
            const float solved = __shfl_sync(
                0xffffffffu, residual, pivot);
            const bool active = Upper
                ? lane > pivot && lane < kTile
                : lane < pivot;
            if (active) {
                const int factor_entry = Upper
                    ? pivot * kTile + lane
                    : pivot * kTile + lane;
                residual = fmaf(
                    -shared.factor[factor_base + factor_entry],
                    solved, residual);
            }
        }
        if (lane < kTile) {
            shared.solve[solve_target * kRank + row_start + lane] = residual;
        }
    }

    const int group_begin = group * (kThreads / 2);
    const bool owns_group = tid >= group_begin
        && tid < group_begin + kThreads / 2;
    if (!owns_group) return;
    const int local = tid - group_begin;
    const int target = group * kFactorGroup + local / kTile;
    const int coordinate = local % kTile;
    const int local_factor_base = (target - group * kFactorGroup)
        * kTile * kTile;

    if constexpr (!Diagonal) {
        float transpose_action = 0.0f;
#pragma unroll
        for (int inner = 0; inner < kTile; ++inner) {
            transpose_action = fmaf(
                shared.factor[
                    local_factor_base + inner * kTile + coordinate],
                shared.solve[target * kRank + row_start + inner],
                transpose_action);
        }
        shared.solve[target * kRank + col_start + coordinate]
            -= transpose_action;
    }

    float action0 = 0.0f;
    float action1 = 0.0f;
#pragma unroll
    for (int inner = 0; inner < kTile; ++inner) {
        const bool strict = Upper ? inner > coordinate : inner < coordinate;
        if constexpr (!Diagonal) {
            const float factor = shared.factor[
                local_factor_base + coordinate * kTile + inner];
            action0 = fmaf(
                factor,
                direct_value<Packed>(
                    direct_first, direct_second, vector_base, dual_base,
                    target, col_start + inner, 0),
                action0);
            action1 = fmaf(
                factor,
                direct_value<Packed>(
                    direct_first, direct_second, vector_base, dual_base,
                    target, col_start + inner, 1),
                action1);
        } else if (strict) {
            const float factor = shared.factor[
                local_factor_base + coordinate * kTile + inner];
            action0 = fmaf(
                factor,
                direct_value<Packed>(
                    direct_first, direct_second, vector_base, dual_base,
                    target, row_start + inner, 0),
                action0);
            action1 = fmaf(
                factor,
                direct_value<Packed>(
                    direct_first, direct_second, vector_base, dual_base,
                    target, row_start + inner, 1),
                action1);
        }
    }
    direct_accumulator0 += action0;
    direct_accumulator1 += action1;
}

template <bool Upper, bool Diagonal, bool Packed>
__device__ __forceinline__ void process_factor_block(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ alpha0,
    const float* __restrict__ diagonal_weight,
    const float* __restrict__ coefficient,
    const float* __restrict__ direct_first,
    const float* __restrict__ direct_second,
    float& direct_accumulator0,
    float& direct_accumulator1,
    ActionShared& shared,
    int panel,
    int row_start,
    int col_start) {
    const int tid = threadIdx.x;
    const int matrix_base = panel * kRank * kRank;
    const int vector_base = panel * kChunk * kRank;
    const int dual_base = panel * kChunk * 2 * kRank;
    const int scalar_base = panel * kChunk;
    constexpr int component = Upper ? 2 : 0;

    float moment_j = 0.0f;
    float moment_d = 0.0f;
    int row = 0;
    int col = 0;
    if (tid < kTile * kTile) {
        row = tid / kTile;
        col = tid % kTile;
        const int global = matrix_base
            + (row_start + row) * kRank + col_start + col;
        moment_j = fmaf(
            alpha0[panel], boundary_j[global],
            diagonal_weight[scalar_base]
                * u[vector_base + row_start + row]
                * u[vector_base + col_start + col]);
        moment_d = fmaf(
            alpha0[panel], boundary_d[global],
            diagonal_weight[scalar_base]
                * u[vector_base + row_start + row]
                * h[vector_base + col_start + col]);
#pragma unroll
        for (int target = 0; target < kFactorGroup; ++target) {
            if (target > 0) {
                const float weight = diagonal_weight[scalar_base + target];
                const float beta = 1.0f - weight;
                const int target_vector = vector_base + target * kRank;
                const float u_row = u[target_vector + row_start + row];
                moment_j = fmaf(
                    beta, moment_j,
                    weight * u_row * u[target_vector + col_start + col]);
                moment_d = fmaf(
                    beta, moment_d,
                    weight * u_row * h[target_vector + col_start + col]);
            }
            shared.factor[target * kTile * kTile + tid] = fmaf(
                coefficient[(scalar_base + target) * 4 + component], moment_j,
                coefficient[(scalar_base + target) * 4 + component + 1]
                    * moment_d);
        }
    }
    __syncthreads();
    apply_factor_group<Upper, Diagonal, Packed>(
        0, row_start, col_start, direct_first, direct_second,
        direct_accumulator0, direct_accumulator1, shared,
        vector_base, dual_base);
    __syncthreads();

    if (tid < kTile * kTile) {
#pragma unroll
        for (int target = kFactorGroup; target < kChunk; ++target) {
            const float weight = diagonal_weight[scalar_base + target];
            const float beta = 1.0f - weight;
            const int target_vector = vector_base + target * kRank;
            const float u_row = u[target_vector + row_start + row];
            moment_j = fmaf(
                beta, moment_j,
                weight * u_row * u[target_vector + col_start + col]);
            moment_d = fmaf(
                beta, moment_d,
                weight * u_row * h[target_vector + col_start + col]);
            shared.factor[(target - kFactorGroup) * kTile * kTile + tid] = fmaf(
                coefficient[(scalar_base + target) * 4 + component], moment_j,
                coefficient[(scalar_base + target) * 4 + component + 1]
                    * moment_d);
        }
    }
    __syncthreads();
    apply_factor_group<Upper, Diagonal, Packed>(
        1, row_start, col_start, direct_first, direct_second,
        direct_accumulator0, direct_accumulator1, shared,
        vector_base, dual_base);
    __syncthreads();
}

template <bool Upper, bool Packed>
__launch_bounds__(kThreads, 2)
__global__ void transpose_solve_direct_kernel(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ alpha0,
    const float* __restrict__ diagonal_weight,
    const float* __restrict__ coefficient,
    const float* __restrict__ solve_rhs,
    const float* __restrict__ direct_first,
    const float* __restrict__ direct_second,
    float* __restrict__ solve_output,
    float* __restrict__ direct_output,
    int panels) {
    __shared__ ActionShared shared;
    const int panel = blockIdx.x;
    const int tid = threadIdx.x;
    if (panel >= panels) return;
    const int vector_base = panel * kChunk * kRank;
    const int dual_base = panel * kChunk * 2 * kRank;
    for (int index = tid; index < kChunk * kRank; index += kThreads) {
        shared.solve[index] = solve_rhs[vector_base + index];
    }
    __syncthreads();

#pragma unroll 1
    for (int block_step = 0; block_step < kTiles; ++block_step) {
        const int block = Upper ? block_step : kTiles - 1 - block_step;
        const int row_start = block * kTile;
        const int target = tid / kTile;
        const int coordinate = tid % kTile;
        float direct0 = direct_value<Packed>(
            direct_first, direct_second, vector_base, dual_base,
            target, row_start + coordinate, 0);
        float direct1 = direct_value<Packed>(
            direct_first, direct_second, vector_base, dual_base,
            target, row_start + coordinate, 1);

        process_factor_block<Upper, true, Packed>(
            boundary_j, boundary_d, u, h, alpha0, diagonal_weight,
            coefficient, direct_first, direct_second, direct0, direct1,
            shared, panel, row_start, row_start);

        const int remaining = Upper ? kTiles - block - 1 : block;
#pragma unroll 1
        for (int relative = 0; relative < remaining; ++relative) {
            const int other = Upper ? block + 1 + relative : relative;
            process_factor_block<Upper, false, Packed>(
                boundary_j, boundary_d, u, h, alpha0, diagonal_weight,
                coefficient, direct_first, direct_second, direct0, direct1,
                shared, panel, row_start, other * kTile);
        }
        direct_output[
            dual_base + target * 2 * kRank + row_start + coordinate] = direct0;
        direct_output[
            dual_base + target * 2 * kRank + kRank
            + row_start + coordinate] = direct1;
        __syncthreads();
    }

    for (int index = tid; index < kChunk * kRank; index += kThreads) {
        solve_output[vector_base + index] = shared.solve[index];
    }
}

__global__ void scale_and_describe_kernel(
    const float* __restrict__ diagonal,
    const float* __restrict__ write_direction,
    const float* __restrict__ lower_solved,
    const float* __restrict__ dual_scaled,
    const float* __restrict__ grad_e,
    const float* __restrict__ grad_chi,
    const float* __restrict__ c,
    const float* __restrict__ gw,
    float* __restrict__ gy,
    float* __restrict__ gv,
    float* __restrict__ grad_sigma,
    float* __restrict__ upper_left,
    float* __restrict__ upper_right,
    float* __restrict__ lower_right,
    int vectors) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vectors) return;
    const int scalar = index / kRank;
    const int coordinate = index % kRank;
    const int dual_base = scalar * 2 * kRank;
    const int descriptor_base = scalar * kDescriptorRank * kRank;
    const float sigma = diagonal[index];
    const float inverse = 1.0f / sigma;
    const float c_value = c[index];
    const float gw0 = gw[dual_base + coordinate];
    const float gw1 = gw[dual_base + kRank + coordinate];
    const float w0 = dual_scaled[dual_base + coordinate];
    const float w1 = dual_scaled[dual_base + kRank + coordinate];
    const float gv0 = sigma * gw0;
    const float gv1 = sigma * gw1;
    gy[index] = c_value * inverse;
    gv[dual_base + coordinate] = gv0;
    gv[dual_base + kRank + coordinate] = gv1;
    grad_sigma[index] = -c_value * lower_solved[index] * inverse * inverse
        + gw0 * w0 * inverse + gw1 * w1 * inverse;

    upper_left[descriptor_base + coordinate] = -c_value;
    upper_left[descriptor_base + kRank + coordinate] = w0;
    upper_left[descriptor_base + 2 * kRank + coordinate] = w1;
    upper_right[descriptor_base + coordinate] = write_direction[index];
    upper_right[descriptor_base + kRank + coordinate] = grad_e[index];
    upper_right[descriptor_base + 2 * kRank + coordinate] = grad_chi[index];
    lower_right[descriptor_base + coordinate] = lower_solved[index];
    lower_right[descriptor_base + kRank + coordinate] = gv0;
    lower_right[descriptor_base + 2 * kRank + coordinate] = gv1;
}

__global__ void finalize_kernel(
    const float* __restrict__ key,
    const float* __restrict__ erase,
    const float* __restrict__ query,
    const float* __restrict__ gk,
    const float* __restrict__ gx,
    float* __restrict__ lower_left,
    float* __restrict__ grad_key,
    float* __restrict__ grad_erase,
    float* __restrict__ grad_query,
    int vectors) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vectors) return;
    const int scalar = index / kRank;
    const int coordinate = index % kRank;
    const int dual_base = scalar * 2 * kRank;
    const int descriptor_base = scalar * kDescriptorRank * kRank;
    const float gx0 = gx[dual_base + coordinate];
    const float gx1 = gx[dual_base + kRank + coordinate];
    lower_left[descriptor_base + coordinate] = -gk[index];
    lower_left[descriptor_base + kRank + coordinate]
        = erase[index] * key[index];
    lower_left[descriptor_base + 2 * kRank + coordinate] = query[index];
    grad_key[index] = gk[index] + erase[index] * gx0;
    grad_erase[index] = key[index] * gx0;
    grad_query[index] = gx1;
}

void check_vector(const at::Tensor& value, const at::Tensor& reference,
                  const char* name) {
    TORCH_CHECK(value.is_cuda() && value.scalar_type() == at::kFloat
                && value.is_contiguous() && value.device() == reference.device(),
                name, " must be contiguous CUDA FP32 on the same device");
}

}  // namespace

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor>
panel_frame32_action_vjp_cuda(
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& alpha0,
    const at::Tensor& inv_mass,
    const at::Tensor& coefficient,
    const at::Tensor& sigma,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& d,
    const at::Tensor& saved_y,
    const at::Tensor& saved_w,
    const at::Tensor& grad_d,
    const at::Tensor& grad_e,
    const at::Tensor& grad_chi) {
    TORCH_CHECK(boundary_j.is_cuda() && boundary_j.scalar_type() == at::kFloat
                && boundary_j.is_contiguous() && boundary_j.dim() == 3
                && boundary_j.size(1) == kRank && boundary_j.size(2) == kRank,
                "boundary_j must be contiguous CUDA FP32 [P,128,128]");
    const int64_t panels = boundary_j.size(0);
    TORCH_CHECK(panels > 0, "panel_frame32 action VJP requires at least one panel");
    TORCH_CHECK(
        panels <= std::numeric_limits<int>::max(), "too many panels");
    const std::vector<int64_t> vector_shape{panels, kChunk, kRank};
    const std::vector<int64_t> dual_shape{panels, kChunk, 2, kRank};
    const std::vector<int64_t> descriptor_shape{
        panels, kChunk, kDescriptorRank, kRank};
    check_vector(boundary_d, boundary_j, "boundary_d");
    for (const auto* item : {&u, &h, &sigma, &key, &erase, &query,
                             &d, &saved_y, &grad_d,
                             &grad_e, &grad_chi}) {
        check_vector(*item, boundary_j, "vector input");
        TORCH_CHECK(item->sizes() == vector_shape, "vector input shape");
    }
    check_vector(alpha0, boundary_j, "alpha0");
    check_vector(inv_mass, boundary_j, "inv_mass");
    check_vector(coefficient, boundary_j, "coefficient");
    check_vector(saved_w, boundary_j, "saved_w");
    TORCH_CHECK(boundary_d.sizes() == boundary_j.sizes(), "boundary_d shape");
    TORCH_CHECK(alpha0.sizes() == at::IntArrayRef({panels}), "alpha0 shape");
    TORCH_CHECK(inv_mass.sizes()
                == at::IntArrayRef({panels, kChunk}), "inv_mass shape");
    TORCH_CHECK(coefficient.sizes()
                == at::IntArrayRef({panels, kChunk, 4}), "coefficient shape");
    TORCH_CHECK(saved_w.sizes() == dual_shape, "saved_w shape");

    c10::cuda::CUDAGuard guard(boundary_j.device());
    auto c = at::empty_like(key);
    auto gw = at::empty(dual_shape, key.options());
    auto gy = at::empty_like(key);
    auto gv = at::empty(dual_shape, key.options());
    auto gk = at::empty_like(key);
    auto gx = at::empty(dual_shape, key.options());
    auto upper_left = at::empty(descriptor_shape, key.options());
    auto upper_right = at::empty(descriptor_shape, key.options());
    auto lower_left = at::empty(descriptor_shape, key.options());
    auto lower_right = at::empty(descriptor_shape, key.options());
    auto grad_sigma = at::empty_like(key);
    auto grad_key = at::empty_like(key);
    auto grad_erase = at::empty_like(key);
    auto grad_query = at::empty_like(key);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int panel_count = static_cast<int>(panels);

    transpose_solve_direct_kernel<true, false><<<
        panel_count, kThreads, 0, stream>>>(
        boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), alpha0.data_ptr<float>(),
        inv_mass.data_ptr<float>(), coefficient.data_ptr<float>(),
        grad_d.data_ptr<float>(), grad_e.data_ptr<float>(),
        grad_chi.data_ptr<float>(), c.data_ptr<float>(), gw.data_ptr<float>(),
        panel_count);

    const int vectors = panel_count * kChunk * kRank;
    constexpr int element_threads = 256;
    const int element_blocks = (vectors + element_threads - 1) / element_threads;
    scale_and_describe_kernel<<<element_blocks, element_threads, 0, stream>>>(
        sigma.data_ptr<float>(), d.data_ptr<float>(),
        saved_y.data_ptr<float>(), saved_w.data_ptr<float>(),
        grad_e.data_ptr<float>(), grad_chi.data_ptr<float>(),
        c.data_ptr<float>(), gw.data_ptr<float>(), gy.data_ptr<float>(),
        gv.data_ptr<float>(), grad_sigma.data_ptr<float>(),
        upper_left.data_ptr<float>(), upper_right.data_ptr<float>(),
        lower_right.data_ptr<float>(), vectors);

    transpose_solve_direct_kernel<false, true><<<
        panel_count, kThreads, 0, stream>>>(
        boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), alpha0.data_ptr<float>(),
        inv_mass.data_ptr<float>(), coefficient.data_ptr<float>(),
        gy.data_ptr<float>(), gv.data_ptr<float>(), nullptr,
        gk.data_ptr<float>(), gx.data_ptr<float>(), panel_count);

    finalize_kernel<<<element_blocks, element_threads, 0, stream>>>(
        key.data_ptr<float>(), erase.data_ptr<float>(), query.data_ptr<float>(),
        gk.data_ptr<float>(), gx.data_ptr<float>(), lower_left.data_ptr<float>(),
        grad_key.data_ptr<float>(), grad_erase.data_ptr<float>(),
        grad_query.data_ptr<float>(), vectors);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {upper_left, upper_right, lower_left, lower_right,
            grad_sigma, grad_key, grad_erase, grad_query};
}

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    // Forward auxiliaries are y=L^{-1}key and
    // saved_w=sigma*L^T[erase*key, query], laid out [P,32,2,128].
    // The first four outputs are [P,32,3,128] outer-product descriptors:
    // grad_U = [-c,w_e,w_q] [d,grad_e,grad_chi]^T and
    // grad_L = [-gk,erase*key,query] [y,sigma*U*grad_e,sigma*U*grad_chi]^T.
    // The radial VJP owns the strict-triangular mask.
    m.def(
        "panel_frame32_action_vjp128(Tensor boundary_j, Tensor boundary_d, "
        "Tensor u, Tensor h, Tensor alpha0, Tensor inv_mass, "
        "Tensor coefficient, Tensor sigma, Tensor key, Tensor erase, "
        "Tensor query, Tensor d, Tensor saved_y, Tensor saved_w, "
        "Tensor grad_d, Tensor grad_e, Tensor grad_chi) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl(
        "panel_frame32_action_vjp128",
        &panel_frame32_action_vjp_cuda);
}
