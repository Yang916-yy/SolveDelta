/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted and modified for SolveDelta from NVIDIA's cuBLASDx trsm_block
 * example. See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.
 */

// Upstream documentation: https://docs.nvidia.com/cuda/cublasdx/using_trsm.html

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <tuple>

#include <cublasdx.hpp>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr unsigned kRank = 128;
constexpr unsigned kRhs = 2;
constexpr unsigned kEdits = 1;
constexpr unsigned kDualRhs = kEdits + 1;
constexpr unsigned kArch = 1200;
constexpr unsigned kThreads = 512;
constexpr unsigned kChunkThreads = 512;

template<cublasdx::fill_mode Fill>
using Trsm = decltype(
    cublasdx::Size<kRank, kRhs>()
    + cublasdx::Precision<float>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::TRSM>()
    + cublasdx::SM<kArch>()
    + cublasdx::Block()
    + cublasdx::BlockDim<kThreads>()
    + cublasdx::Side<cublasdx::side::left>()
    + cublasdx::FillMode<Fill>()
    + cublasdx::Diag<cublasdx::diag::unit>()
    + cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major>()
    + cublasdx::BatchesPerBlock<1>());

template<class BLAS, class GlobalA, class GlobalB>
__launch_bounds__(BLAS::max_threads_per_block)
__global__ void trsm_kernel(GlobalA global_a, GlobalB global_b, unsigned batches) {
    extern __shared__ __align__(16) cublasdx::byte shared[];
    if (blockIdx.x >= batches) {
        return;
    }
    using alignment = cublasdx::alignment_of<BLAS>;
    auto batch_a = cublasdx::get_batch(
        global_a, BLAS::get_layout_gmem_a(), blockIdx.x);
    auto batch_b = cublasdx::get_batch(
        global_b, BLAS::get_layout_gmem_b(), blockIdx.x);
    auto [smem_a, smem_b] = cublasdx::shared_memory::slice<float, float>(
        shared,
        alignment::a,
        BLAS::get_layout_smem_a(),
        alignment::b,
        BLAS::get_layout_smem_b());
    cublasdx::copy<BLAS, alignment::a>(batch_a, smem_a);
    cublasdx::copy<BLAS, alignment::b>(batch_b, smem_b);
    cublasdx::copy_wait();
    BLAS{}.execute(smem_a, smem_b);
    __syncthreads();
    cublasdx::copy<BLAS, alignment::b>(smem_b, batch_b);
}

template<class BLAS>
void launch_trsm(const at::Tensor& factor_col, at::Tensor& rhs_col) {
    const auto batches = static_cast<unsigned>(factor_col.size(0));
    auto global_a = cublasdx::make_gmem_tensor_batched<cublasdx::col_major>(
        factor_col.data_ptr<float>(), kRank, kRank, batches);
    auto global_b = cublasdx::make_gmem_tensor_batched<cublasdx::col_major>(
        rhs_col.data_ptr<float>(), kRank, kRhs, batches);
    using GlobalA = decltype(global_a);
    using GlobalB = decltype(global_b);
    const unsigned shared_bytes = cublasdx::make_shared_storage_calculator()
        .add(cublasdx::alignment_of<BLAS>::a, sizeof(float), BLAS::get_layout_smem_a())
        .add(cublasdx::alignment_of<BLAS>::b, sizeof(float), BLAS::get_layout_smem_b())
        .get();
    auto kernel = trsm_kernel<BLAS, GlobalA, GlobalB>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    const auto stream = at::cuda::getCurrentCUDAStream();
    kernel<<<batches, kThreads, shared_bytes, stream>>>(global_a, global_b, batches);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<class LowerBLAS, class UpperBLAS>
__launch_bounds__(kThreads)
__global__ void solve_frame_kernel(
    const float* lower,
    const float* diagonal,
    const float* upper,
    const float* keys,
    const float* dual_rhs,
    float* write_direction,
    float* dual_output,
    unsigned batches) {
    extern __shared__ __align__(16) cublasdx::byte shared[];
    __shared__ float dual_mid[3 * kRank];
    const unsigned batch = blockIdx.x;
    if (batch >= batches) return;

    using lower_alignment = cublasdx::alignment_of<LowerBLAS>;
    auto [smem_a, smem_b] = cublasdx::shared_memory::slice<float, float>(
        shared,
        lower_alignment::a,
        LowerBLAS::get_layout_smem_a(),
        lower_alignment::b,
        LowerBLAS::get_layout_smem_b());
    const float* lower_batch = lower + batch * kRank * kRank;
    const float* upper_batch = upper + batch * kRank * kRank;
    const float* key_batch = keys + batch * kRhs * kRank;
    const float* dual_batch = dual_rhs + batch * 3 * kRank;

    for (unsigned index = threadIdx.x; index < kRank * kRank; index += blockDim.x) {
        const unsigned row = index / kRank;
        const unsigned col = index % kRank;
        smem_a(row, col) = lower_batch[index];
    }
    for (unsigned index = threadIdx.x; index < kRank * kRhs; index += blockDim.x) {
        const unsigned row = index % kRank;
        const unsigned col = index / kRank;
        smem_b(row, col) = key_batch[index];
    }
    // L^T times all erase/read covectors while L is hot in global cache.
    for (unsigned index = threadIdx.x; index < 3 * kRank; index += blockDim.x) {
        const unsigned rhs = index / kRank;
        const unsigned row = index % kRank;
        float value = 0.0f;
        for (unsigned source = row; source < kRank; ++source) {
            value += lower_batch[source * kRank + row] * dual_batch[rhs * kRank + source];
        }
        dual_mid[index] = value * diagonal[batch * kRank + row];
    }
    __syncthreads();
    LowerBLAS{}.execute(smem_a, smem_b);
    __syncthreads();

    for (unsigned index = threadIdx.x; index < kRank * kRhs; index += blockDim.x) {
        const unsigned row = index % kRank;
        const unsigned col = index / kRank;
        smem_b(row, col) /= diagonal[batch * kRank + row];
    }
    for (unsigned index = threadIdx.x; index < kRank * kRank; index += blockDim.x) {
        const unsigned row = index / kRank;
        const unsigned col = index % kRank;
        smem_a(row, col) = upper_batch[index];
    }
    __syncthreads();
    UpperBLAS{}.execute(smem_a, smem_b);
    __syncthreads();

    for (unsigned index = threadIdx.x; index < kRank * kRhs; index += blockDim.x) {
        const unsigned row = index % kRank;
        const unsigned col = index / kRank;
        write_direction[batch * kRhs * kRank + col * kRank + row] = smem_b(row, col);
    }
    // U^T times the diagonal-scaled intermediate.
    for (unsigned index = threadIdx.x; index < 3 * kRank; index += blockDim.x) {
        const unsigned rhs = index / kRank;
        const unsigned row = index % kRank;
        float value = 0.0f;
        for (unsigned source = 0; source <= row; ++source) {
            value += upper_batch[source * kRank + row] * dual_mid[rhs * kRank + source];
        }
        dual_output[batch * 3 * kRank + index] = value;
    }
}

std::tuple<at::Tensor, at::Tensor> mathdx_solve_frame128_cuda(
    const at::Tensor& lower,
    const at::Tensor& diagonal,
    const at::Tensor& upper,
    const at::Tensor& keys,
    const at::Tensor& dual_rhs) {
    TORCH_CHECK(lower.is_cuda(), "frame inputs must be CUDA");
    for (const auto& tensor : {diagonal, upper, keys, dual_rhs}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == lower.get_device(), "frame inputs must share one CUDA device");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.is_contiguous(), "frame inputs must be contiguous FP32");
    }
    TORCH_CHECK(lower.scalar_type() == at::kFloat && lower.is_contiguous(), "lower must be contiguous FP32");
    TORCH_CHECK(lower.dim() == 3 && lower.size(1) == kRank && lower.size(2) == kRank, "lower must be [batch,128,128]");
    const auto batches = lower.size(0);
    TORCH_CHECK(batches > 0 && upper.sizes() == lower.sizes(), "upper shape must match lower");
    TORCH_CHECK(diagonal.sizes() == at::IntArrayRef({batches, kRank}), "diagonal must be [batch,128]");
    TORCH_CHECK(keys.sizes() == at::IntArrayRef({batches, kRhs, kRank}), "keys must be [batch,2,128]");
    TORCH_CHECK(dual_rhs.sizes() == at::IntArrayRef({batches, 3, kRank}), "dual_rhs must be [batch,3,128]");
    c10::cuda::CUDAGuard guard(lower.device());
    auto write_direction = at::empty_like(keys);
    auto dual_output = at::empty_like(dual_rhs);
    using Lower = Trsm<cublasdx::fill_mode::lower>;
    using Upper = Trsm<cublasdx::fill_mode::upper>;
    const unsigned shared_bytes = cublasdx::make_shared_storage_calculator()
        .add(cublasdx::alignment_of<Lower>::a, sizeof(float), Lower::get_layout_smem_a())
        .add(cublasdx::alignment_of<Lower>::b, sizeof(float), Lower::get_layout_smem_b())
        .get();
    auto kernel = solve_frame_kernel<Lower, Upper>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    kernel<<<batches, kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        lower.data_ptr<float>(), diagonal.data_ptr<float>(), upper.data_ptr<float>(),
        keys.data_ptr<float>(), dual_rhs.data_ptr<float>(),
        write_direction.data_ptr<float>(), dual_output.data_ptr<float>(),
        static_cast<unsigned>(batches));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(write_direction, dual_output);
}

__device__ float block_sum(float value, float* warp_sums) {
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    for (unsigned offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    float total = threadIdx.x < 16 ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (unsigned offset = 16; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
        if (lane == 0) warp_sums[0] = total;
    }
    __syncthreads();
    return warp_sums[0];
}

__device__ __forceinline__ unsigned offdiag_index(unsigned row, unsigned col) {
    return row * (kRank - 1) + col - (col > row ? 1 : 0);
}

template<class LowerBLAS, class UpperBLAS>
__launch_bounds__(kChunkThreads)
__global__ void chunk_solve_frame_kernel(
    const float* boundary_m,
    const float* boundary_j,
    const float* boundary_d,
    const float* u,
    const float* h_value,
    const float* geometry_log_decay,
    const float* keys,
    const float* erase,
    const float* query,
    const float* skew,
    const float* strength,
    float* write_direction,
    float* erase_direction,
    float* solved_query,
    unsigned length,
    unsigned heads,
    unsigned chunks,
    unsigned chunk_size) {
    constexpr unsigned matrix_size = kRank * kRank;
    constexpr unsigned d_items_per_thread = matrix_size / kChunkThreads;
    constexpr unsigned packed_j_size = kRank * (kRank + 1) / 2;
    constexpr unsigned j_items_per_thread = (packed_j_size + kChunkThreads - 1) / kChunkThreads;
    constexpr unsigned native_subchunk = 8;
    constexpr float radius = 0.125f;
    __shared__ __half factor[kRank * (kRank - 1)];
    __shared__ __half shared_j[packed_j_size];
    __shared__ float warp_sums[16];
    __shared__ float scales[4];
    __shared__ float tau_values[kEdits];
    __shared__ float direction_norms[kEdits];

    const unsigned subchunks = (length + native_subchunk - 1) / native_subchunk;
    const unsigned token_program = blockIdx.x;
    const unsigned subchunk = token_program % subchunks;
    const unsigned block_start = subchunk * native_subchunk;
    const unsigned head_batch = token_program / subchunks;
    const unsigned head = head_batch % heads;
    const unsigned batch = head_batch / heads;
    const unsigned chunk = block_start / chunk_size;
    const unsigned start_local = block_start - chunk * chunk_size;
    const unsigned end_local = min(min(start_local + native_subchunk, chunk_size), length - chunk * chunk_size);
    const unsigned boundary_program = head_batch * chunks + chunk;
    const unsigned boundary_base = boundary_program * matrix_size;
    __nv_bfloat162 d_values[d_items_per_thread / 2];
#pragma unroll
    for (unsigned item = 0; item < j_items_per_thread; ++item) {
        const unsigned packed = threadIdx.x + item * kChunkThreads;
        if (packed < packed_j_size) {
            const unsigned row = static_cast<unsigned>((sqrtf(8.0f * packed + 1.0f) - 1.0f) * 0.5f);
            const unsigned col = packed - row * (row + 1) / 2;
            shared_j[packed] = __float2half_rn(boundary_j[boundary_base + row * kRank + col]);
        }
    }
#pragma unroll
    for (unsigned pair = 0; pair < d_items_per_thread / 2; ++pair) {
        const unsigned index0 = threadIdx.x + (2 * pair) * kChunkThreads;
        const unsigned index1 = index0 + kChunkThreads;
        d_values[pair] = __floats2bfloat162_rn(
            boundary_d[boundary_base + index0], boundary_d[boundary_base + index1]);
    }
    float mass = boundary_m[boundary_program];
    const float geometry_strength = strength[head];

    for (unsigned local_t = 0; local_t < end_local; ++local_t) {
        const unsigned local_token = chunk * chunk_size + local_t;
        const unsigned vector_base = ((batch * length + local_token) * heads + head) * kRank;
        const unsigned edit_base = vector_base;
        const float lambda = expf(geometry_log_decay[(batch * length + local_token) * heads + head]);
        mass = lambda * mass + 1.0f;
#pragma unroll
        for (unsigned item = 0; item < j_items_per_thread; ++item) {
            const unsigned packed = threadIdx.x + item * kChunkThreads;
            if (packed < packed_j_size) {
                const unsigned row = static_cast<unsigned>((sqrtf(8.0f * packed + 1.0f) - 1.0f) * 0.5f);
                const unsigned col = packed - row * (row + 1) / 2;
                shared_j[packed] = __float2half_rn(
                    lambda * __half2float(shared_j[packed])
                    + u[vector_base + row] * u[vector_base + col]);
            }
        }
#pragma unroll
        for (unsigned pair = 0; pair < d_items_per_thread / 2; ++pair) {
            const unsigned index0 = threadIdx.x + (2 * pair) * kChunkThreads;
            const unsigned index1 = index0 + kChunkThreads;
            const unsigned row0 = index0 / kRank, col0 = index0 % kRank;
            const unsigned row1 = index1 / kRank, col1 = index1 % kRank;
            const float value0 = lambda * __low2float(d_values[pair])
                + u[vector_base + row0] * h_value[vector_base + col0];
            const float value1 = lambda * __high2float(d_values[pair])
                + u[vector_base + row1] * h_value[vector_base + col1];
            d_values[pair] = __floats2bfloat162_rn(value0, value1);
        }
        if (local_t < start_local) continue;

        float sum_h = 0.0f;
        float sum_r_lower = 0.0f, sum_r_upper = 0.0f;
#pragma unroll
        for (unsigned item = 0; item < j_items_per_thread; ++item) {
            const unsigned packed = threadIdx.x + item * kChunkThreads;
            if (packed < packed_j_size) {
                const unsigned row = static_cast<unsigned>((sqrtf(8.0f * packed + 1.0f) - 1.0f) * 0.5f);
                const unsigned col = packed - row * (row + 1) / 2;
                if (row > col) {
                    const float xh = geometry_strength * __half2float(shared_j[packed]) / mass;
                    sum_h += xh * xh;
                }
            }
        }
#pragma unroll
        for (unsigned pair = 0; pair < d_items_per_thread / 2; ++pair) {
#pragma unroll
            for (unsigned lane = 0; lane < 2; ++lane) {
                const unsigned index = threadIdx.x + (2 * pair + lane) * kChunkThreads;
                const unsigned row = index / kRank;
                const unsigned col = index % kRank;
                const float d_entry = lane ? __high2float(d_values[pair]) : __low2float(d_values[pair]);
                const float xr = geometry_strength * d_entry / mass;
                if (row > col) sum_r_lower += xr * xr;
                if (row < col) sum_r_upper += xr * xr;
            }
        }
        const float norm_h = block_sum(sum_h, warp_sums);
        const float norm_r_lower = block_sum(sum_r_lower, warp_sums);
        const float norm_r_upper = block_sum(sum_r_upper, warp_sums);
        if (threadIdx.x == 0) {
            scales[0] = radius / sqrtf(radius * radius + norm_h);
            scales[1] = scales[0];
            scales[2] = radius / sqrtf(radius * radius + norm_r_lower);
            scales[3] = radius / sqrtf(radius * radius + norm_r_upper);
        }
        __syncthreads();

        // Store the complete strict off-diagonal E=N^-+N^+ once. It is used
        // for the skew residual and then overwritten by the two factors.
#pragma unroll
        for (unsigned pair = 0; pair < d_items_per_thread / 2; ++pair) {
#pragma unroll
            for (unsigned lane = 0; lane < 2; ++lane) {
                const unsigned index = threadIdx.x + (2 * pair + lane) * kChunkThreads;
                const unsigned row = index / kRank;
                const unsigned col = index % kRank;
                const float d_entry = lane ? __high2float(d_values[pair]) : __low2float(d_values[pair]);
                const float xr = geometry_strength * d_entry / mass;
                float entry = 0.0f;
                if (row > col) entry = scales[2] * xr;
                if (row < col) entry = scales[3] * xr;
                if (row != col) factor[offdiag_index(row, col)] = __float2half_rn(entry);
                if (row == col) write_direction[edit_base + row] = xr;
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned item = 0; item < j_items_per_thread; ++item) {
            const unsigned packed = threadIdx.x + item * kChunkThreads;
            if (packed < packed_j_size) {
                const unsigned row = static_cast<unsigned>((sqrtf(8.0f * packed + 1.0f) - 1.0f) * 0.5f);
                const unsigned col = packed - row * (row + 1) / 2;
                const float xh = geometry_strength * (__half2float(shared_j[packed]) / mass - (row == col ? 1.0f / kRank : 0.0f));
                if (row > col) {
                    const float entry = scales[0] * xh;
                    const unsigned lower_index = offdiag_index(row, col);
                    const unsigned upper_index = offdiag_index(col, row);
                    factor[lower_index] = __float2half_rn(
                        __half2float(factor[lower_index]) + entry);
                    factor[upper_index] = __float2half_rn(
                        __half2float(factor[upper_index]) + entry);
                } else {
                    erase_direction[edit_base + row] = xh;
                }
            }
        }
        __syncthreads();
        float solve_diagonal = 1.0f;
        if (threadIdx.x < kRank) {
            write_direction[edit_base + threadIdx.x] = expf(
                radius * tanhf(erase_direction[edit_base + threadIdx.x] / radius)
                + radius * tanhf(write_direction[edit_base + threadIdx.x] / radius));
            solve_diagonal = write_direction[edit_base + threadIdx.x];
        }
        __syncthreads();

        if (threadIdx.x < kRank) {
            const unsigned row = threadIdx.x;
            for (unsigned edit = 0; edit < kEdits; ++edit) {
                float action = 0.0f;
                for (unsigned col = 0; col < kRank; ++col) {
                    if (col != row) {
                        action += 0.5f * (
                            __half2float(factor[offdiag_index(row, col)])
                            - __half2float(factor[offdiag_index(col, row)]))
                            * keys[edit_base + edit * kRank + col];
                    }
                }
                erase_direction[edit_base + edit * kRank + row] = action;
            }
        }
        __syncthreads();
        for (unsigned edit = 0; edit < kEdits; ++edit) {
            float tau_term = 0.0f;
            float norm_term = 0.0f;
            if (threadIdx.x < kRank) {
                const float key = keys[edit_base + edit * kRank + threadIdx.x];
                tau_term = erase[edit_base + edit * kRank + threadIdx.x] * key * key;
                const float direction = erase_direction[edit_base + edit * kRank + threadIdx.x];
                norm_term = direction * direction;
            }
            const float tau = block_sum(tau_term, warp_sums);
            const float direction_norm = block_sum(norm_term, warp_sums);
            if (threadIdx.x == 0) {
                tau_values[edit] = tau;
                direction_norms[edit] = direction_norm;
            }
        }
        __syncthreads();
        if (threadIdx.x < kRank) {
            const unsigned row = threadIdx.x;
            for (unsigned edit = 0; edit < kEdits; ++edit) {
                const float key = keys[edit_base + edit * kRank + row];
                const float b0 = erase[edit_base + edit * kRank + row] * key;
                const float coefficient = tau_values[edit] * (2.0f - tau_values[edit])
                    * skew[((batch * length + local_token) * heads + head) * kEdits + edit];
                erase_direction[edit_base + edit * kRank + row] = b0 + coefficient
                    * erase_direction[edit_base + edit * kRank + row]
                    / sqrtf(1.0f + direction_norms[edit]);
            }
        }
        __syncthreads();

        // Direct lower-transpose dual, then turn E into the unit-lower factor.
        float lower_value = 0.0f;
        {
            const unsigned index = threadIdx.x;
            if (index < kDualRhs * kRank) {
            const unsigned rhs = index / kRank;
            const unsigned row = index % kRank;
            float value = rhs < kEdits
                ? erase_direction[edit_base + rhs * kRank + row]
                : query[vector_base + row];
            for (unsigned source = row + 1; source < kRank; ++source) {
                const float input = rhs < kEdits
                    ? erase_direction[edit_base + rhs * kRank + source]
                    : query[vector_base + source];
                value += __half2float(factor[offdiag_index(source, row)]) * input;
            }
            lower_value = value * write_direction[edit_base + row];
            }
        }
        __syncthreads();
        {
            const unsigned index = threadIdx.x;
            if (index < kDualRhs * kRank) {
            const unsigned rhs = index / kRank;
            const unsigned row = index % kRank;
            if (rhs < kEdits) erase_direction[edit_base + rhs * kRank + row] = lower_value;
            else solved_query[vector_base + row] = lower_value;
            }
        }
        // Four-term Neumann action for the bounded strict-lower factor.
        // ||N^-||_F < 1/4, and triangular nilpotence makes the full series exact.
        float accumulated0 = 0.0f;
        if (threadIdx.x < kRank) {
            accumulated0 = keys[edit_base + threadIdx.x];
            write_direction[edit_base + threadIdx.x] = accumulated0;
        }
        __syncthreads();
        for (unsigned order = 0; order < 4; ++order) {
            float next0 = 0.0f;
            if (threadIdx.x < kRank) {
                const unsigned row = threadIdx.x;
                for (unsigned col = 0; col < row; ++col) {
                    const float entry = __half2float(factor[offdiag_index(row, col)]);
                    next0 -= entry * write_direction[edit_base + col];
                }
            }
            __syncthreads();
            if (threadIdx.x < kRank) {
                accumulated0 += next0;
                write_direction[edit_base + threadIdx.x] = next0;
            }
            __syncthreads();
        }
        if (threadIdx.x < kRank) {
            accumulated0 /= solve_diagonal;
            write_direction[edit_base + threadIdx.x] = accumulated0;
        }
        __syncthreads();

        // Regenerate only the upper factor from register-resident moments.
#pragma unroll
        for (unsigned pair = 0; pair < d_items_per_thread / 2; ++pair) {
#pragma unroll
            for (unsigned lane = 0; lane < 2; ++lane) {
                const unsigned index = threadIdx.x + (2 * pair + lane) * kChunkThreads;
                const unsigned row = index / kRank;
                const unsigned col = index % kRank;
                float entry = 0.0f;
                if (row < col) {
                    const float d_entry = lane ? __high2float(d_values[pair]) : __low2float(d_values[pair]);
                    entry = scales[3] * geometry_strength * d_entry / mass;
                }
                if (row < col) factor[offdiag_index(row, col)] = __float2half_rn(entry);
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned item = 0; item < j_items_per_thread; ++item) {
            const unsigned packed = threadIdx.x + item * kChunkThreads;
            if (packed < packed_j_size) {
                const unsigned row = static_cast<unsigned>((sqrtf(8.0f * packed + 1.0f) - 1.0f) * 0.5f);
                const unsigned col = packed - row * (row + 1) / 2;
                if (row > col) {
                    const unsigned upper_index = offdiag_index(col, row);
                    factor[upper_index] = __float2half_rn(
                        __half2float(factor[upper_index])
                        + scales[1] * geometry_strength * __half2float(shared_j[packed]) / mass);
                }
            }
        }
        __syncthreads();
        // Four-term Neumann action for the bounded strict-upper factor.
        accumulated0 = threadIdx.x < kRank ? write_direction[edit_base + threadIdx.x] : 0.0f;
        for (unsigned order = 0; order < 4; ++order) {
            float next0 = 0.0f;
            if (threadIdx.x < kRank) {
                const unsigned row = threadIdx.x;
                for (unsigned col = row + 1; col < kRank; ++col) {
                    const float entry = __half2float(factor[offdiag_index(row, col)]);
                    next0 -= entry * write_direction[edit_base + col];
                }
            }
            __syncthreads();
            if (threadIdx.x < kRank) {
                accumulated0 += next0;
                write_direction[edit_base + threadIdx.x] = next0;
            }
            __syncthreads();
        }
        if (threadIdx.x < kRank) {
            write_direction[edit_base + threadIdx.x] = accumulated0;
        }
        __syncthreads();
        float upper_value = 0.0f;
        {
            const unsigned index = threadIdx.x;
            if (index < kDualRhs * kRank) {
            const unsigned rhs = index / kRank;
            const unsigned row = index % kRank;
            float value = rhs < kEdits
                ? erase_direction[edit_base + rhs * kRank + row]
                : solved_query[vector_base + row];
            for (unsigned source = 0; source < row; ++source) {
                const float input = rhs < kEdits
                    ? erase_direction[edit_base + rhs * kRank + source]
                    : solved_query[vector_base + source];
                value += __half2float(factor[offdiag_index(source, row)]) * input;
            }
            upper_value = value;
            }
        }
        __syncthreads();
        {
            const unsigned index = threadIdx.x;
            if (index < kDualRhs * kRank) {
            const unsigned rhs = index / kRank;
            const unsigned row = index % kRank;
            if (rhs < kEdits) erase_direction[edit_base + rhs * kRank + row] = upper_value;
            else solved_query[vector_base + row] = upper_value;
            }
        }
        __syncthreads();
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cuda_chunk_solve_frame128_cuda(
    const at::Tensor& boundary_m,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h_value,
    const at::Tensor& geometry_log_decay,
    const at::Tensor& keys,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& skew,
    const at::Tensor& strength,
    int64_t chunk_size) {
    TORCH_CHECK(u.is_cuda() && u.scalar_type() == at::kFloat && u.is_contiguous(), "u must be contiguous CUDA FP32");
    const auto batch = u.size(0), length = u.size(1), heads = u.size(2);
    TORCH_CHECK(u.dim() == 4 && u.size(3) == kRank, "u must be [B,T,H,128]");
    TORCH_CHECK(chunk_size >= 8 && chunk_size <= 64 && chunk_size % 8 == 0, "chunk_size must be 8,16,...,64");
    const auto chunks = (length + chunk_size - 1) / chunk_size;
    const auto check = [&](const at::Tensor& x) {
        TORCH_CHECK(x.is_cuda() && x.get_device() == u.get_device() && x.scalar_type() == at::kFloat && x.is_contiguous(), "all chunk-frame inputs must be contiguous CUDA FP32");
    };
    for (const auto& x : {boundary_m, boundary_j, boundary_d, h_value, geometry_log_decay, keys, erase, query, skew, strength}) check(x);
    TORCH_CHECK(boundary_m.sizes() == at::IntArrayRef({batch, heads, chunks}), "boundary_m shape mismatch");
    TORCH_CHECK(boundary_j.sizes() == at::IntArrayRef({batch, heads, chunks, kRank, kRank}), "boundary_j shape mismatch");
    TORCH_CHECK(boundary_d.sizes() == boundary_j.sizes(), "boundary_d shape mismatch");
    TORCH_CHECK(h_value.sizes() == u.sizes() && query.sizes() == u.sizes(), "h/query shape mismatch");
    TORCH_CHECK(geometry_log_decay.sizes() == at::IntArrayRef({batch, length, heads}), "geometry decay shape mismatch");
    TORCH_CHECK(keys.sizes() == at::IntArrayRef({batch, length, heads, kEdits, kRank}) && erase.sizes() == keys.sizes(), "key/erase shape mismatch");
    TORCH_CHECK(skew.sizes() == at::IntArrayRef({batch, length, heads, kEdits}), "skew shape mismatch");
    TORCH_CHECK(strength.sizes() == at::IntArrayRef({heads}), "strength shape mismatch");
    c10::cuda::CUDAGuard guard(u.device());
    auto d = at::empty_like(keys), e = at::empty_like(keys), chi = at::empty_like(query);
    using Lower = Trsm<cublasdx::fill_mode::lower>;
    using Upper = Trsm<cublasdx::fill_mode::upper>;
    auto kernel = chunk_solve_frame_kernel<Lower, Upper>;
    const auto native_subchunks = (length + 7) / 8;
    kernel<<<batch * heads * native_subchunks, kChunkThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        boundary_m.data_ptr<float>(), boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        u.data_ptr<float>(), h_value.data_ptr<float>(), geometry_log_decay.data_ptr<float>(),
        keys.data_ptr<float>(), erase.data_ptr<float>(), query.data_ptr<float>(), skew.data_ptr<float>(),
        strength.data_ptr<float>(), d.data_ptr<float>(), e.data_ptr<float>(), chi.data_ptr<float>(),
        length, heads, chunks, chunk_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(d, e, chi);
}

at::Tensor mathdx_trsm128_cuda(
    const at::Tensor& factor_col,
    const at::Tensor& rhs_col,
    bool upper) {
    TORCH_CHECK(factor_col.is_cuda() && rhs_col.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(factor_col.scalar_type() == at::kFloat, "factor must be FP32");
    TORCH_CHECK(rhs_col.scalar_type() == at::kFloat, "rhs must be FP32");
    TORCH_CHECK(factor_col.is_contiguous() && rhs_col.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(factor_col.dim() == 3 && factor_col.size(1) == kRank && factor_col.size(2) == kRank,
                "factor_col must be [batch, 128, 128]");
    TORCH_CHECK(rhs_col.dim() == 3 && rhs_col.size(1) == kRhs && rhs_col.size(2) == kRank,
                "rhs_col must be [batch, 2, 128]");
    TORCH_CHECK(factor_col.size(0) == rhs_col.size(0), "batch sizes must match");
    TORCH_CHECK(factor_col.size(0) > 0, "batch size must be positive");
    TORCH_CHECK(factor_col.get_device() == rhs_col.get_device(), "devices must match");
    c10::cuda::CUDAGuard guard(factor_col.device());
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, factor_col.get_device()));
    TORCH_CHECK(
        properties.major == 12 && properties.minor == 0,
        "this native build contains only the MathDx SM120 specialization; got SM",
        properties.major,
        properties.minor);
    auto output = rhs_col.clone();
    if (upper) {
        launch_trsm<Trsm<cublasdx::fill_mode::upper>>(factor_col, output);
    } else {
        launch_trsm<Trsm<cublasdx::fill_mode::lower>>(factor_col, output);
    }
    return output;
}

}  // namespace

TORCH_LIBRARY(causallsso, m) {
    m.def("mathdx_trsm128(Tensor factor_col, Tensor rhs_col, bool upper) -> Tensor");
    m.def("mathdx_solve_frame128(Tensor lower, Tensor diagonal, Tensor upper, Tensor keys, Tensor dual_rhs) -> (Tensor, Tensor)");
    m.def("cuda_chunk_solve_frame128(Tensor boundary_m, Tensor boundary_j, Tensor boundary_d, Tensor u, Tensor h, Tensor geometry_log_decay, Tensor keys, Tensor erase, Tensor query, Tensor skew, Tensor strength, int chunk_size) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("mathdx_trsm128", &mathdx_trsm128_cuda);
    m.impl("mathdx_solve_frame128", &mathdx_solve_frame128_cuda);
    m.impl("cuda_chunk_solve_frame128", &cuda_chunk_solve_frame128_cuda);
}
