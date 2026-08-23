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

namespace {

constexpr unsigned kRank = 128;
constexpr unsigned kRhs = 2;
constexpr unsigned kArch = 1200;
constexpr unsigned kThreads = 512;

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
    if (blockIdx.x >= batches) return;
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
    kernel<<<batches, kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        global_a, global_b, batches);
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
    for (unsigned index = threadIdx.x; index < 3 * kRank; index += blockDim.x) {
        const unsigned rhs = index / kRank;
        const unsigned row = index % kRank;
        float value = 0.0f;
        for (unsigned source = row; source < kRank; ++source) {
            value += lower_batch[source * kRank + row]
                * dual_batch[rhs * kRank + source];
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
        write_direction[batch * kRhs * kRank + col * kRank + row] =
            smem_b(row, col);
    }
    for (unsigned index = threadIdx.x; index < 3 * kRank; index += blockDim.x) {
        const unsigned rhs = index / kRank;
        const unsigned row = index % kRank;
        float value = 0.0f;
        for (unsigned source = 0; source <= row; ++source) {
            value += upper_batch[source * kRank + row]
                * dual_mid[rhs * kRank + source];
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
        TORCH_CHECK(
            tensor.is_cuda() && tensor.get_device() == lower.get_device(),
            "frame inputs must share one CUDA device");
        TORCH_CHECK(
            tensor.scalar_type() == at::kFloat && tensor.is_contiguous(),
            "frame inputs must be contiguous FP32");
    }
    TORCH_CHECK(
        lower.scalar_type() == at::kFloat && lower.is_contiguous(),
        "lower must be contiguous FP32");
    TORCH_CHECK(
        lower.dim() == 3 && lower.size(1) == kRank && lower.size(2) == kRank,
        "lower must be [batch,128,128]");
    const auto batches = lower.size(0);
    TORCH_CHECK(
        batches > 0 && upper.sizes() == lower.sizes(),
        "upper shape must match lower");
    TORCH_CHECK(
        diagonal.sizes() == at::IntArrayRef({batches, kRank}),
        "diagonal must be [batch,128]");
    TORCH_CHECK(
        keys.sizes() == at::IntArrayRef({batches, kRhs, kRank}),
        "keys must be [batch,2,128]");
    TORCH_CHECK(
        dual_rhs.sizes() == at::IntArrayRef({batches, 3, kRank}),
        "dual_rhs must be [batch,3,128]");
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
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    kernel<<<batches, kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        lower.data_ptr<float>(),
        diagonal.data_ptr<float>(),
        upper.data_ptr<float>(),
        keys.data_ptr<float>(),
        dual_rhs.data_ptr<float>(),
        write_direction.data_ptr<float>(),
        dual_output.data_ptr<float>(),
        static_cast<unsigned>(batches));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(write_direction, dual_output);
}

at::Tensor mathdx_trsm128_cuda(
    const at::Tensor& factor_col,
    const at::Tensor& rhs_col,
    bool upper) {
    TORCH_CHECK(factor_col.is_cuda() && rhs_col.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(factor_col.scalar_type() == at::kFloat, "factor must be FP32");
    TORCH_CHECK(rhs_col.scalar_type() == at::kFloat, "rhs must be FP32");
    TORCH_CHECK(
        factor_col.is_contiguous() && rhs_col.is_contiguous(),
        "inputs must be contiguous");
    TORCH_CHECK(
        factor_col.dim() == 3 && factor_col.size(1) == kRank
            && factor_col.size(2) == kRank,
        "factor_col must be [batch, 128, 128]");
    TORCH_CHECK(
        rhs_col.dim() == 3 && rhs_col.size(1) == kRhs
            && rhs_col.size(2) == kRank,
        "rhs_col must be [batch, 2, 128]");
    TORCH_CHECK(factor_col.size(0) == rhs_col.size(0), "batch sizes must match");
    TORCH_CHECK(factor_col.size(0) > 0, "batch size must be positive");
    TORCH_CHECK(
        factor_col.get_device() == rhs_col.get_device(), "devices must match");
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
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("mathdx_trsm128", &mathdx_trsm128_cuda);
    m.impl("mathdx_solve_frame128", &mathdx_solve_frame128_cuda);
}
