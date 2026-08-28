#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <type_traits>

// The 256-thread staged-output ownership follows FLA DPLR's MIT-licensed
// low-register backward schedule. SolveDelta supplies the coordinate-axis
// generalized-Delta recurrence and its H/R generator epilogue.

namespace {

constexpr int kChunk = 32;
constexpr int kWidth = 128;
constexpr int kBlock = 16;
constexpr int kWarps = 8;
constexpr int kMixedRhs = 3;
constexpr int kMixedRows = kChunk * kMixedRhs;
constexpr int kMixedRowTiles = kMixedRows / kBlock;

union __align__(32) UShared {
  __half half[kChunk * kWidth];
  __nv_bfloat16 bf16[kChunk * kWidth];
};

struct TensorCorePrefixScratch {
  float dot_u[kMixedRows * kChunk];
  float dot_h[kMixedRows * kChunk];
  __nv_bfloat16 x_tile[kMixedRowTiles * kBlock * kBlock];
};

union __align__(32) OutputScratch {
  float output_partial[2][kWarps][kChunk][kBlock];
  TensorCorePrefixScratch tensor_core;
};

template <typename T>
__device__ __forceinline__ float load_float(const T* pointer) {
  return static_cast<float>(*pointer);
}

template <>
__device__ __forceinline__ float load_float<c10::Half>(const c10::Half* pointer) {
  return __half2float(*reinterpret_cast<const __half*>(pointer));
}

template <>
__device__ __forceinline__ float load_float<c10::BFloat16>(
    const c10::BFloat16* pointer) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(pointer));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename X, typename Z, int NRHS>
struct HomogeneousRoutes {
  static constexpr int kRhs = NRHS;
  static constexpr bool kTensorCorePrefix = false;
  const X* x;
  const Z* z;
  bool negate_z;

  __device__ __forceinline__ float load_x(
      int panel, int target, int rhs, int coordinate) const {
    const int index =
        ((panel * NRHS + rhs) * kChunk + target) * kWidth + coordinate;
    return load_float(x + index);
  }

  __device__ __forceinline__ float load_z(
      int panel, int target, int rhs, int coordinate) const {
    const int index =
        ((panel * NRHS + rhs) * kChunk + target) * kWidth + coordinate;
    const float value = load_float(z + index);
    return negate_z ? -value : value;
  }
};

template <typename XP, typename XD>
struct MixedRoutes {
  static constexpr int kRhs = kMixedRhs;
  static constexpr bool kTensorCorePrefix = true;
  const XP* primal_x;
  const float* primal_z;
  const XD* dual_x;
  const c10::Half* dual_z;

  __device__ __forceinline__ float load_x(
      int panel, int target, int rhs, int coordinate) const {
    if (rhs == 0) {
      const int index = (panel * kChunk + target) * kWidth + coordinate;
      return load_float(primal_x + index);
    }
    const int index =
        ((panel * 2 + rhs - 1) * kChunk + target) * kWidth + coordinate;
    return load_float(dual_x + index);
  }

  __device__ __forceinline__ float load_z(
      int panel, int target, int rhs, int coordinate) const {
    if (rhs == 0) {
      const int index = (panel * kChunk + target) * kWidth + coordinate;
      return -primal_z[index];
    }
    const int index =
        ((panel * 2 + rhs - 1) * kChunk + target) * kWidth + coordinate;
    return load_float(dual_z + index);
  }
};

template <typename Routes>
__device__ __forceinline__ void tensor_core_prefix(
    Routes routes,
    int panel,
    int warp,
    int lane,
    const __nv_bfloat16* __restrict__ u,
    const __nv_bfloat16* __restrict__ h,
    TensorCorePrefixScratch* __restrict__ scratch) {
  using namespace nvcuda;
  if (warp < kMixedRowTiles) {
    wmma::fragment<wmma::matrix_a, kBlock, kBlock, kBlock,
                   __nv_bfloat16, wmma::row_major>
        x_fragment;
    wmma::fragment<wmma::matrix_b, kBlock, kBlock, kBlock,
                   __nv_bfloat16, wmma::col_major>
        source_0;
    wmma::fragment<wmma::matrix_b, kBlock, kBlock, kBlock,
                   __nv_bfloat16, wmma::col_major>
        source_1;
    wmma::fragment<wmma::accumulator, kBlock, kBlock, kBlock, float>
        dot_u_0;
    wmma::fragment<wmma::accumulator, kBlock, kBlock, kBlock, float>
        dot_u_1;
    wmma::fragment<wmma::accumulator, kBlock, kBlock, kBlock, float>
        dot_h_0;
    wmma::fragment<wmma::accumulator, kBlock, kBlock, kBlock, float>
        dot_h_1;
    wmma::fill_fragment(dot_u_0, 0.f);
    wmma::fill_fragment(dot_u_1, 0.f);
    wmma::fill_fragment(dot_h_0, 0.f);
    wmma::fill_fragment(dot_h_1, 0.f);

    __nv_bfloat16* x_tile =
        scratch->x_tile + warp * kBlock * kBlock;
    const int row_base = warp * kBlock;
#pragma unroll
    for (int coordinate_block = 0; coordinate_block < kWidth / kBlock;
         ++coordinate_block) {
      for (int element = lane; element < kBlock * kBlock; element += 32) {
        const int row = row_base + element / kBlock;
        const int target = row / kMixedRhs;
        const int rhs = row % kMixedRhs;
        const int coordinate =
            coordinate_block * kBlock + element % kBlock;
        x_tile[element] = __float2bfloat16(
            routes.load_x(panel, target, rhs, coordinate));
      }
      __syncwarp();
      wmma::load_matrix_sync(x_fragment, x_tile, kBlock);

      const int coordinate = coordinate_block * kBlock;
      wmma::load_matrix_sync(source_0, u + coordinate, kWidth);
      wmma::load_matrix_sync(
          source_1, u + kBlock * kWidth + coordinate, kWidth);
      wmma::mma_sync(dot_u_0, x_fragment, source_0, dot_u_0);
      wmma::mma_sync(dot_u_1, x_fragment, source_1, dot_u_1);

      wmma::load_matrix_sync(source_0, h + coordinate, kWidth);
      wmma::load_matrix_sync(
          source_1, h + kBlock * kWidth + coordinate, kWidth);
      wmma::mma_sync(dot_h_0, x_fragment, source_0, dot_h_0);
      wmma::mma_sync(dot_h_1, x_fragment, source_1, dot_h_1);
    }

    wmma::store_matrix_sync(
        scratch->dot_u + row_base * kChunk,
        dot_u_0, kChunk, wmma::mem_row_major);
    wmma::store_matrix_sync(
        scratch->dot_u + row_base * kChunk + kBlock,
        dot_u_1, kChunk, wmma::mem_row_major);
    wmma::store_matrix_sync(
        scratch->dot_h + row_base * kChunk,
        dot_h_0, kChunk, wmma::mem_row_major);
    wmma::store_matrix_sync(
        scratch->dot_h + row_base * kChunk + kBlock,
        dot_h_1, kChunk, wmma::mem_row_major);
  }
  __syncthreads();
}

template <typename Routes>
__global__ __launch_bounds__(256, 2) void local_transpose_owner_kernel(
    Routes routes,
    const c10::Half* __restrict__ u,
    const c10::BFloat16* __restrict__ h,
    const float* __restrict__ decay,
    const float* __restrict__ kappa_h,
    const float* __restrict__ kappa_r,
    const float* __restrict__ mass,
    float* __restrict__ grad_u,
    float* __restrict__ grad_h,
    float* __restrict__ grad_kappa_h,
    float* __restrict__ grad_kappa_r,
    float* __restrict__ grad_cumulative,
    bool lower,
    bool accumulate) {
  constexpr int NRHS = Routes::kRhs;
  const int panel = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int vector_base = panel * kChunk * kWidth;
  const int pair_base = panel * kChunk * kChunk;

  __shared__ UShared u_shared;
  __shared__ __align__(32) __nv_bfloat16 h_shared[kChunk * kWidth];
  __shared__ OutputScratch output_scratch;
  float (&output_partial)[2][kWarps][kChunk][kBlock] =
      output_scratch.output_partial;

  for (int index = tid; index < kChunk * kWidth; index += blockDim.x) {
    if constexpr (Routes::kTensorCorePrefix) {
      u_shared.bf16[index] = __float2bfloat16(__half2float(
          *reinterpret_cast<const __half*>(u + vector_base + index)));
    } else {
      u_shared.half[index] = *reinterpret_cast<const __half*>(
          u + vector_base + index);
    }
    h_shared[index] = *reinterpret_cast<const __nv_bfloat16*>(
        h + vector_base + index);
  }
  __syncthreads();

  float prefix[4 * NRHS] = {0.f};
  float suffix[4 * NRHS] = {0.f};
  float grad_weight_h[4] = {0.f, 0.f, 0.f, 0.f};
  float grad_weight_r[4] = {0.f, 0.f, 0.f, 0.f};
  float weight_h[4];
  float weight_r[4];

#pragma unroll
  for (int target_local = 0; target_local < 4; ++target_local) {
    const int target = warp * 4 + target_local;
    const float d = decay[pair_base + target * kChunk + lane];
    weight_h[target_local] = d * kappa_h[panel * kChunk + target];
    weight_r[target_local] = d * kappa_r[panel * kChunk + target];
  }

  if constexpr (Routes::kTensorCorePrefix) {
    tensor_core_prefix(
        routes, panel, warp, lane, u_shared.bf16, h_shared,
        &output_scratch.tensor_core);
#pragma unroll
    for (int target_local = 0; target_local < 4; ++target_local) {
      const int target = warp * 4 + target_local;
#pragma unroll
      for (int rhs = 0; rhs < NRHS; ++rhs) {
        const int route = target_local * NRHS + rhs;
        const int row = target * NRHS + rhs;
        const float dot_u =
            output_scratch.tensor_core.dot_u[row * kChunk + lane];
        const float dot_h =
            output_scratch.tensor_core.dot_h[row * kChunk + lane];
        prefix[route] = weight_h[target_local] * dot_u +
                        weight_r[target_local] * dot_h;
      }
    }
    for (int index = tid; index < kChunk * kWidth; index += blockDim.x) {
      u_shared.half[index] = *reinterpret_cast<const __half*>(
          u + vector_base + index);
    }
    __syncthreads();
  } else {
    float dot_u[4 * NRHS] = {0.f};
    float dot_h[4 * NRHS] = {0.f};
#pragma unroll 4
    for (int coordinate = 0; coordinate < kWidth; ++coordinate) {
      const float uv = __bfloat162float(__float2bfloat16(
          __half2float(u_shared.half[lane * kWidth + coordinate])));
      const float hv =
          __bfloat162float(h_shared[lane * kWidth + coordinate]);
#pragma unroll
      for (int target_local = 0; target_local < 4; ++target_local) {
        const int target = warp * 4 + target_local;
#pragma unroll
        for (int rhs = 0; rhs < NRHS; ++rhs) {
          const int route = target_local * NRHS + rhs;
          const float xv = __bfloat162float(__float2bfloat16(
              routes.load_x(panel, target, rhs, coordinate)));
          dot_u[route] = fmaf(xv, uv, dot_u[route]);
          dot_h[route] = fmaf(xv, hv, dot_h[route]);
        }
      }
    }
#pragma unroll
    for (int target_local = 0; target_local < 4; ++target_local) {
#pragma unroll
      for (int rhs = 0; rhs < NRHS; ++rhs) {
        const int route = target_local * NRHS + rhs;
        prefix[route] = weight_h[target_local] * dot_u[route] +
                        weight_r[target_local] * dot_h[route];
      }
    }
  }

#pragma unroll
  for (int coordinate_block = 0; coordinate_block < kWidth / kBlock;
       ++coordinate_block) {
    float grad_u_block[kBlock] = {0.f};
    float grad_h_block[kBlock] = {0.f};
#pragma unroll
    for (int step = 0; step < kBlock; ++step) {
      const int linear = coordinate_block * kBlock + step;
      const int coordinate = lower ? kWidth - 1 - linear : linear;
      const float uv =
          __half2float(u_shared.half[lane * kWidth + coordinate]);
      const float hv =
          __bfloat162float(h_shared[lane * kWidth + coordinate]);
      float grad_u_value = 0.f;
      float grad_h_value = 0.f;
#pragma unroll
      for (int target_local = 0; target_local < 4; ++target_local) {
        const int target = warp * 4 + target_local;
        const float generator = weight_h[target_local] * uv +
                                weight_r[target_local] * hv;
#pragma unroll
        for (int rhs = 0; rhs < NRHS; ++rhs) {
          const int route = target_local * NRHS + rhs;
          const float xv = routes.load_x(panel, target, rhs, coordinate);
          const float zv = routes.load_z(panel, target, rhs, coordinate);
          prefix[route] -= xv * generator;
          const float grad_generator = xv * suffix[route];
          grad_u_value +=
              zv * prefix[route] + grad_generator * weight_h[target_local];
          grad_h_value += grad_generator * weight_r[target_local];
          grad_weight_h[target_local] =
              fmaf(grad_generator, uv, grad_weight_h[target_local]);
          grad_weight_r[target_local] =
              fmaf(grad_generator, hv, grad_weight_r[target_local]);
          suffix[route] = fmaf(zv, uv, suffix[route]);
        }
      }
      grad_u_block[step] = grad_u_value;
      grad_h_block[step] = grad_h_value;
    }

#pragma unroll
    for (int step = 0; step < kBlock; ++step) {
      output_partial[0][warp][lane][step] = grad_u_block[step];
      output_partial[1][warp][lane][step] = grad_h_block[step];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int step = 0; step < kBlock; ++step) {
        float gu = 0.f;
        float gh = 0.f;
#pragma unroll
        for (int source_warp = 0; source_warp < kWarps; ++source_warp) {
          gu += output_partial[0][source_warp][lane][step];
          gh += output_partial[1][source_warp][lane][step];
        }
        const int linear = coordinate_block * kBlock + step;
        const int coordinate = lower ? kWidth - 1 - linear : linear;
        const int output = vector_base + lane * kWidth + coordinate;
        if (accumulate) {
          gu += grad_u[output];
          gh += grad_h[output];
        }
        grad_u[output] = gu;
        grad_h[output] = gh;
      }
    }
    __syncthreads();
  }

  // The block-output scratch is dead now. Reuse its first rows for the
  // target-row and source-column decay cotangents.
  float* row_partial = &output_partial[0][0][0][0];
  float* column_partial = row_partial + kChunk;
  float column_value = 0.f;
#pragma unroll
  for (int target_local = 0; target_local < 4; ++target_local) {
    const int target = warp * 4 + target_local;
    const float d = decay[pair_base + target * kChunk + lane];
    float gkh = grad_weight_h[target_local] * d;
    float gkr = grad_weight_r[target_local] * d;
    const float scaled =
        gkh * kappa_h[panel * kChunk + target] +
        gkr * kappa_r[panel * kChunk + target];
    column_value += scaled;
    gkh = warp_sum(gkh);
    gkr = warp_sum(gkr);
    const float row_value = warp_sum(scaled);
    if (lane == 0) {
      const bool valid = mass[panel * kChunk + target] > 0.f;
      const int output = panel * kChunk + target;
      const float prior_h = grad_kappa_h[output];
      const float prior_r = grad_kappa_r[output];
      grad_kappa_h[output] = prior_h + (valid ? gkh : 0.f);
      grad_kappa_r[output] = prior_r + (valid ? gkr : 0.f);
      row_partial[target] = row_value;
    }
  }
  column_partial[warp * kChunk + lane] = column_value;
  __syncthreads();
  if (warp == 0) {
    float column = 0.f;
#pragma unroll
    for (int source_warp = 0; source_warp < kWarps; ++source_warp) {
      column += column_partial[source_warp * kChunk + lane];
    }
    const int output = panel * kChunk + lane;
    const bool valid = mass[output] > 0.f;
    float value = valid ? row_partial[lane] - column : 0.f;
    value += grad_cumulative[output];
    grad_cumulative[output] = value;
  }
}

template <typename X>
void dispatch_z(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& u,
    const torch::Tensor& h,
    const torch::Tensor& decay,
    const torch::Tensor& kappa_h,
    const torch::Tensor& kappa_r,
    const torch::Tensor& mass,
    const torch::Tensor& grad_u,
    const torch::Tensor& grad_h,
    const torch::Tensor& grad_kappa_h,
    const torch::Tensor& grad_kappa_r,
    const torch::Tensor& grad_cumulative,
    bool lower,
    bool negate_z,
    bool accumulate) {
  const int panels = x.size(0);
  const int rhs_count = x.dim() == 3 ? 1 : x.size(1);
  const dim3 block(256);
  const dim3 grid(panels);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_Z_WITH_RHS(ZTYPE, RHS)                                        \
  do {                                                                        \
    using Routes = HomogeneousRoutes<X, ZTYPE, RHS>;                          \
    const Routes routes{x.data_ptr<X>(), z.data_ptr<ZTYPE>(), negate_z};      \
    local_transpose_owner_kernel<Routes><<<grid, block, 0, stream>>>(         \
      routes, u.data_ptr<c10::Half>(),                                        \
      h.data_ptr<c10::BFloat16>(), decay.data_ptr<float>(),                  \
      kappa_h.data_ptr<float>(), kappa_r.data_ptr<float>(),                  \
      mass.data_ptr<float>(), grad_u.data_ptr<float>(),                      \
      grad_h.data_ptr<float>(), grad_kappa_h.data_ptr<float>(),              \
      grad_kappa_r.data_ptr<float>(), grad_cumulative.data_ptr<float>(),     \
      lower, accumulate);                                                     \
  } while (false)
#define LAUNCH_Z_RHS1(ZTYPE) LAUNCH_Z_WITH_RHS(ZTYPE, 1)
#define LAUNCH_Z_RHS2(ZTYPE) LAUNCH_Z_WITH_RHS(ZTYPE, 2)
  if (rhs_count == 1) {
    if constexpr (
        std::is_same_v<X, c10::Half> ||
        std::is_same_v<X, c10::BFloat16>) {
      TORCH_CHECK(z.scalar_type() == at::kFloat,
                  "single-RHS local transpose requires FP32 cotangent");
      LAUNCH_Z_RHS1(float);
    } else {
      TORCH_CHECK(false, "unsupported single-RHS local transpose dtype");
    }
  } else {
    if constexpr (std::is_same_v<X, c10::Half>) {
      TORCH_CHECK(z.scalar_type() == at::kFloat,
                  "FP16 multi-RHS route requires FP32 cotangent");
      LAUNCH_Z_RHS2(float);
    } else if constexpr (std::is_same_v<X, float>) {
      TORCH_CHECK(z.scalar_type() == at::kHalf,
                  "FP32 multi-RHS route requires FP16 cotangent");
      LAUNCH_Z_RHS2(c10::Half);
    } else {
      TORCH_CHECK(
          z.scalar_type() == at::kFloat || z.scalar_type() == at::kHalf,
          "BF16 multi-RHS route requires FP16 or FP32 cotangent");
      if (z.scalar_type() == at::kFloat) {
        LAUNCH_Z_RHS2(float);
      } else {
        LAUNCH_Z_RHS2(c10::Half);
      }
    }
  }
#undef LAUNCH_Z_RHS1
#undef LAUNCH_Z_RHS2
#undef LAUNCH_Z_WITH_RHS
}

void local_transpose_owner(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& u,
    const torch::Tensor& h,
    const torch::Tensor& decay,
    const torch::Tensor& kappa_h,
    const torch::Tensor& kappa_r,
    const torch::Tensor& mass,
    const torch::Tensor& grad_u,
    const torch::Tensor& grad_h,
    const torch::Tensor& grad_kappa_h,
    const torch::Tensor& grad_kappa_r,
    const torch::Tensor& grad_cumulative,
    bool lower,
    bool negate_z,
    bool accumulate) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
  TORCH_CHECK(z.is_cuda() && z.is_contiguous(), "z must be contiguous CUDA");
  TORCH_CHECK(x.sizes() == z.sizes(), "x and z shapes must match");
  const int rhs_count = x.dim() == 3 ? 1 : x.size(1);
  TORCH_CHECK((x.dim() == 3 || x.dim() == 4) && rhs_count >= 1 &&
                  rhs_count <= 2 && x.size(-2) == kChunk &&
                  x.size(-1) == kWidth,
              "native owner requires [P,N,32,128] with N in {1,2}");
  TORCH_CHECK(u.scalar_type() == at::kHalf && h.scalar_type() == at::kBFloat16,
              "prototype requires FP16 u and BF16 h");
  TORCH_CHECK(decay.scalar_type() == at::kFloat &&
              kappa_h.scalar_type() == at::kFloat &&
              kappa_r.scalar_type() == at::kFloat &&
              mass.scalar_type() == at::kFloat,
              "geometry scalars must be FP32");
  TORCH_CHECK(grad_u.scalar_type() == at::kFloat &&
              grad_h.scalar_type() == at::kFloat &&
              grad_kappa_h.scalar_type() == at::kFloat &&
              grad_kappa_r.scalar_type() == at::kFloat &&
              grad_cumulative.scalar_type() == at::kFloat,
              "outputs must be FP32");
  c10::cuda::CUDAGuard guard(x.device());
  if (x.scalar_type() == at::kFloat) {
    dispatch_z<float>(x, z, u, h, decay, kappa_h, kappa_r, mass, grad_u,
                      grad_h, grad_kappa_h, grad_kappa_r, grad_cumulative,
                      lower, negate_z, accumulate);
  } else if (x.scalar_type() == at::kHalf) {
    dispatch_z<c10::Half>(x, z, u, h, decay, kappa_h, kappa_r, mass, grad_u,
                          grad_h, grad_kappa_h, grad_kappa_r, grad_cumulative,
                          lower, negate_z, accumulate);
  } else if (x.scalar_type() == at::kBFloat16) {
    dispatch_z<c10::BFloat16>(x, z, u, h, decay, kappa_h, kappa_r, mass,
                              grad_u, grad_h, grad_kappa_h, grad_kappa_r,
                              grad_cumulative, lower, negate_z, accumulate);
  } else {
    TORCH_CHECK(false, "unsupported x dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename XP>
void dispatch_mixed_dual(
    const torch::Tensor& primal_x,
    const torch::Tensor& primal_z,
    const torch::Tensor& dual_x,
    const torch::Tensor& dual_z,
    const torch::Tensor& u,
    const torch::Tensor& h,
    const torch::Tensor& decay,
    const torch::Tensor& kappa_h,
    const torch::Tensor& kappa_r,
    const torch::Tensor& mass,
    const torch::Tensor& grad_u,
    const torch::Tensor& grad_h,
    const torch::Tensor& grad_kappa_h,
    const torch::Tensor& grad_kappa_r,
    const torch::Tensor& grad_cumulative,
    bool lower,
    bool accumulate) {
  const int panels = primal_x.size(0);
  const dim3 block(256);
  const dim3 grid(panels);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_MIXED(XD)                                                      \
  do {                                                                        \
    using Routes = MixedRoutes<XP, XD>;                                       \
    const Routes routes{                                                      \
        primal_x.data_ptr<XP>(), primal_z.data_ptr<float>(),                  \
        dual_x.data_ptr<XD>(), dual_z.data_ptr<c10::Half>()};                 \
    local_transpose_owner_kernel<Routes><<<grid, block, 0, stream>>>(         \
        routes, u.data_ptr<c10::Half>(), h.data_ptr<c10::BFloat16>(),         \
        decay.data_ptr<float>(), kappa_h.data_ptr<float>(),                   \
        kappa_r.data_ptr<float>(), mass.data_ptr<float>(),                    \
        grad_u.data_ptr<float>(), grad_h.data_ptr<float>(),                   \
        grad_kappa_h.data_ptr<float>(), grad_kappa_r.data_ptr<float>(),       \
        grad_cumulative.data_ptr<float>(), lower, accumulate);                \
  } while (false)
  if (dual_x.scalar_type() == at::kFloat) {
    LAUNCH_MIXED(float);
  } else if (dual_x.scalar_type() == at::kBFloat16) {
    LAUNCH_MIXED(c10::BFloat16);
  } else {
    TORCH_CHECK(false, "mixed local transpose requires FP32 or BF16 dual x");
  }
#undef LAUNCH_MIXED
}

void local_transpose_mixed_owner(
    const torch::Tensor& primal_x,
    const torch::Tensor& primal_z,
    const torch::Tensor& dual_x,
    const torch::Tensor& dual_z,
    const torch::Tensor& u,
    const torch::Tensor& h,
    const torch::Tensor& decay,
    const torch::Tensor& kappa_h,
    const torch::Tensor& kappa_r,
    const torch::Tensor& mass,
    const torch::Tensor& grad_u,
    const torch::Tensor& grad_h,
    const torch::Tensor& grad_kappa_h,
    const torch::Tensor& grad_kappa_r,
    const torch::Tensor& grad_cumulative,
    bool lower,
    bool accumulate) {
  TORCH_CHECK(primal_x.is_cuda() && primal_x.is_contiguous(),
              "primal x must be contiguous CUDA");
  TORCH_CHECK(primal_z.is_cuda() && primal_z.is_contiguous(),
              "primal z must be contiguous CUDA");
  TORCH_CHECK(dual_x.is_cuda() && dual_x.is_contiguous(),
              "dual x must be contiguous CUDA");
  TORCH_CHECK(dual_z.is_cuda() && dual_z.is_contiguous(),
              "dual z must be contiguous CUDA");
  TORCH_CHECK(primal_x.sizes() == primal_z.sizes(),
              "primal x/z shapes must match");
  TORCH_CHECK(dual_x.sizes() == dual_z.sizes(),
              "dual x/z shapes must match");
  TORCH_CHECK(primal_x.dim() == 4 && primal_x.size(1) == 1 &&
                  primal_x.size(2) == kChunk && primal_x.size(3) == kWidth,
              "primal route must have shape [P,1,32,128]");
  TORCH_CHECK(dual_x.dim() == 4 && dual_x.size(1) == 2 &&
                  dual_x.size(2) == kChunk && dual_x.size(3) == kWidth &&
                  dual_x.size(0) == primal_x.size(0),
              "dual route must have shape [P,2,32,128]");
  TORCH_CHECK(primal_z.scalar_type() == at::kFloat &&
                  dual_z.scalar_type() == at::kHalf,
              "mixed cotangents must be FP32 primal and FP16 dual");
  TORCH_CHECK(u.scalar_type() == at::kHalf && h.scalar_type() == at::kBFloat16,
              "mixed owner requires FP16 u and BF16 h");
  TORCH_CHECK(decay.scalar_type() == at::kFloat &&
                  kappa_h.scalar_type() == at::kFloat &&
                  kappa_r.scalar_type() == at::kFloat &&
                  mass.scalar_type() == at::kFloat,
              "geometry scalars must be FP32");
  TORCH_CHECK(grad_u.scalar_type() == at::kFloat &&
                  grad_h.scalar_type() == at::kFloat &&
                  grad_kappa_h.scalar_type() == at::kFloat &&
                  grad_kappa_r.scalar_type() == at::kFloat &&
                  grad_cumulative.scalar_type() == at::kFloat,
              "outputs must be FP32");
  c10::cuda::CUDAGuard guard(primal_x.device());
  if (primal_x.scalar_type() == at::kHalf) {
    dispatch_mixed_dual<c10::Half>(
        primal_x, primal_z, dual_x, dual_z, u, h, decay, kappa_h, kappa_r,
        mass, grad_u, grad_h, grad_kappa_h, grad_kappa_r, grad_cumulative,
        lower, accumulate);
  } else if (primal_x.scalar_type() == at::kBFloat16) {
    dispatch_mixed_dual<c10::BFloat16>(
        primal_x, primal_z, dual_x, dual_z, u, h, decay, kappa_h, kappa_r,
        mass, grad_u, grad_h, grad_kappa_h, grad_kappa_r, grad_cumulative,
        lower, accumulate);
  } else {
    TORCH_CHECK(false, "mixed local transpose requires FP16 or BF16 primal x");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("local_transpose", &local_transpose_owner);
  module.def("local_transpose_mixed", &local_transpose_mixed_owner);
}
