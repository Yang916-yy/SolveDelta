#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

// The 256-thread staged-output ownership follows FLA DPLR's MIT-licensed
// low-register backward schedule. SolveDelta supplies the coordinate-axis
// generalized-Delta recurrence and its H/R generator epilogue.

namespace {

constexpr int kChunk = 32;
constexpr int kWidth = 128;
constexpr int kBlock = 16;
constexpr int kWarps = 8;

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
__global__ __launch_bounds__(256, 2) void local_transpose_owner_kernel(
    const X* __restrict__ x,
    const Z* __restrict__ z,
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
    bool negate_z,
    bool accumulate) {
  const int panel = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int vector_base = panel * kChunk * kWidth;
  const int route_base = panel * NRHS * kChunk * kWidth;
  const int pair_base = panel * kChunk * kChunk;

  __shared__ c10::Half u_shared[kChunk * kWidth];
  __shared__ c10::BFloat16 h_shared[kChunk * kWidth];
  __shared__ float output_partial[2][kWarps][kChunk][kBlock];

  for (int index = tid; index < kChunk * kWidth; index += blockDim.x) {
    u_shared[index] = u[vector_base + index];
    h_shared[index] = h[vector_base + index];
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

  float dot_u[4 * NRHS] = {0.f};
  float dot_h[4 * NRHS] = {0.f};
#pragma unroll 4
  for (int coordinate = 0; coordinate < kWidth; ++coordinate) {
    const float uv = __bfloat162float(__float2bfloat16(
        __half2float(*reinterpret_cast<const __half*>(
            &u_shared[lane * kWidth + coordinate]))));
    const float hv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        &h_shared[lane * kWidth + coordinate]));
#pragma unroll
    for (int target_local = 0; target_local < 4; ++target_local) {
      const int target = warp * 4 + target_local;
#pragma unroll
      for (int rhs = 0; rhs < NRHS; ++rhs) {
        const int route = target_local * NRHS + rhs;
        const int input = route_base + (rhs * kChunk + target) * kWidth + coordinate;
        const float xv = __bfloat162float(
            __float2bfloat16(load_float(x + input)));
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

#pragma unroll
  for (int coordinate_block = 0; coordinate_block < kWidth / kBlock;
       ++coordinate_block) {
    float grad_u_block[kBlock] = {0.f};
    float grad_h_block[kBlock] = {0.f};
#pragma unroll
    for (int step = 0; step < kBlock; ++step) {
      const int linear = coordinate_block * kBlock + step;
      const int coordinate = lower ? kWidth - 1 - linear : linear;
      const float uv = __half2float(*reinterpret_cast<const __half*>(
          &u_shared[lane * kWidth + coordinate]));
      const float hv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
          &h_shared[lane * kWidth + coordinate]));
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
          const int input =
              route_base + (rhs * kChunk + target) * kWidth + coordinate;
          const float xv = load_float(x + input);
          float zv = load_float(z + input);
          zv = negate_z ? -zv : zv;
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
  local_transpose_owner_kernel<X, ZTYPE, RHS><<<grid, block, 0, stream>>>(   \
      x.data_ptr<X>(), z.data_ptr<ZTYPE>(), u.data_ptr<c10::Half>(),         \
      h.data_ptr<c10::BFloat16>(), decay.data_ptr<float>(),                  \
      kappa_h.data_ptr<float>(), kappa_r.data_ptr<float>(),                  \
      mass.data_ptr<float>(), grad_u.data_ptr<float>(),                      \
      grad_h.data_ptr<float>(), grad_kappa_h.data_ptr<float>(),              \
      grad_kappa_r.data_ptr<float>(), grad_cumulative.data_ptr<float>(),     \
      lower, negate_z, accumulate)
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

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("local_transpose", &local_transpose_owner);
}
