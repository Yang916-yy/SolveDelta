#include <torch/library.h>
#include <ATen/ATen.h>
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
constexpr int kComponents = 4;
constexpr int kDescriptors = 3;
constexpr int kTile = 16;
constexpr int kTiles = kRank / kTile;
constexpr int kMatrixTiles = kTiles * kTiles;
constexpr int kEntryThreads = kTile * kTile;
constexpr int kScalarSlots = kChunk + 2;
constexpr float kRadius = 1.0f / 8.0f;

struct FloatFloat {
    float hi = 0.0f;
    float lo = 0.0f;

    __device__ __forceinline__ void add(float value) {
        const float next = __fadd_rn(hi, value);
        const float bridge = __fsub_rn(next, hi);
        const float error = __fadd_rn(
            __fsub_rn(hi, __fsub_rn(next, bridge)),
            __fsub_rn(value, bridge));
        hi = next;
        lo = __fadd_rn(lo, error);
    }

    __device__ __forceinline__ void add_product(float left, float right) {
        const float product = __fmul_rn(left, right);
        const float error = __fmaf_rn(left, right, -product);
        add(product);
        lo = __fadd_rn(lo, error);
    }

    __device__ __forceinline__ void add_triple(
        float first, float second, float third) {
        const float product = __fmul_rn(first, second);
        const float error = __fmaf_rn(first, second, -product);
        add_product(product, third);
        lo = __fmaf_rn(error, third, lo);
    }

    __device__ __forceinline__ void add_accumulator(
        const FloatFloat& other) {
        add(other.hi);
        add(other.lo);
    }
};

__device__ __forceinline__ FloatFloat scaled(
    const FloatFloat& value, float factor) {
    FloatFloat result;
    result.add_product(value.hi, factor);
    result.add_product(value.lo, factor);
    return result;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(
            value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float block_sum(
    float value, float* warp_workspace) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) warp_workspace[warp] = value;
    __syncthreads();
    float total = threadIdx.x < 8 ? warp_workspace[lane] : 0.0f;
    if (warp == 0) total = warp_sum(total);
    __syncthreads();
    return total;
}

__device__ __forceinline__ FloatFloat block_sum_floatfloat(
    FloatFloat value, float* warp_workspace) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        FloatFloat other;
        other.hi = __shfl_down_sync(0xffffffffu, value.hi, offset);
        other.lo = __shfl_down_sync(0xffffffffu, value.lo, offset);
        value.add_accumulator(other);
    }
    if (lane == 0) {
        warp_workspace[warp] = value.hi;
        warp_workspace[8 + warp] = value.lo;
    }
    __syncthreads();
    FloatFloat total;
    if (threadIdx.x < 8) {
        total.hi = warp_workspace[lane];
        total.lo = warp_workspace[8 + lane];
    }
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            FloatFloat other;
            other.hi = __shfl_down_sync(0xffffffffu, total.hi, offset);
            other.lo = __shfl_down_sync(0xffffffffu, total.lo, offset);
            total.add_accumulator(other);
        }
    }
    __syncthreads();
    return total;
}

__device__ __forceinline__ void block_sum4(
    float (&value)[kComponents], float* warp_workspace) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int component = 0; component < kComponents; ++component) {
        value[component] = warp_sum(value[component]);
        if (lane == 0) {
            warp_workspace[component * 8 + warp] = value[component];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int component = 0; component < kComponents; ++component) {
            float total = lane < 8
                ? warp_workspace[component * 8 + lane] : 0.0f;
            total = warp_sum(total);
            if (lane == 0) value[component] = total;
        }
    }
    __syncthreads();
}

constexpr int descriptor_shared_floats() {
    return 3 * kChunk * kTile
        + 4 * kChunk * kDescriptors * kTile
        + kComponents * 8;
}

__launch_bounds__(kEntryThreads, 1)
__global__ void descriptor_contraction_kernel(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ alpha0,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ upper_left,
    const float* __restrict__ upper_right,
    const float* __restrict__ lower_left,
    const float* __restrict__ lower_right,
    float* __restrict__ partial_coefficient,
    int chunks,
    int length,
    int panels) {
    extern __shared__ float shared[];
    float* u_row = shared;
    float* u_col = u_row + kChunk * kTile;
    float* h_col = u_col + kChunk * kTile;
    float* descriptor = h_col + kChunk * kTile;
    float* warp_workspace = descriptor
        + 4 * kChunk * kDescriptors * kTile;

    const int panel = blockIdx.x;
    const int tile = blockIdx.y;
    const int tid = threadIdx.x;
    if (panel >= panels) return;
    const int local_row = tid / kTile;
    const int local_col = tid % kTile;
    const int row_tile = tile / kTiles;
    const int col_tile = tile % kTiles;
    const int row = row_tile * kTile + local_row;
    const int col = col_tile * kTile + local_col;
    const int chunk = panel % chunks;
    const int matrix_base = panel * kRank * kRank;
    const int vector_base = panel * kChunk * kRank;

    for (int index = tid; index < kChunk * kTile;
         index += kEntryThreads) {
        const int target = index / kTile;
        const int coordinate = index % kTile;
        u_row[index] = u[
            vector_base + target * kRank + row_tile * kTile + coordinate];
        u_col[index] = u[
            vector_base + target * kRank + col_tile * kTile + coordinate];
        h_col[index] = h[
            vector_base + target * kRank + col_tile * kTile + coordinate];
    }
    for (int index = tid;
         index < 4 * kChunk * kDescriptors * kTile;
         index += kEntryThreads) {
        const int which = index / (kChunk * kDescriptors * kTile);
        const int remainder = index % (kChunk * kDescriptors * kTile);
        const int target = remainder / (kDescriptors * kTile);
        const int descriptor_index = (remainder / kTile) % kDescriptors;
        const int coordinate = remainder % kTile;
        const float* source = which == 0 ? upper_left
            : which == 1 ? upper_right
            : which == 2 ? lower_left : lower_right;
        const int coordinate_tile = (which & 1) == 0 ? row_tile : col_tile;
        descriptor[index] = source[
            ((panel * kChunk + target) * kDescriptors + descriptor_index)
                * kRank
            + coordinate_tile * kTile + coordinate];
    }
    __syncthreads();

    const float boundary_j_value = boundary_j[
        matrix_base + row * kRank + col];
    const float boundary_d_value = boundary_d[
        matrix_base + row * kRank + col];
    float current_j = 0.0f;
    float current_d = 0.0f;
#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        const bool valid = chunk * kChunk + target < length;
        if (valid) {
            const float ui = u_row[target * kTile + local_row];
            const float uj = u_col[target * kTile + local_col];
            const float hj = h_col[target * kTile + local_col];
            const float w = inverse_mass[panel * kChunk + target];
            if (target == 0) {
                current_j = fmaf(alpha0[panel], boundary_j_value, w * ui * uj);
                current_d = fmaf(alpha0[panel], boundary_d_value, w * ui * hj);
            } else {
                current_j = fmaf(1.0f - w, current_j, w * ui * uj);
                current_d = fmaf(1.0f - w, current_d, w * ui * hj);
            }
        }
        float factor_cotangent = 0.0f;
        if (valid && row != col) {
            const int side = row < col ? 0 : 2;
            const int left_base =
                (side + 0) * kChunk * kDescriptors * kTile
                + target * kDescriptors * kTile;
            const int right_base =
                (side + 1) * kChunk * kDescriptors * kTile
                + target * kDescriptors * kTile;
#pragma unroll
            for (int item = 0; item < kDescriptors; ++item) {
                factor_cotangent = fmaf(
                    descriptor[left_base + item * kTile + local_row],
                    descriptor[right_base + item * kTile + local_col],
                    factor_cotangent);
            }
        }
        float values[kComponents] = {};
        if (row > col) {
            values[0] = factor_cotangent * current_j;
            values[1] = factor_cotangent * current_d;
        } else if (row < col) {
            values[2] = factor_cotangent * current_j;
            values[3] = factor_cotangent * current_d;
        }
        block_sum4(values, warp_workspace);
        if (tid == 0) {
            const int output =
                ((panel * kMatrixTiles + tile) * kChunk + target)
                * kComponents;
#pragma unroll
            for (int component = 0; component < kComponents; ++component) {
                partial_coefficient[output + component] = valid
                    ? values[component] : 0.0f;
            }
        }
    }
}

__global__ void coefficient_partial_reduce_kernel(
    const float* __restrict__ partial_coefficient,
    float* __restrict__ grad_coefficient,
    int items) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= items) return;
    const int component = item % kComponents;
    const int scalar = item / kComponents;
    const int panel = scalar / kChunk;
    const int target = scalar % kChunk;
    float total = 0.0f;
#pragma unroll
    for (int tile = 0; tile < kMatrixTiles; ++tile) {
        total += partial_coefficient[
            ((panel * kMatrixTiles + tile) * kChunk + target)
                * kComponents + component];
    }
    grad_coefficient[item] = total;
}

__launch_bounds__(128, 4)
__global__ void radial_scalars_kernel(
    const float* __restrict__ strength,
    const float* __restrict__ norm_sq,
    const float* __restrict__ grad_coefficient,
    float* __restrict__ eta,
    float* __restrict__ coefficient_strength,
    int heads,
    int chunks,
    int length,
    int panels) {
    __shared__ float partial[4];
    const int panel = blockIdx.x;
    const int tid = threadIdx.x;
    if (panel >= panels) return;
    const int chunk = panel % chunks;
    const int head = (panel / chunks) % heads;
    const int item = tid;
    float strength_value = 0.0f;
    if (item < kChunk * kComponents) {
        const int target = item / kComponents;
        const bool valid = chunk * kChunk + target < length;
        const float g = strength[head];
        const float norm = norm_sq[panel * kChunk * kComponents + item];
        const float denominator = kRadius * kRadius + norm;
        const float inverse_root = rsqrtf(denominator);
        const float upstream = grad_coefficient[
            panel * kChunk * kComponents + item];
        eta[panel * kChunk * kComponents + item] = valid
            ? -upstream * g * g * g * kRadius
                * inverse_root / denominator
            : 0.0f;
        strength_value = valid
            ? upstream * kRadius * kRadius * kRadius
                * inverse_root / denominator
            : 0.0f;
    }
    strength_value = warp_sum(strength_value);
    if ((tid & 31) == 0) partial[tid >> 5] = strength_value;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
#pragma unroll
        for (int warp = 0; warp < 4; ++warp) total += partial[warp];
        coefficient_strength[panel] = total;
    }
}

constexpr int entry_shared_floats() {
    return 2 * kChunk * kEntryThreads
        + 3 * kChunk * kTile
        + 4 * kChunk * kDescriptors * kTile
        + 3 * kEntryThreads
        + 16;
}

__launch_bounds__(kEntryThreads, 1)
__global__ void radial_entry_vjp_kernel(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ strength,
    const float* __restrict__ alpha0,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ coefficient,
    const float* __restrict__ eta,
    const float* __restrict__ upper_left,
    const float* __restrict__ upper_right,
    const float* __restrict__ lower_left,
    const float* __restrict__ lower_right,
    const float* __restrict__ grad_diagonal,
    float* __restrict__ grad_boundary_j,
    float* __restrict__ grad_boundary_d,
    float* __restrict__ partial_row_u,
    float* __restrict__ partial_col_u,
    float* __restrict__ partial_col_h,
    float* __restrict__ partial_scalar,
    float* __restrict__ partial_scalar_lo,
    int heads,
    int chunks,
    int length,
    int panels) {
    extern __shared__ float shared[];
    float* state_j = shared;
    float* state_d = state_j + kChunk * kEntryThreads;
    float* u_row = state_d + kChunk * kEntryThreads;
    float* u_col = u_row + kChunk * kTile;
    float* h_col = u_col + kChunk * kTile;
    float* descriptor = h_col + kChunk * kTile;
    float* row_contribution = descriptor
        + 4 * kChunk * kDescriptors * kTile;
    float* col_u_contribution = row_contribution + kEntryThreads;
    float* col_h_contribution = col_u_contribution + kEntryThreads;
    float* warp_workspace = col_h_contribution + kEntryThreads;

    const int panel = blockIdx.x;
    const int tile = blockIdx.y;
    const int tid = threadIdx.x;
    if (panel >= panels) return;
    const int local_row = tid / kTile;
    const int local_col = tid % kTile;
    const int row_tile = tile / kTiles;
    const int col_tile = tile % kTiles;
    const int row = row_tile * kTile + local_row;
    const int col = col_tile * kTile + local_col;
    const int chunk = panel % chunks;
    const int head = (panel / chunks) % heads;
    const int matrix_base = panel * kRank * kRank;
    const int vector_base = panel * kChunk * kRank;

    for (int index = tid; index < kChunk * kTile;
         index += kEntryThreads) {
        const int target = index / kTile;
        const int coordinate = index % kTile;
        u_row[index] = u[
            vector_base + target * kRank + row_tile * kTile + coordinate];
        u_col[index] = u[
            vector_base + target * kRank + col_tile * kTile + coordinate];
        h_col[index] = h[
            vector_base + target * kRank + col_tile * kTile + coordinate];
    }
    for (int index = tid;
         index < 4 * kChunk * kDescriptors * kTile;
         index += kEntryThreads) {
        const int which = index / (kChunk * kDescriptors * kTile);
        const int remainder = index % (kChunk * kDescriptors * kTile);
        const int target = remainder / (kDescriptors * kTile);
        const int descriptor_index = (remainder / kTile) % kDescriptors;
        const int coordinate = remainder % kTile;
        const float* source = which == 0 ? upper_left
            : which == 1 ? upper_right
            : which == 2 ? lower_left : lower_right;
        const int coordinate_tile = (which & 1) == 0 ? row_tile : col_tile;
        descriptor[index] = source[
            ((panel * kChunk + target) * kDescriptors + descriptor_index)
                * kRank
            + coordinate_tile * kTile + coordinate];
    }
    __syncthreads();

    const float boundary_j_value = boundary_j[
        matrix_base + row * kRank + col];
    const float boundary_d_value = boundary_d[
        matrix_base + row * kRank + col];
    float current_j = 0.0f;
    float current_d = 0.0f;
#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        const bool valid = chunk * kChunk + target < length;
        if (valid) {
            const float ui = u_row[target * kTile + local_row];
            const float uj = u_col[target * kTile + local_col];
            const float hj = h_col[target * kTile + local_col];
            const float w = inverse_mass[panel * kChunk + target];
            if (target == 0) {
                current_j = fmaf(alpha0[panel], boundary_j_value, w * ui * uj);
                current_d = fmaf(alpha0[panel], boundary_d_value, w * ui * hj);
            } else {
                current_j = fmaf(1.0f - w, current_j, w * ui * uj);
                current_d = fmaf(1.0f - w, current_d, w * ui * hj);
            }
        }
        state_j[target * kEntryThreads + tid] = valid ? current_j : 0.0f;
        state_d[target * kEntryThreads + tid] = valid ? current_d : 0.0f;
    }
    __syncthreads();

    const float g = strength[head];
    float adjoint_j = 0.0f;
    float adjoint_d = 0.0f;
    FloatFloat scaled_mass_adjoint;
    FloatFloat boundary_mass_entry;
    float diagonal_strength_entry = 0.0f;

    for (int target = kChunk - 1; target >= 0; --target) {
        const bool valid = chunk * kChunk + target < length;
        float outer_grad_j = 0.0f;
        float outer_grad_d = 0.0f;
        if (valid) {
            const int state_index = target * kEntryThreads + tid;
            const float sj = state_j[state_index];
            const float sd = state_d[state_index];
            float state_grad_j = 0.0f;
            float state_grad_d = 0.0f;
            if (row > col) {
                state_grad_j = eta[
                    (panel * kChunk + target) * kComponents + 0] * sj;
                state_grad_d = eta[
                    (panel * kChunk + target) * kComponents + 1] * sd;
            } else if (row < col) {
                state_grad_j = eta[
                    (panel * kChunk + target) * kComponents + 2] * sj;
                state_grad_d = eta[
                    (panel * kChunk + target) * kComponents + 3] * sd;
            } else {
                const float base_h = sj - 1.0f / static_cast<float>(kRank);
                const float base_r = sd;
                const float th = tanhf(g * base_h / kRadius);
                const float tr = tanhf(g * base_r / kRadius);
                const float sigma = expf(kRadius * th + kRadius * tr);
                const float grad_log = grad_diagonal[
                    vector_base + target * kRank + row] * sigma;
                const float sech_h = 1.0f - th * th;
                const float sech_r = 1.0f - tr * tr;
                state_grad_j = grad_log * g * sech_h;
                state_grad_d = grad_log * g * sech_r;
                diagonal_strength_entry += grad_log
                    * (base_h * sech_h + base_r * sech_r);
            }
            if (row != col) {
                const int side = row < col ? 0 : 2;
                const int left_base =
                    (side + 0) * kChunk * kDescriptors * kTile
                    + target * kDescriptors * kTile;
                const int right_base =
                    (side + 1) * kChunk * kDescriptors * kTile
                    + target * kDescriptors * kTile;
                float factor_cotangent = 0.0f;
#pragma unroll
                for (int item = 0; item < kDescriptors; ++item) {
                    factor_cotangent = fmaf(
                        descriptor[left_base + item * kTile + local_row],
                        descriptor[right_base + item * kTile + local_col],
                        factor_cotangent);
                }
                const int component = row < col ? 2 : 0;
                state_grad_j = fmaf(
                    coefficient[
                        (panel * kChunk + target) * kComponents + component],
                    factor_cotangent,
                    state_grad_j);
                state_grad_d = fmaf(
                    coefficient[
                        (panel * kChunk + target) * kComponents + component + 1],
                    factor_cotangent,
                    state_grad_d);
            }
            adjoint_j += state_grad_j;
            adjoint_d += state_grad_d;

            const float ui = u_row[target * kTile + local_row];
            const float uj = u_col[target * kTile + local_col];
            const float hj = h_col[target * kTile + local_col];
            const float w = inverse_mass[panel * kChunk + target];
            outer_grad_j = w * adjoint_j;
            outer_grad_d = w * adjoint_d;
            if (target == 0) {
                boundary_mass_entry = scaled(
                    scaled_mass_adjoint, alpha0[panel]);
                boundary_mass_entry.add_triple(
                    -alpha0[panel], adjoint_j, sj);
                boundary_mass_entry.add_triple(
                    -alpha0[panel], adjoint_d, sd);
            }
            FloatFloat next_mass_adjoint = scaled(
                scaled_mass_adjoint, 1.0f - w);
            const float weighted_j = w * adjoint_j;
            const float weighted_d = w * adjoint_d;
            next_mass_adjoint.add_product(weighted_j, sj);
            next_mass_adjoint.add_triple(-weighted_j, ui, uj);
            next_mass_adjoint.add_product(weighted_d, sd);
            next_mass_adjoint.add_triple(-weighted_d, ui, hj);
            scaled_mass_adjoint = next_mass_adjoint;
            if (target > 0) {
                adjoint_j *= 1.0f - w;
                adjoint_d *= 1.0f - w;
            }

            row_contribution[tid] = fmaf(
                outer_grad_j, uj, outer_grad_d * hj);
            col_u_contribution[tid] = outer_grad_j * ui;
            col_h_contribution[tid] = outer_grad_d * ui;
        } else {
            row_contribution[tid] = 0.0f;
            col_u_contribution[tid] = 0.0f;
            col_h_contribution[tid] = 0.0f;
        }

        const FloatFloat scaled_mass_sum = block_sum_floatfloat(
            scaled_mass_adjoint, warp_workspace);
        if (tid < kTile) {
            float row_sum = 0.0f;
            float col_u_sum = 0.0f;
            float col_h_sum = 0.0f;
#pragma unroll
            for (int other = 0; other < kTile; ++other) {
                row_sum += row_contribution[tid * kTile + other];
                col_u_sum += col_u_contribution[other * kTile + tid];
                col_h_sum += col_h_contribution[other * kTile + tid];
            }
            const int partial_base =
                (((panel * kChunk + target) * kMatrixTiles + tile) * kTile);
            partial_row_u[partial_base + tid] = row_sum;
            partial_col_u[partial_base + tid] = col_u_sum;
            partial_col_h[partial_base + tid] = col_h_sum;
        }
        if (tid == 0) {
            partial_scalar[
                (panel * kMatrixTiles + tile) * kScalarSlots + target]
                = valid ? scaled_mass_sum.hi : 0.0f;
            partial_scalar_lo[
                (panel * kMatrixTiles + tile) * kScalarSlots + target]
                = valid ? scaled_mass_sum.lo : 0.0f;
        }
        __syncthreads();
    }

    const FloatFloat boundary_mass_sum = block_sum_floatfloat(
        boundary_mass_entry, warp_workspace);
    const float diagonal_strength_sum = block_sum(
        diagonal_strength_entry, warp_workspace);
    if (tid == 0) {
        const int scalar_base =
            (panel * kMatrixTiles + tile) * kScalarSlots;
        partial_scalar[scalar_base + kChunk] = boundary_mass_sum.hi;
        partial_scalar_lo[scalar_base + kChunk] = boundary_mass_sum.lo;
        partial_scalar[scalar_base + kChunk + 1] = diagonal_strength_sum;
        partial_scalar_lo[scalar_base + kChunk + 1] = 0.0f;
    }
    grad_boundary_j[matrix_base + row * kRank + col]
        = alpha0[panel] * adjoint_j;
    grad_boundary_d[matrix_base + row * kRank + col]
        = alpha0[panel] * adjoint_d;
}

__global__ void vector_partial_reduce_kernel(
    const float* __restrict__ partial_row_u,
    const float* __restrict__ partial_col_u,
    const float* __restrict__ partial_col_h,
    float* __restrict__ grad_u,
    float* __restrict__ grad_h,
    int items) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= items) return;
    const int coordinate = item % kRank;
    const int token_panel = item / kRank;
    const int coordinate_tile = coordinate / kTile;
    const int local = coordinate % kTile;
    float gu = 0.0f;
    float gh = 0.0f;
#pragma unroll
    for (int other = 0; other < kTiles; ++other) {
        const int row_tile = coordinate_tile * kTiles + other;
        const int col_tile = other * kTiles + coordinate_tile;
        gu += partial_row_u[
            (token_panel * kMatrixTiles + row_tile) * kTile + local];
        gu += partial_col_u[
            (token_panel * kMatrixTiles + col_tile) * kTile + local];
        gh += partial_col_h[
            (token_panel * kMatrixTiles + col_tile) * kTile + local];
    }
    grad_u[item] = gu;
    grad_h[item] = gh;
}

__launch_bounds__(64, 4)
__global__ void scalar_partial_reduce_kernel(
    const float* __restrict__ partial_scalar,
    const float* __restrict__ partial_scalar_lo,
    const float* __restrict__ coefficient_strength,
    float* __restrict__ grad_boundary_m,
    float* __restrict__ grad_log_decay,
    float* __restrict__ panel_strength,
    int panels) {
    const int panel = blockIdx.x;
    const int tid = threadIdx.x;
    if (panel >= panels) return;
    if (tid < kChunk) {
        FloatFloat total;
#pragma unroll
        for (int tile = 0; tile < kMatrixTiles; ++tile) {
            const int index =
                (panel * kMatrixTiles + tile) * kScalarSlots + tid;
            FloatFloat value;
            value.hi = partial_scalar[index];
            value.lo = partial_scalar_lo[index];
            total.add_accumulator(value);
        }
        grad_log_decay[panel * kChunk + tid] = total.hi + total.lo;
    } else if (tid == kChunk) {
        FloatFloat total;
#pragma unroll
        for (int tile = 0; tile < kMatrixTiles; ++tile) {
            const int index =
                (panel * kMatrixTiles + tile) * kScalarSlots + kChunk;
            FloatFloat value;
            value.hi = partial_scalar[index];
            value.lo = partial_scalar_lo[index];
            total.add_accumulator(value);
        }
        grad_boundary_m[panel] = total.hi + total.lo;
    } else if (tid == kChunk + 1) {
        float total = coefficient_strength[panel];
#pragma unroll
        for (int tile = 0; tile < kMatrixTiles; ++tile) {
            total += partial_scalar[
                (panel * kMatrixTiles + tile) * kScalarSlots + kChunk + 1];
        }
        panel_strength[panel] = total;
    }
}

__launch_bounds__(64, 1)
__global__ void strength_reduce_kernel(
    const float* __restrict__ panel_strength,
    float* __restrict__ grad_strength,
    int heads,
    int chunks,
    int batches) {
    const int head = blockIdx.x;
    if (head >= heads || threadIdx.x != 0) return;
    float total = 0.0f;
    for (int batch = 0; batch < batches; ++batch) {
        for (int chunk = 0; chunk < chunks; ++chunk) {
            total += panel_strength[(batch * heads + head) * chunks + chunk];
        }
    }
    grad_strength[head] = total;
}

void check_inputs(
    const at::Tensor& boundary_m,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& log_decay,
    const at::Tensor& strength,
    const at::Tensor& alpha0,
    const at::Tensor& inverse_mass,
    const at::Tensor& coefficient,
    const at::Tensor& norm_sq,
    const at::Tensor& upper_left,
    const at::Tensor& upper_right,
    const at::Tensor& lower_left,
    const at::Tensor& lower_right,
    const at::Tensor& grad_diagonal,
    int64_t heads,
    int64_t chunks,
    int64_t length) {
    TORCH_CHECK(
        boundary_j.is_cuda() && boundary_j.scalar_type() == at::kFloat
            && boundary_j.is_contiguous() && boundary_j.dim() == 3
            && boundary_j.size(1) == kRank && boundary_j.size(2) == kRank,
        "boundary_j must be contiguous CUDA FP32 [P,128,128]");
    const int64_t panels = boundary_j.size(0);
    const auto check = [&](const at::Tensor& value) {
        TORCH_CHECK(
            value.is_cuda() && value.get_device() == boundary_j.get_device()
                && value.scalar_type() == at::kFloat && value.is_contiguous(),
            "all radial VJP inputs must be contiguous CUDA FP32 on one device");
    };
    for (const auto& value : {
             boundary_m, boundary_d, u, h, log_decay, strength,
             alpha0, inverse_mass, coefficient, norm_sq,
             upper_left, upper_right, lower_left, lower_right,
             grad_diagonal}) check(value);
    TORCH_CHECK(
        panels > 0 && panels <= std::numeric_limits<int>::max(),
        "invalid panel count");
    TORCH_CHECK(
        boundary_m.sizes() == at::IntArrayRef({panels}),
        "boundary_m shape mismatch");
    TORCH_CHECK(
        alpha0.sizes() == at::IntArrayRef({panels}),
        "alpha0 shape mismatch");
    TORCH_CHECK(boundary_d.sizes() == boundary_j.sizes(), "boundary_d shape mismatch");
    TORCH_CHECK(
        u.sizes() == at::IntArrayRef({panels, kChunk, kRank})
            && h.sizes() == u.sizes() && grad_diagonal.sizes() == u.sizes(),
        "vector shape mismatch");
    TORCH_CHECK(
        log_decay.sizes() == at::IntArrayRef({panels, kChunk})
            && inverse_mass.sizes() == log_decay.sizes(),
        "prefix scalar shape mismatch");
    TORCH_CHECK(
        norm_sq.sizes() == at::IntArrayRef({panels, kChunk, kComponents})
            && coefficient.sizes() == norm_sq.sizes(),
        "radial scalar shape mismatch");
    TORCH_CHECK(
        upper_left.sizes()
                == at::IntArrayRef({panels, kChunk, kDescriptors, kRank})
            && upper_right.sizes() == upper_left.sizes()
            && lower_left.sizes() == upper_left.sizes()
            && lower_right.sizes() == upper_left.sizes(),
        "descriptor shape mismatch");
    TORCH_CHECK(
        heads > 0 && chunks > 0 && length > 0
            && strength.numel() == heads,
        "invalid heads/chunks/length/strength");
    TORCH_CHECK(
        chunks == (length + kChunk - 1) / kChunk,
        "chunks must equal ceil(length/32)");
    TORCH_CHECK(panels % (heads * chunks) == 0,
                "panels must be divisible by heads*chunks");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
panel_frame32_radial_vjp_cuda(
    const at::Tensor& boundary_m,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& log_decay,
    const at::Tensor& strength,
    const at::Tensor& alpha0,
    const at::Tensor& inverse_mass,
    const at::Tensor& coefficient,
    const at::Tensor& norm_sq,
    const at::Tensor& upper_left,
    const at::Tensor& upper_right,
    const at::Tensor& lower_left,
    const at::Tensor& lower_right,
    const at::Tensor& grad_diagonal,
    int64_t heads_value,
    int64_t chunks_value,
    int64_t length_value) {
    check_inputs(
        boundary_m, boundary_j, boundary_d, u, h, log_decay,
        strength, alpha0, inverse_mass, coefficient, norm_sq,
        upper_left, upper_right, lower_left, lower_right, grad_diagonal,
        heads_value, chunks_value, length_value);
    const int panels = static_cast<int>(boundary_j.size(0));
    const int heads = static_cast<int>(heads_value);
    const int chunks = static_cast<int>(chunks_value);
    const int length = static_cast<int>(length_value);
    const int batches = panels / (heads * chunks);
    auto partial_coefficient = at::empty(
        {panels, kMatrixTiles, kChunk, kComponents}, boundary_m.options());
    auto grad_coefficient = at::empty_like(coefficient);
    auto eta = at::empty_like(norm_sq);
    auto coefficient_strength = at::empty({panels}, boundary_m.options());
    auto grad_boundary_j = at::empty_like(boundary_j);
    auto grad_boundary_d = at::empty_like(boundary_d);
    auto grad_u = at::empty_like(u);
    auto grad_h = at::empty_like(h);
    auto partial_options = u.options();
    auto partial_row_u = at::empty(
        {panels, kChunk, kMatrixTiles, kTile}, partial_options);
    auto partial_col_u = at::empty_like(partial_row_u);
    auto partial_col_h = at::empty_like(partial_row_u);
    auto partial_scalar = at::empty(
        {panels, kMatrixTiles, kScalarSlots}, partial_options);
    auto partial_scalar_lo = at::empty_like(partial_scalar);
    auto panel_strength = at::empty({panels}, boundary_m.options());
    auto grad_boundary_m = at::empty_like(boundary_m);
    auto grad_log_decay = at::empty_like(log_decay);
    auto grad_strength = at::empty_like(strength);

    c10::cuda::CUDAGuard guard(boundary_j.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    constexpr size_t descriptor_shared_bytes =
        sizeof(float) * descriptor_shared_floats();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        descriptor_contraction_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(descriptor_shared_bytes)));
    descriptor_contraction_kernel<<<
        dim3(panels, kMatrixTiles), kEntryThreads,
        descriptor_shared_bytes, stream>>>(
        boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), alpha0.data_ptr<float>(),
        inverse_mass.data_ptr<float>(), upper_left.data_ptr<float>(),
        upper_right.data_ptr<float>(), lower_left.data_ptr<float>(),
        lower_right.data_ptr<float>(), partial_coefficient.data_ptr<float>(),
        chunks, length, panels);
    constexpr int reduce_threads = 256;
    const int coefficient_items = panels * kChunk * kComponents;
    coefficient_partial_reduce_kernel<<<
        (coefficient_items + reduce_threads - 1) / reduce_threads,
        reduce_threads, 0, stream>>>(
        partial_coefficient.data_ptr<float>(),
        grad_coefficient.data_ptr<float>(), coefficient_items);
    radial_scalars_kernel<<<panels, 128, 0, stream>>>(
        strength.data_ptr<float>(), norm_sq.data_ptr<float>(),
        grad_coefficient.data_ptr<float>(), eta.data_ptr<float>(),
        coefficient_strength.data_ptr<float>(), heads, chunks, length, panels);
    constexpr size_t shared_bytes =
        sizeof(float) * entry_shared_floats();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        radial_entry_vjp_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    radial_entry_vjp_kernel<<<
        dim3(panels, kMatrixTiles), kEntryThreads, shared_bytes, stream>>>(
        boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), strength.data_ptr<float>(),
        alpha0.data_ptr<float>(), inverse_mass.data_ptr<float>(),
        coefficient.data_ptr<float>(), eta.data_ptr<float>(),
        upper_left.data_ptr<float>(), upper_right.data_ptr<float>(),
        lower_left.data_ptr<float>(), lower_right.data_ptr<float>(),
        grad_diagonal.data_ptr<float>(), grad_boundary_j.data_ptr<float>(),
        grad_boundary_d.data_ptr<float>(), partial_row_u.data_ptr<float>(),
        partial_col_u.data_ptr<float>(), partial_col_h.data_ptr<float>(),
        partial_scalar.data_ptr<float>(), partial_scalar_lo.data_ptr<float>(),
        heads, chunks, length, panels);
    const int vector_items = panels * kChunk * kRank;
    vector_partial_reduce_kernel<<<
        (vector_items + reduce_threads - 1) / reduce_threads,
        reduce_threads, 0, stream>>>(
        partial_row_u.data_ptr<float>(), partial_col_u.data_ptr<float>(),
        partial_col_h.data_ptr<float>(), grad_u.data_ptr<float>(),
        grad_h.data_ptr<float>(), vector_items);
    scalar_partial_reduce_kernel<<<panels, 64, 0, stream>>>(
        partial_scalar.data_ptr<float>(), partial_scalar_lo.data_ptr<float>(),
        coefficient_strength.data_ptr<float>(),
        grad_boundary_m.data_ptr<float>(), grad_log_decay.data_ptr<float>(),
        panel_strength.data_ptr<float>(), panels);
    strength_reduce_kernel<<<heads, 64, 0, stream>>>(
        panel_strength.data_ptr<float>(), grad_strength.data_ptr<float>(),
        heads, chunks, batches);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_boundary_m, grad_boundary_j, grad_boundary_d,
        grad_u, grad_h, grad_log_decay, grad_strength};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "panel_frame32_radial_vjp128(Tensor boundary_m, Tensor boundary_j, "
        "Tensor boundary_d, Tensor u, Tensor h, Tensor log_decay, "
        "Tensor strength, Tensor alpha0, Tensor inverse_mass, "
        "Tensor coefficient, Tensor norm_sq, Tensor upper_left, "
        "Tensor upper_right, Tensor lower_left, Tensor lower_right, "
        "Tensor grad_diagonal, int heads, int chunks, int length) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl(
        "panel_frame32_radial_vjp128",
        &panel_frame32_radial_vjp_cuda);
}
