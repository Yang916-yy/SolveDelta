#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <tuple>

namespace {

constexpr int kRank = 128;
constexpr int kChunk = 16;
constexpr int kPanel = 5;
constexpr int kTile = 16;
constexpr int kTiles = kRank / kTile;
constexpr int kTileCount = kTiles * kTiles;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

struct FloatFloat {
    float hi;
    float lo;
};

__device__ __forceinline__ FloatFloat ff_add(
    FloatFloat left, FloatFloat right) {
    const float sum = __fadd_rn(left.hi, right.hi);
    const float virtual_right = __fsub_rn(sum, left.hi);
    const float sum_error = __fadd_rn(
        __fsub_rn(left.hi, __fsub_rn(sum, virtual_right)),
        __fsub_rn(right.hi, virtual_right));
    const float correction = __fadd_rn(
        sum_error, __fadd_rn(left.lo, right.lo));
    const float high = __fadd_rn(sum, correction);
    return {high, __fsub_rn(correction, __fsub_rn(high, sum))};
}

__device__ __forceinline__ FloatFloat ff_product(float left, float right) {
    const float product = __fmul_rn(left, right);
    return {product, __fmaf_rn(left, right, -product)};
}

__device__ __forceinline__ FloatFloat ff_multiply(
    FloatFloat value, float scale) {
    return ff_add(
        ff_product(value.hi, scale), ff_product(value.lo, scale));
}

__device__ __forceinline__ FloatFloat ff_add_product(
    FloatFloat value, float left, float right) {
    return ff_add(value, ff_product(left, right));
}

__device__ __forceinline__ FloatFloat ff_warp_sum(FloatFloat value) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        const FloatFloat other{
            __shfl_down_sync(0xffffffffu, value.hi, offset),
            __shfl_down_sync(0xffffffffu, value.lo, offset)};
        if (lane + offset < 32) {
            value = ff_add(value, other);
        }
    }
    return value;
}

__device__ __forceinline__ FloatFloat ff_shuffle_up(
    FloatFloat value, int offset) {
    return {
        __shfl_up_sync(0xffffffffu, value.hi, offset),
        __shfl_up_sync(0xffffffffu, value.lo, offset)};
}

__device__ __forceinline__ FloatFloat ff_shuffle_down(
    FloatFloat value, int offset) {
    return {
        __shfl_down_sync(0xffffffffu, value.hi, offset),
        __shfl_down_sync(0xffffffffu, value.lo, offset)};
}

__device__ __forceinline__ FloatFloat ff_shuffle(
    FloatFloat value, int lane) {
    return {
        __shfl_sync(0xffffffffu, value.hi, lane),
        __shfl_sync(0xffffffffu, value.lo, lane)};
}

__device__ __forceinline__ FloatFloat ff_lazy_add(
    FloatFloat left, FloatFloat right) {
    const float sum = __fadd_rn(left.hi, right.hi);
    const float virtual_right = __fsub_rn(sum, left.hi);
    const float error = __fadd_rn(
        __fsub_rn(left.hi, __fsub_rn(sum, virtual_right)),
        __fsub_rn(right.hi, virtual_right));
    return {sum, __fadd_rn(error, __fadd_rn(left.lo, right.lo))};
}

template <bool Upper>
__device__ __forceinline__ FloatFloat ff_lazy_exclusive(
    FloatFloat value, FloatFloat carry, FloatFloat& block_total) {
    const int lane = threadIdx.x & 31;
    FloatFloat total = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const FloatFloat other = Upper
            ? ff_shuffle_down(total, offset)
            : ff_shuffle_up(total, offset);
        if (Upper ? lane + offset < 32 : lane >= offset) {
            total = Upper
                ? ff_lazy_add(total, other)
                : ff_lazy_add(other, total);
        }
    }
    const FloatFloat neighbor = Upper
        ? ff_shuffle_down(total, 1)
        : ff_shuffle_up(total, 1);
    block_total = ff_shuffle(total, Upper ? 0 : 31);
    if (Upper ? lane == 31 : lane == 0) {
        return carry;
    }
    return Upper
        ? ff_lazy_add(neighbor, carry)
        : ff_lazy_add(carry, neighbor);
}

__device__ __forceinline__ float ff_value(FloatFloat value) {
    return __fadd_rn(value.hi, value.lo);
}

__device__ __forceinline__ float warp_prefix_exclusive(float value, float carry) {
    const int lane = threadIdx.x & 31;
    float total = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float other = __shfl_up_sync(0xffffffffu, total, offset);
        if (lane >= offset) {
            total += other;
        }
    }
    return carry + total - value;
}

__device__ __forceinline__ float warp_suffix_exclusive(float value, float carry) {
    const int lane = threadIdx.x & 31;
    float total = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float other = __shfl_down_sync(0xffffffffu, total, offset);
        if (lane + offset < 32) {
            total += other;
        }
    }
    return carry + total - value;
}

template <bool Upper>
__device__ __forceinline__ void panel_actions(
    const float* __restrict__ left,
    const float* __restrict__ right,
    const float* __restrict__ source_u,
    const float* __restrict__ source_h,
    float (&action_u)[4],
    float (&action_h)[4],
    float (&action_h_low)[4],
    float (&transpose_u)[4]) {
    const int lane = threadIdx.x & 31;
    float carry_bu[kPanel] = {};
    FloatFloat carry_bh[kPanel] = {};

#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int block = Upper ? 3 - pass : pass;
        const int coordinate = block * 32 + lane;
        const float uv = source_u[coordinate];
        const float hv = source_h[coordinate];
        float result_u = 0.0f;
        FloatFloat result_h{0.0f, 0.0f};
#pragma unroll
        for (int k = 0; k < kPanel; ++k) {
            const float av = left[coordinate * kPanel + k];
            const float bv = right[coordinate * kPanel + k];
            const float bu = bv * uv;
            const FloatFloat bh = ff_product(bv, hv);
            const float exclusive_u = Upper
                ? warp_suffix_exclusive(bu, carry_bu[k])
                : warp_prefix_exclusive(bu, carry_bu[k]);
            FloatFloat block_total;
            const FloatFloat exclusive_h = Upper
                ? ff_lazy_exclusive<true>(bh, carry_bh[k], block_total)
                : ff_lazy_exclusive<false>(bh, carry_bh[k], block_total);
            result_u = fmaf(av, exclusive_u, result_u);
            result_h = ff_add(result_h, ff_multiply(exclusive_h, av));
            const int terminal = Upper ? 0 : 31;
            const float inclusive = exclusive_u - carry_bu[k] + bu;
            carry_bu[k] += __shfl_sync(0xffffffffu, inclusive, terminal);
            carry_bh[k] = ff_add(carry_bh[k], block_total);
        }
        action_u[block] = result_u;
        action_h[block] = result_h.hi;
        action_h_low[block] = result_h.lo;
    }

    float carry_au[kPanel] = {};
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int block = Upper ? pass : 3 - pass;
        const int coordinate = block * 32 + lane;
        const float uv = source_u[coordinate];
        float result = 0.0f;
#pragma unroll
        for (int k = 0; k < kPanel; ++k) {
            const float av = left[coordinate * kPanel + k];
            const float bv = right[coordinate * kPanel + k];
            const float au = av * uv;
            const float exclusive = Upper
                ? warp_prefix_exclusive(au, carry_au[k])
                : warp_suffix_exclusive(au, carry_au[k]);
            result = fmaf(bv, exclusive, result);
            const int terminal = Upper ? 31 : 0;
            const float inclusive = exclusive - carry_au[k] + au;
            carry_au[k] += __shfl_sync(0xffffffffu, inclusive, terminal);
        }
        transpose_u[block] = result;
    }
}

__launch_bounds__(kThreads, 2)
__global__ void boundary_tile_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ lower_left,
    const float* __restrict__ lower_right,
    const float* __restrict__ upper_left,
    const float* __restrict__ upper_right,
    float* __restrict__ grad_boundary_h,
    float* __restrict__ grad_boundary_r,
    float* __restrict__ boundary_partial,
    const int panels) {
    __shared__ float panel_cache[4][kTile][kPanel];
    __shared__ float reduction[4][kChunk][kWarps];
    const int tile = blockIdx.x;
    const int panel = blockIdx.y;
    if (panel >= panels) {
        return;
    }
    const int row_tile = tile / kTiles;
    const int col_tile = tile % kTiles;
    const int local_row = threadIdx.x / kTile;
    const int local_col = threadIdx.x % kTile;
    const int row = row_tile * kTile + local_row;
    const int col = col_tile * kTile + local_col;
    const int entry = row * kRank + col;
    const int matrix_base = panel * kRank * kRank;
    const float bh = boundary_h[matrix_base + entry];
    const float br = boundary_r[matrix_base + entry];
    float grad_h = 0.0f;
    float grad_r = 0.0f;
    float local_partial[4][kChunk] = {};

#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        const int panel_base = (panel * kChunk + target) * kRank * kPanel;
        for (int item = threadIdx.x; item < 4 * kTile * kPanel; item += kThreads) {
            const int which = item / (kTile * kPanel);
            const int remainder = item % (kTile * kPanel);
            const int coordinate = remainder / kPanel;
            const int k = remainder % kPanel;
            const int global_coordinate = (which == 0 || which == 2)
                ? row_tile * kTile + coordinate
                : col_tile * kTile + coordinate;
            const float* source = which == 0 ? lower_left
                : which == 1 ? lower_right
                : which == 2 ? upper_left : upper_right;
            panel_cache[which][coordinate][k] =
                source[panel_base + global_coordinate * kPanel + k];
        }
        __syncthreads();
        float packet = 0.0f;
        int side = -1;
        if (row > col) {
            side = 0;
#pragma unroll
            for (int k = 0; k < kPanel; ++k) {
                packet = fmaf(
                    panel_cache[0][local_row][k],
                    panel_cache[1][local_col][k], packet);
            }
        } else if (row < col) {
            side = 1;
#pragma unroll
            for (int k = 0; k < kPanel; ++k) {
                packet = fmaf(
                    panel_cache[2][local_row][k],
                    panel_cache[3][local_col][k], packet);
            }
        }
        const int scalar = panel * kChunk + target;
        const int coeff_base = scalar * 4;
        if (side == 0) {
            const float scale = alpha[scalar];
            grad_h = fmaf(scale * coefficient[coeff_base + 0], packet, grad_h);
            grad_r = fmaf(scale * coefficient[coeff_base + 1], packet, grad_r);
            local_partial[0][target] = packet * bh;
            local_partial[1][target] = packet * br;
        } else if (side == 1) {
            const float scale = alpha[scalar];
            grad_h = fmaf(scale * coefficient[coeff_base + 2], packet, grad_h);
            grad_r = fmaf(scale * coefficient[coeff_base + 3], packet, grad_r);
            local_partial[2][target] = packet * bh;
            local_partial[3][target] = packet * br;
        }
        __syncthreads();
    }
    grad_boundary_h[matrix_base + entry] = grad_h;
    grad_boundary_r[matrix_base + entry] = grad_r;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int component = 0; component < 4; ++component) {
#pragma unroll
        for (int target = 0; target < kChunk; ++target) {
            const float sum = warp_sum(local_partial[component][target]);
            if (lane == 0) {
                reduction[component][target][warp] = sum;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < 4 * kChunk) {
        const int component = threadIdx.x / kChunk;
        const int target = threadIdx.x % kChunk;
        float sum = 0.0f;
#pragma unroll
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            sum += reduction[component][target][warp_index];
        }
        boundary_partial[
            ((panel * kTileCount + tile) * kChunk + target) * 4 + component
        ] = sum;
    }
}

__launch_bounds__(512, 1)
__global__ void boundary_panel_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ lower_left,
    const float* __restrict__ lower_right,
    const float* __restrict__ upper_left,
    const float* __restrict__ upper_right,
    float* __restrict__ grad_boundary_h,
    float* __restrict__ grad_boundary_r,
    float* __restrict__ boundary_contraction,
    const int panels) {
    __shared__ float panel_cache[4][kRank][kPanel];
    __shared__ float reduction[4][16];
    const int panel = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (panel >= panels) {
        return;
    }
    constexpr int kEntriesPerThread = kRank * kRank / 512;
    float grad_h[kEntriesPerThread] = {};
    float grad_r[kEntriesPerThread] = {};

#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        const int panel_base = (panel * kChunk + target) * kRank * kPanel;
        for (int item = threadIdx.x; item < 4 * kRank * kPanel; item += 512) {
            const int which = item / (kRank * kPanel);
            const int remainder = item % (kRank * kPanel);
            const float* source = which == 0 ? lower_left
                : which == 1 ? lower_right
                : which == 2 ? upper_left : upper_right;
            panel_cache[which][remainder / kPanel][remainder % kPanel] =
                source[panel_base + remainder];
        }
        __syncthreads();
        float contractions[4] = {};
#pragma unroll
        for (int item = 0; item < kEntriesPerThread; ++item) {
            const int entry = threadIdx.x + item * 512;
            const int row = entry / kRank;
            const int col = entry % kRank;
            float packet = 0.0f;
            int side = -1;
            if (row > col) {
                side = 0;
#pragma unroll
                for (int k = 0; k < kPanel; ++k) {
                    packet = fmaf(
                        panel_cache[0][row][k], panel_cache[1][col][k], packet);
                }
            } else if (row < col) {
                side = 1;
#pragma unroll
                for (int k = 0; k < kPanel; ++k) {
                    packet = fmaf(
                        panel_cache[2][row][k], panel_cache[3][col][k], packet);
                }
            }
            const int scalar = panel * kChunk + target;
            const int coeff_base = scalar * 4;
            const int matrix_index = panel * kRank * kRank + entry;
            if (side == 0) {
                const float scale = alpha[scalar];
                grad_h[item] = fmaf(
                    scale * coefficient[coeff_base + 0], packet, grad_h[item]);
                grad_r[item] = fmaf(
                    scale * coefficient[coeff_base + 1], packet, grad_r[item]);
                contractions[0] = fmaf(packet, boundary_h[matrix_index], contractions[0]);
                contractions[1] = fmaf(packet, boundary_r[matrix_index], contractions[1]);
            } else if (side == 1) {
                const float scale = alpha[scalar];
                grad_h[item] = fmaf(
                    scale * coefficient[coeff_base + 2], packet, grad_h[item]);
                grad_r[item] = fmaf(
                    scale * coefficient[coeff_base + 3], packet, grad_r[item]);
                contractions[2] = fmaf(packet, boundary_h[matrix_index], contractions[2]);
                contractions[3] = fmaf(packet, boundary_r[matrix_index], contractions[3]);
            }
        }
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            const float sum = warp_sum(contractions[component]);
            if (lane == 0) {
                reduction[component][warp] = sum;
            }
        }
        __syncthreads();
        if (threadIdx.x < 4) {
            float sum = 0.0f;
#pragma unroll
            for (int warp_index = 0; warp_index < 16; ++warp_index) {
                sum += reduction[threadIdx.x][warp_index];
            }
            boundary_contraction[(panel * kChunk + target) * 4 + threadIdx.x] = sum;
        }
        __syncthreads();
    }

#pragma unroll
    for (int item = 0; item < kEntriesPerThread; ++item) {
        const int entry = threadIdx.x + item * 512;
        const int matrix_index = panel * kRank * kRank + entry;
        grad_boundary_h[matrix_index] = grad_h[item];
        grad_boundary_r[matrix_index] = grad_r[item];
    }
}

__launch_bounds__(256, 2)
__global__ void boundary_rowblock_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ lower_left,
    const float* __restrict__ lower_right,
    const float* __restrict__ upper_left,
    const float* __restrict__ upper_right,
    float* __restrict__ grad_boundary_h,
    float* __restrict__ grad_boundary_r,
    float* __restrict__ boundary_partial,
    const int panels) {
    __shared__ float left_cache[2][32][kPanel];
    __shared__ float right_cache[2][kRank][kPanel];
    __shared__ float reduction[4][8];
    const int row_block = blockIdx.x;
    const int panel = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (panel >= panels) {
        return;
    }
    constexpr int kEntriesPerThread = 32 * kRank / 256;
    float grad_h[kEntriesPerThread] = {};
    float grad_r[kEntriesPerThread] = {};

#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        const int panel_base = (panel * kChunk + target) * kRank * kPanel;
        for (int item = threadIdx.x; item < 2 * (32 + kRank) * kPanel; item += 256) {
            const int side = item / ((32 + kRank) * kPanel);
            const int side_item = item % ((32 + kRank) * kPanel);
            if (side_item < 32 * kPanel) {
                const int coordinate = side_item / kPanel;
                const int k = side_item % kPanel;
                const float* source = side == 0 ? lower_left : upper_left;
                left_cache[side][coordinate][k] = source[
                    panel_base + (row_block * 32 + coordinate) * kPanel + k];
            } else {
                const int right_item = side_item - 32 * kPanel;
                const int coordinate = right_item / kPanel;
                const int k = right_item % kPanel;
                const float* source = side == 0 ? lower_right : upper_right;
                right_cache[side][coordinate][k] = source[
                    panel_base + coordinate * kPanel + k];
            }
        }
        __syncthreads();
        float contractions[4] = {};
#pragma unroll
        for (int item = 0; item < kEntriesPerThread; ++item) {
            const int local_entry = threadIdx.x + item * 256;
            const int local_row = local_entry / kRank;
            const int row = row_block * 32 + local_row;
            const int col = local_entry % kRank;
            float packet = 0.0f;
            int side = -1;
            if (row > col) {
                side = 0;
#pragma unroll
                for (int k = 0; k < kPanel; ++k) {
                    packet = fmaf(
                        left_cache[0][local_row][k], right_cache[0][col][k], packet);
                }
            } else if (row < col) {
                side = 1;
#pragma unroll
                for (int k = 0; k < kPanel; ++k) {
                    packet = fmaf(
                        left_cache[1][local_row][k], right_cache[1][col][k], packet);
                }
            }
            const int scalar = panel * kChunk + target;
            const int coeff_base = scalar * 4;
            const int entry = row * kRank + col;
            const int matrix_index = panel * kRank * kRank + entry;
            if (side == 0) {
                const float scale = alpha[scalar];
                grad_h[item] = fmaf(
                    scale * coefficient[coeff_base + 0], packet, grad_h[item]);
                grad_r[item] = fmaf(
                    scale * coefficient[coeff_base + 1], packet, grad_r[item]);
                contractions[0] = fmaf(packet, boundary_h[matrix_index], contractions[0]);
                contractions[1] = fmaf(packet, boundary_r[matrix_index], contractions[1]);
            } else if (side == 1) {
                const float scale = alpha[scalar];
                grad_h[item] = fmaf(
                    scale * coefficient[coeff_base + 2], packet, grad_h[item]);
                grad_r[item] = fmaf(
                    scale * coefficient[coeff_base + 3], packet, grad_r[item]);
                contractions[2] = fmaf(packet, boundary_h[matrix_index], contractions[2]);
                contractions[3] = fmaf(packet, boundary_r[matrix_index], contractions[3]);
            }
        }
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            const float sum = warp_sum(contractions[component]);
            if (lane == 0) {
                reduction[component][warp] = sum;
            }
        }
        __syncthreads();
        if (threadIdx.x < 4) {
            float sum = 0.0f;
#pragma unroll
            for (int warp_index = 0; warp_index < 8; ++warp_index) {
                sum += reduction[threadIdx.x][warp_index];
            }
            boundary_partial[
                ((panel * 4 + row_block) * kChunk + target) * 4 + threadIdx.x
            ] = sum;
        }
        __syncthreads();
    }

#pragma unroll
    for (int item = 0; item < kEntriesPerThread; ++item) {
        const int local_entry = threadIdx.x + item * 256;
        const int row = row_block * 32 + local_entry / kRank;
        const int col = local_entry % kRank;
        const int matrix_index = panel * kRank * kRank + row * kRank + col;
        grad_boundary_h[matrix_index] = grad_h[item];
        grad_boundary_r[matrix_index] = grad_r[item];
    }
}

__launch_bounds__(kThreads, 2)
__global__ void boundary_gemm80_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ lower_left,
    const float* __restrict__ lower_right,
    const float* __restrict__ upper_left,
    const float* __restrict__ upper_right,
    float* __restrict__ grad_boundary_h,
    float* __restrict__ grad_boundary_r,
    const int panels) {
    __shared__ float shared_h[kTile][kTile];
    __shared__ float shared_r[kTile][kTile];
    __shared__ float shared_b[kTile][kTile];
    const int tile = blockIdx.x;
    const int panel = blockIdx.y;
    const int row_tile = tile / kTiles;
    const int col_tile = tile % kTiles;
    const int local_row = threadIdx.x / kTile;
    const int local_col = threadIdx.x % kTile;
    const int row = row_tile * kTile + local_row;
    const int col = col_tile * kTile + local_col;
    if (panel >= panels) {
        return;
    }
    float result_h = 0.0f;
    float result_r = 0.0f;
    const int first_side = row_tile < col_tile ? 1 : 0;
    const int side_count = row_tile == col_tile ? 2 : 1;
    for (int side_index = 0; side_index < side_count; ++side_index) {
        const int side = first_side + side_index;
        float side_h = 0.0f;
        float side_r = 0.0f;
#pragma unroll
        for (int k_block = 0; k_block < 5; ++k_block) {
            const int packed_k = k_block * kTile + local_col;
            const int target = packed_k / kPanel;
            const int k = packed_k % kPanel;
            const int panel_base = (panel * kChunk + target) * kRank * kPanel;
            const int scalar = panel * kChunk + target;
            const int coeff_base = scalar * 4;
            const float* left = side == 0 ? lower_left : upper_left;
            const float* right = side == 0 ? lower_right : upper_right;
            const int component_base = side == 0 ? 0 : 2;
            const float common = alpha[scalar]
                * left[panel_base + row * kPanel + k];
            shared_h[local_row][local_col] =
                common * coefficient[coeff_base + component_base + 0];
            shared_r[local_row][local_col] =
                common * coefficient[coeff_base + component_base + 1];
            const int b_packed_k = k_block * kTile + local_row;
            const int b_target = b_packed_k / kPanel;
            const int b_k = b_packed_k % kPanel;
            const int b_base = (panel * kChunk + b_target) * kRank * kPanel;
            shared_b[local_row][local_col] =
                right[b_base + col * kPanel + b_k];
            __syncthreads();
#pragma unroll
            for (int inner = 0; inner < kTile; ++inner) {
                side_h = fmaf(
                    shared_h[local_row][inner], shared_b[inner][local_col], side_h);
                side_r = fmaf(
                    shared_r[local_row][inner], shared_b[inner][local_col], side_r);
            }
            __syncthreads();
        }
        if ((side == 0 && row > col) || (side == 1 && row < col)) {
            result_h = side_h;
            result_r = side_r;
        }
    }
    const int index = panel * kRank * kRank + row * kRank + col;
    grad_boundary_h[index] = result_h;
    grad_boundary_r[index] = result_r;
}

__launch_bounds__(kThreads, 2)
__global__ void boundary_contract_kernel(
    const float* __restrict__ boundary_h,
    const float* __restrict__ boundary_r,
    const float* __restrict__ lower_left,
    const float* __restrict__ lower_right,
    const float* __restrict__ upper_left,
    const float* __restrict__ upper_right,
    float* __restrict__ boundary_contraction,
    const int panels) {
    __shared__ float descriptor[4][kRank][kPanel];
    __shared__ float reduction_hi[4][kWarps];
    __shared__ float reduction_lo[4][kWarps];
    const int target = blockIdx.x;
    const int panel = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (panel >= panels) {
        return;
    }
    const int panel_base = (panel * kChunk + target) * kRank * kPanel;
    const int matrix_base = panel * kRank * kRank;
    for (int item = threadIdx.x; item < 4 * kRank * kPanel;
         item += kThreads) {
        const int which = item / (kRank * kPanel);
        const int remainder = item % (kRank * kPanel);
        const float* source = which == 0 ? lower_left
            : which == 1 ? lower_right
            : which == 2 ? upper_left : upper_right;
        descriptor[which][remainder / kPanel][remainder % kPanel] =
            source[panel_base + remainder];
    }
    __syncthreads();

    FloatFloat contraction[4] = {};
    for (int entry = threadIdx.x; entry < kRank * kRank; entry += kThreads) {
        const int row = entry / kRank;
        const int col = entry % kRank;
        FloatFloat packet{0.0f, 0.0f};
        int side = -1;
        if (row > col) {
            side = 0;
#pragma unroll
            for (int k = 0; k < kPanel; ++k) {
                packet = ff_lazy_add(
                    packet,
                    ff_product(
                        descriptor[0][row][k], descriptor[1][col][k]));
            }
        } else if (row < col) {
            side = 1;
#pragma unroll
            for (int k = 0; k < kPanel; ++k) {
                packet = ff_lazy_add(
                    packet,
                    ff_product(
                        descriptor[2][row][k], descriptor[3][col][k]));
            }
        }
        if (side == 0) {
            contraction[0] = ff_add(
                contraction[0],
                ff_multiply(packet, boundary_h[matrix_base + entry]));
            contraction[1] = ff_add(
                contraction[1],
                ff_multiply(packet, boundary_r[matrix_base + entry]));
        } else if (side == 1) {
            contraction[2] = ff_add(
                contraction[2],
                ff_multiply(packet, boundary_h[matrix_base + entry]));
            contraction[3] = ff_add(
                contraction[3],
                ff_multiply(packet, boundary_r[matrix_base + entry]));
        }
    }
#pragma unroll
    for (int component = 0; component < 4; ++component) {
        const FloatFloat sum = ff_warp_sum(contraction[component]);
        if (lane == 0) {
            reduction_hi[component][warp] = sum.hi;
            reduction_lo[component][warp] = sum.lo;
        }
    }
    __syncthreads();
    if (threadIdx.x < 4) {
        FloatFloat sum{0.0f, 0.0f};
#pragma unroll
        for (int warp_index = 0; warp_index < kWarps; ++warp_index) {
            sum = ff_add(sum, {
                reduction_hi[threadIdx.x][warp_index],
                reduction_lo[threadIdx.x][warp_index]});
        }
        const int output =
            ((panel * kChunk + target) * 4 + threadIdx.x) * 2;
        boundary_contraction[output + 0] = sum.hi;
        boundary_contraction[output + 1] = sum.lo;
    }
}

__launch_bounds__(512, 1)
__global__ void local_source_kernel(
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ coefficient,
    const float* __restrict__ lower_left,
    const float* __restrict__ lower_right,
    const float* __restrict__ upper_left,
    const float* __restrict__ upper_right,
    float* __restrict__ grad_u,
    float* __restrict__ grad_h,
    float* __restrict__ grad_weights,
    float* __restrict__ local_contraction,
    const int panels) {
    __shared__ float local_hi[kChunk][kChunk][4];
    __shared__ float local_lo[kChunk][kChunk][4];
    const int panel = blockIdx.x;
    const int source = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (panel >= panels) {
        return;
    }
    const int source_base = (panel * kChunk + source) * kRank;
    float source_u[4];
    float gu[4] = {};
    float gh[4] = {};
#pragma unroll
    for (int block = 0; block < 4; ++block) {
        const int coordinate = block * 32 + lane;
        source_u[block] = u[source_base + coordinate];
    }

    for (int target = source; target < kChunk; ++target) {
        const int target_base = (panel * kChunk + target) * kRank * kPanel;
        float lower_u[4], lower_h[4], lower_h_low[4], lower_tu[4];
        float upper_u[4], upper_h[4], upper_h_low[4], upper_tu[4];
        panel_actions<false>(
            lower_left + target_base, lower_right + target_base,
            u + source_base, h + source_base,
            lower_u, lower_h, lower_h_low, lower_tu);
        panel_actions<true>(
            upper_left + target_base, upper_right + target_base,
            u + source_base, h + source_base,
            upper_u, upper_h, upper_h_low, upper_tu);
        FloatFloat local[4] = {};
#pragma unroll
        for (int block = 0; block < 4; ++block) {
            local[0] = ff_add_product(local[0], source_u[block], lower_u[block]);
            local[1] = ff_add(
                local[1],
                ff_multiply(
                    {lower_h[block], lower_h_low[block]}, source_u[block]));
            local[2] = ff_add_product(local[2], source_u[block], upper_u[block]);
            local[3] = ff_add(
                local[3],
                ff_multiply(
                    {upper_h[block], upper_h_low[block]}, source_u[block]));
        }
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            local[component] = ff_warp_sum(local[component]);
        }
        const int packet = (panel * kChunk + source) * kChunk + target;
        const float weight = weights[packet];
        const int coeff_base = (panel * kChunk + target) * 4;
        const float phl = coefficient[coeff_base + 0];
        const float prl = coefficient[coeff_base + 1];
        const float phu = coefficient[coeff_base + 2];
        const float pru = coefficient[coeff_base + 3];
#pragma unroll
        for (int block = 0; block < 4; ++block) {
            gu[block] += weight * (
                phl * (lower_u[block] + lower_tu[block])
                + phu * (upper_u[block] + upper_tu[block])
                + prl * ff_value({lower_h[block], lower_h_low[block]})
                + pru * ff_value({upper_h[block], upper_h_low[block]}));
            gh[block] += weight * (
                prl * lower_tu[block] + pru * upper_tu[block]);
        }
        if (lane == 0) {
            FloatFloat grad_weight{0.0f, 0.0f};
            grad_weight = ff_add(grad_weight, ff_multiply(local[0], phl));
            grad_weight = ff_add(grad_weight, ff_multiply(local[1], prl));
            grad_weight = ff_add(grad_weight, ff_multiply(local[2], phu));
            grad_weight = ff_add(grad_weight, ff_multiply(local[3], pru));
            grad_weights[packet] = ff_value(grad_weight);
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                local_hi[source][target][component] = local[component].hi;
                local_lo[source][target][component] = local[component].lo;
            }
        }
    }
#pragma unroll
    for (int block = 0; block < 4; ++block) {
        const int coordinate = block * 32 + lane;
        grad_u[source_base + coordinate] = gu[block];
        grad_h[source_base + coordinate] = gh[block];
    }
    __syncthreads();
    if (threadIdx.x < kChunk) {
        const int target = threadIdx.x;
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            FloatFloat total{0.0f, 0.0f};
            for (int source_index = 0; source_index <= target; ++source_index) {
                const int packet =
                    (panel * kChunk + source_index) * kChunk + target;
                total = ff_add(total, ff_multiply(
                    {local_hi[source_index][target][component],
                     local_lo[source_index][target][component]},
                    weights[packet]));
            }
            const int output =
                ((panel * kChunk + target) * 4 + component) * 2;
            local_contraction[output + 0] = total.hi;
            local_contraction[output + 1] = total.lo;
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
blocked_rank5_cuda(
    const at::Tensor& boundary_h,
    const at::Tensor& boundary_r,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& weights,
    const at::Tensor& alpha,
    const at::Tensor& coefficient,
    const at::Tensor& lower_left,
    const at::Tensor& lower_right,
    const at::Tensor& upper_left,
    const at::Tensor& upper_right) {
    const int panels = static_cast<int>(u.size(0));
    auto grad_boundary_h = at::empty(
        {panels, kRank, kRank}, u.options());
    auto grad_boundary_r = at::empty_like(grad_boundary_h);
    auto grad_u = at::empty_like(u);
    auto grad_h = at::empty_like(h);
    auto grad_weights = at::zeros_like(weights);
    auto local_contraction = at::empty(
        {panels, kChunk, 4, 2}, u.options());
    auto boundary_contraction = at::empty(
        {panels, kChunk, 4, 2}, u.options());
    c10::cuda::CUDAGuard guard(u.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    dim3 gemm_grid(kTileCount, panels);
    boundary_gemm80_kernel<<<gemm_grid, kThreads, 0, stream>>>(
        alpha.data_ptr<float>(), coefficient.data_ptr<float>(),
        lower_left.data_ptr<float>(), lower_right.data_ptr<float>(),
        upper_left.data_ptr<float>(), upper_right.data_ptr<float>(),
        grad_boundary_h.data_ptr<float>(), grad_boundary_r.data_ptr<float>(),
        panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dim3 contract_grid(kChunk, panels);
    boundary_contract_kernel<<<contract_grid, kThreads, 0, stream>>>(
        boundary_h.data_ptr<float>(), boundary_r.data_ptr<float>(),
        lower_left.data_ptr<float>(), lower_right.data_ptr<float>(),
        upper_left.data_ptr<float>(), upper_right.data_ptr<float>(),
        boundary_contraction.data_ptr<float>(), panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    local_source_kernel<<<panels, 512, 0, stream>>>(
        u.data_ptr<float>(), h.data_ptr<float>(), weights.data_ptr<float>(),
        coefficient.data_ptr<float>(),
        lower_left.data_ptr<float>(), lower_right.data_ptr<float>(),
        upper_left.data_ptr<float>(), upper_right.data_ptr<float>(),
        grad_u.data_ptr<float>(), grad_h.data_ptr<float>(),
        grad_weights.data_ptr<float>(), local_contraction.data_ptr<float>(),
        panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_boundary_h, grad_boundary_r, grad_u, grad_h,
        grad_weights, local_contraction, boundary_contraction};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "packet_frame_rank5_vjp128(Tensor boundary_h, Tensor boundary_r, "
        "Tensor u, Tensor h, Tensor weights, "
        "Tensor alpha, Tensor coefficient, Tensor lower_left, "
        "Tensor lower_right, Tensor upper_left, Tensor upper_right) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("packet_frame_rank5_vjp128", &blocked_rank5_cuda);
}
