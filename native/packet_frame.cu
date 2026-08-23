#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

namespace {

constexpr unsigned kRank = 128;
constexpr unsigned kChunk = 16;
constexpr unsigned kThreads = 256;
constexpr unsigned kLocalThreads = 128;
constexpr unsigned kPacket = kChunk * kChunk;
constexpr unsigned kCoefficients = 4;
constexpr unsigned kDualRhs = 2;
constexpr unsigned kOmegaTile = 16;
constexpr unsigned kOmegaTiles = kRank / kOmegaTile;
constexpr unsigned kOmegaPairs = kOmegaTiles * (kOmegaTiles + 1) / 2;

__device__ __forceinline__ float reduce_group4(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 2, 4);
    value += __shfl_down_sync(0xffffffffu, value, 1, 4);
    return value;
}

__device__ __forceinline__ float reduce_group8(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 4, 8);
    value += __shfl_down_sync(0xffffffffu, value, 2, 8);
    value += __shfl_down_sync(0xffffffffu, value, 1, 8);
    return value;
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

__device__ __forceinline__ FloatFloat ff_negate(FloatFloat value) {
    return {-value.hi, -value.lo};
}

__device__ __forceinline__ FloatFloat ff_product(float left, float right) {
    const float product = __fmul_rn(left, right);
    return {product, __fmaf_rn(left, right, -product)};
}

__device__ __forceinline__ FloatFloat ff_multiply(
    FloatFloat value, float scale) {
    const FloatFloat leading = ff_product(value.hi, scale);
    return ff_add(leading, ff_product(value.lo, scale));
}

__device__ __forceinline__ FloatFloat ff_add_product(
    FloatFloat value, float left, float right) {
    return ff_add(value, ff_product(left, right));
}

__device__ __forceinline__ FloatFloat reduce_group8(FloatFloat value) {
#pragma unroll
    for (int offset = 4; offset; offset >>= 1) {
        const FloatFloat other{
            __shfl_down_sync(0xffffffffu, value.hi, offset, 8),
            __shfl_down_sync(0xffffffffu, value.lo, offset, 8)};
        value = ff_add(value, other);
    }
    return value;
}

__device__ __forceinline__ FloatFloat local_driven_omega_action(
    FloatFloat prefix_u,
    FloatFloat prefix_h,
    FloatFloat suffix_u,
    FloatFloat suffix_h,
    float u_value,
    float h_value,
    float r_lower,
    float r_upper,
    float weight) {
    const FloatFloat u_inner = ff_add(
        ff_multiply(prefix_h, r_lower), ff_multiply(suffix_h, r_upper));
    FloatFloat action = ff_multiply(u_inner, u_value);
    const FloatFloat h_inner = ff_add(
        ff_multiply(prefix_u, r_upper), ff_multiply(suffix_u, r_lower));
    action = ff_add(action, ff_negate(ff_multiply(h_inner, h_value)));
    return ff_multiply(action, weight);
}

struct TriangularShared {
    float* z_u;
    float* z_h;
    float* weights;
    float* solution;
    float* boundary_j_row;
    float* boundary_d_row;
    float* u_coordinate;
    float* h_coordinate;
    float* p;
    float* q;
    float* alpha;
};

__device__ __forceinline__ TriangularShared partition_triangular(float* storage) {
    TriangularShared shared{};
    shared.z_u = storage;
    storage += kPacket;
    shared.z_h = storage;
    storage += kPacket;
    shared.weights = storage;
    storage += kPacket;
    shared.solution = storage;
    storage += kChunk * kRank;
    shared.boundary_j_row = storage;
    storage += kRank;
    shared.boundary_d_row = storage;
    storage += kRank;
    shared.u_coordinate = storage;
    storage += kChunk;
    shared.h_coordinate = storage;
    storage += kChunk;
    shared.p = storage;
    storage += kChunk;
    shared.q = storage;
    storage += kChunk;
    shared.alpha = storage;
    return shared;
}

constexpr size_t triangular_shared_bytes() {
    return sizeof(float) * (
        3 * kPacket + kChunk * kRank + 2 * kRank + 5 * kChunk);
}

template<bool Upper>
__device__ __forceinline__ void packet_triangular_phase(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ rhs,
    float* __restrict__ output,
    TriangularShared shared,
    unsigned program) {
    const unsigned vector_base = program * kChunk * kRank;
    const unsigned boundary_base = program * kRank * kRank;

#pragma unroll 1
    for (unsigned step = 0; step < kRank; ++step) {
        const unsigned coordinate = Upper ? kRank - 1 - step : step;
        if (threadIdx.x < kRank) {
            const unsigned col = threadIdx.x;
            shared.boundary_j_row[col] =
                boundary_j[boundary_base + coordinate * kRank + col];
            shared.boundary_d_row[col] =
                boundary_d[boundary_base + coordinate * kRank + col];
        }
        if (threadIdx.x < kChunk) {
            const unsigned source = threadIdx.x;
            shared.u_coordinate[source] = u[vector_base + source * kRank + coordinate];
            shared.h_coordinate[source] = h[vector_base + source * kRank + coordinate];
        }
        __syncthreads();

        if (threadIdx.x < kChunk * 4) {
            const unsigned target = threadIdx.x >> 2;
            const unsigned lane = threadIdx.x & 3;
            const float p_t = shared.p[target];
            const float q_t = shared.q[target];
            float boundary_j_action = 0.0f;
            float boundary_d_action = 0.0f;
            if constexpr (Upper) {
                for (unsigned col = coordinate + 1 + lane; col < kRank; col += 4) {
                    const float solved = shared.solution[target * kRank + col];
                    boundary_j_action = fmaf(
                        shared.boundary_j_row[col], solved, boundary_j_action);
                    boundary_d_action = fmaf(
                        shared.boundary_d_row[col], solved, boundary_d_action);
                }
            } else {
                for (unsigned col = lane; col < coordinate; col += 4) {
                    const float solved = shared.solution[target * kRank + col];
                    boundary_j_action = fmaf(
                        shared.boundary_j_row[col], solved, boundary_j_action);
                    boundary_d_action = fmaf(
                        shared.boundary_d_row[col], solved, boundary_d_action);
                }
            }

            float local_action = 0.0f;
            for (unsigned source = lane; source <= target; source += 4) {
                const unsigned packet = source * kChunk + target;
                const float score = fmaf(
                    p_t, shared.z_u[packet], q_t * shared.z_h[packet]);
                local_action = fmaf(
                    shared.weights[packet] * shared.u_coordinate[source],
                    score,
                    local_action);
            }
            boundary_j_action = reduce_group4(boundary_j_action);
            boundary_d_action = reduce_group4(boundary_d_action);
            local_action = reduce_group4(local_action);
            const float boundary_action = shared.alpha[target] * fmaf(
                p_t, boundary_j_action, q_t * boundary_d_action);
            if (lane == 0) {
                const unsigned index = vector_base + target * kRank + coordinate;
                if constexpr (Upper) {
                    const float value = shared.solution[target * kRank + coordinate]
                        / diagonal[index];
                    const float solved = value - boundary_action - local_action;
                    shared.solution[target * kRank + coordinate] = solved;
                    output[index] = solved;
                } else {
                    const float solved = rhs[index] - boundary_action - local_action;
                    shared.solution[target * kRank + coordinate] = solved;
                }
            }
        }
        __syncthreads();

        for (unsigned packet = threadIdx.x; packet < kPacket; packet += kThreads) {
            const unsigned source = packet / kChunk;
            const unsigned target = packet % kChunk;
            if (source <= target) {
                const float solved = shared.solution[target * kRank + coordinate];
                shared.z_u[packet] = fmaf(
                    shared.u_coordinate[source], solved, shared.z_u[packet]);
                shared.z_h[packet] = fmaf(
                    shared.h_coordinate[source], solved, shared.z_h[packet]);
            }
        }
        __syncthreads();
    }
}

__launch_bounds__(kThreads, 1)
__global__ void packet_primal_kernel(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ rhs,
    float* __restrict__ output,
    float* __restrict__ lower_solution,
    unsigned programs) {
    extern __shared__ float shared_storage[];
    TriangularShared shared = partition_triangular(shared_storage);
    const unsigned program = blockIdx.x;
    if (program >= programs) {
        return;
    }
    const unsigned packet_base = program * kPacket;
    const unsigned scalar_base = program * kChunk;
    for (unsigned index = threadIdx.x; index < kPacket; index += kThreads) {
        shared.z_u[index] = 0.0f;
        shared.z_h[index] = 0.0f;
        shared.weights[index] = weights[packet_base + index];
    }
    if (threadIdx.x < kChunk) {
        const unsigned target = threadIdx.x;
        shared.p[target] = coefficient[(scalar_base + target) * kCoefficients + 0];
        shared.q[target] = coefficient[(scalar_base + target) * kCoefficients + 1];
        shared.alpha[target] = alpha[scalar_base + target];
    }
    __syncthreads();
    packet_triangular_phase<false>(
        boundary_j, boundary_d, u, h, weights, alpha, coefficient,
        diagonal, rhs, output, shared, program);

    for (unsigned index = threadIdx.x; index < kChunk * kRank; index += kThreads) {
        lower_solution[program * kChunk * kRank + index] = shared.solution[index];
    }

    for (unsigned index = threadIdx.x; index < 2 * kPacket; index += kThreads) {
        if (index < kPacket) {
            shared.z_u[index] = 0.0f;
        } else {
            shared.z_h[index - kPacket] = 0.0f;
        }
    }
    if (threadIdx.x < kChunk) {
        const unsigned target = threadIdx.x;
        shared.p[target] = coefficient[(scalar_base + target) * kCoefficients + 2];
        shared.q[target] = coefficient[(scalar_base + target) * kCoefficients + 3];
    }
    __syncthreads();
    packet_triangular_phase<true>(
        boundary_j, boundary_d, u, h, weights, alpha, coefficient,
        diagonal, rhs, output, shared, program);
}


__device__ __forceinline__ void decode_omega_pair(
    unsigned pair, unsigned& high, unsigned& low) {
    high = 0;
    while (pair > high) {
        pair -= high + 1;
        ++high;
    }
    low = pair;
}

// One unordered tile pair owns both skew-related output blocks.  Boundary
// J/D are therefore fetched once per packet and shared across all 16 RHS.
__launch_bounds__(kThreads, 2)
__global__ void packet_omega_pair_boundary_kernel(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ key,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    float* __restrict__ workspace,
    unsigned programs) {
    __shared__ float j_hl[kOmegaTile * kOmegaTile];
    __shared__ float d_hl[kOmegaTile * kOmegaTile];
    __shared__ float j_lh[kOmegaTile * kOmegaTile];
    __shared__ float d_lh[kOmegaTile * kOmegaTile];
    __shared__ float key_high[kChunk * kOmegaTile];
    __shared__ float key_low[kChunk * kOmegaTile];
    const unsigned program = blockIdx.x;
    if (program >= programs) return;
    unsigned high, low;
    decode_omega_pair(blockIdx.y, high, low);
    const bool diagonal = high == low;
    const unsigned element = threadIdx.x;
    const unsigned load_row = element / kOmegaTile;
    const unsigned load_col = element % kOmegaTile;
    const unsigned matrix_base = program * kRank * kRank;
    const unsigned high_row = high * kOmegaTile + load_row;
    const unsigned high_col = high * kOmegaTile + load_col;
    const unsigned low_row = low * kOmegaTile + load_row;
    const unsigned low_col = low * kOmegaTile + load_col;
    j_hl[element] = boundary_j[matrix_base + high_row * kRank + low_col];
    d_hl[element] = boundary_d[matrix_base + high_row * kRank + low_col];
    if (diagonal) {
        j_lh[element] = j_hl[element];
        d_lh[element] = d_hl[element];
    } else {
        j_lh[element] = boundary_j[matrix_base + low_row * kRank + high_col];
        d_lh[element] = boundary_d[matrix_base + low_row * kRank + high_col];
    }
    const unsigned target = element / kOmegaTile;
    const unsigned row = element % kOmegaTile;
    const unsigned vector_base = program * kChunk * kRank;
    key_high[element] = key[
        vector_base + target * kRank + high * kOmegaTile + load_col];
    key_low[element] = key[
        vector_base + target * kRank + low * kOmegaTile + load_col];
    __syncthreads();

    const unsigned scalar = program * kChunk + target;
    const float scale = 0.5f * alpha[scalar];
    const float h_lower = coefficient[scalar * kCoefficients + 0];
    const float r_lower = coefficient[scalar * kCoefficients + 1];
    const float h_upper = coefficient[scalar * kCoefficients + 2];
    const float r_upper = coefficient[scalar * kCoefficients + 3];
    FloatFloat jl{0.0f, 0.0f};
    FloatFloat dl{0.0f, 0.0f};
    FloatFloat ju{0.0f, 0.0f};
    FloatFloat du{0.0f, 0.0f};
    FloatFloat reverse_jl{0.0f, 0.0f};
    FloatFloat reverse_dl{0.0f, 0.0f};
    FloatFloat reverse_ju{0.0f, 0.0f};
    FloatFloat reverse_du{0.0f, 0.0f};
#pragma unroll
    for (unsigned inner = 0; inner < kOmegaTile; ++inner) {
        const float low_value = key_low[target * kOmegaTile + inner];
        const float high_value = key_high[target * kOmegaTile + inner];
        if (!diagonal) {
            jl = ff_add_product(jl, j_hl[row * kOmegaTile + inner], low_value);
            dl = ff_add_product(dl, d_hl[row * kOmegaTile + inner], low_value);
            ju = ff_add_product(ju, j_lh[inner * kOmegaTile + row], low_value);
            du = ff_add_product(du, d_lh[inner * kOmegaTile + row], low_value);
            reverse_ju = ff_add_product(
                reverse_ju, j_lh[row * kOmegaTile + inner], high_value);
            reverse_du = ff_add_product(
                reverse_du, d_lh[row * kOmegaTile + inner], high_value);
            reverse_jl = ff_add_product(
                reverse_jl, j_hl[inner * kOmegaTile + row], high_value);
            reverse_dl = ff_add_product(
                reverse_dl, d_hl[inner * kOmegaTile + row], high_value);
        } else if (row > inner) {
            jl = ff_add_product(jl, j_hl[row * kOmegaTile + inner], low_value);
            dl = ff_add_product(dl, d_hl[row * kOmegaTile + inner], low_value);
            ju = ff_add_product(ju, j_hl[inner * kOmegaTile + row], low_value);
            du = ff_add_product(du, d_hl[inner * kOmegaTile + row], low_value);
        } else if (row < inner) {
            ju = ff_add_product(ju, -j_hl[row * kOmegaTile + inner], low_value);
            du = ff_add_product(du, -d_hl[row * kOmegaTile + inner], low_value);
            jl = ff_add_product(jl, -j_hl[inner * kOmegaTile + row], low_value);
            dl = ff_add_product(dl, -d_hl[inner * kOmegaTile + row], low_value);
        }
    }
    FloatFloat action = ff_add(
        ff_multiply(jl, h_lower), ff_multiply(dl, r_lower));
    action = ff_add(action, ff_negate(ff_multiply(ju, h_upper)));
    action = ff_add(action, ff_negate(ff_multiply(du, r_upper)));
    action = ff_multiply(action, scale);
    constexpr unsigned block_stride = kChunk * kOmegaTile * 2;
    constexpr unsigned panel_stride =
        kOmegaTiles * kOmegaTiles * block_stride;
    unsigned destination = program * panel_stride
        + (high * kOmegaTiles + low) * block_stride
        + (target * kOmegaTile + row) * 2;
    workspace[destination] = action.hi;
    workspace[destination + 1] = action.lo;
    if (!diagonal) {
        FloatFloat reverse = ff_add(
            ff_multiply(reverse_ju, h_upper),
            ff_multiply(reverse_du, r_upper));
        reverse = ff_add(
            reverse, ff_negate(ff_multiply(reverse_jl, h_lower)));
        reverse = ff_add(
            reverse, ff_negate(ff_multiply(reverse_dl, r_lower)));
        reverse = ff_multiply(reverse, scale);
        destination = program * panel_stride
            + (low * kOmegaTiles + high) * block_stride
            + (target * kOmegaTile + row) * 2;
        workspace[destination] = reverse.hi;
        workspace[destination + 1] = reverse.lo;
    }
}

__launch_bounds__(kLocalThreads, 2)
__global__ void packet_omega_pair_local_kernel(
    const float* __restrict__ workspace,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ coefficient,
    const float* __restrict__ key,
    const float* __restrict__ erase,
    const float* __restrict__ query,
    const float* __restrict__ skew,
    float* __restrict__ omega,
    float* __restrict__ dual_rhs,
    unsigned programs) {
    __shared__ float shared_key[kChunk * kRank];
    const unsigned program = blockIdx.x;
    if (program >= programs) return;
    const unsigned vector_base = program * kChunk * kRank;
    for (unsigned index = threadIdx.x; index < kChunk * kRank;
         index += kLocalThreads) {
        shared_key[index] = key[vector_base + index];
    }
    __syncthreads();
    const unsigned target = threadIdx.x >> 3;
    const unsigned lane = threadIdx.x & 7;
    const unsigned source_0 = lane;
    const unsigned source_1 = lane + 8;
    const unsigned scalar = program * kChunk + target;
    const float h_lower = coefficient[scalar * kCoefficients + 0];
    const float r_lower = coefficient[scalar * kCoefficients + 1];
    const float h_upper = coefficient[scalar * kCoefficients + 2];
    const float r_upper = coefficient[scalar * kCoefficients + 3];
    const unsigned packet_base = program * kPacket + target;
    const float weight_0 = source_0 <= target
        ? weights[packet_base + source_0 * kChunk] : 0.0f;
    const float weight_1 = source_1 <= target
        ? weights[packet_base + source_1 * kChunk] : 0.0f;
    FloatFloat total_u_0{0.0f, 0.0f};
    FloatFloat total_h_0{0.0f, 0.0f};
    FloatFloat total_u_1{0.0f, 0.0f};
    FloatFloat total_h_1{0.0f, 0.0f};
    if (source_0 <= target) {
        for (unsigned coordinate = 0; coordinate < kRank; ++coordinate) {
            const float kv = shared_key[target * kRank + coordinate];
            total_u_0 = ff_add_product(
                total_u_0, u[vector_base + source_0 * kRank + coordinate], kv);
            total_h_0 = ff_add_product(
                total_h_0, h[vector_base + source_0 * kRank + coordinate], kv);
        }
    }
    if (source_1 <= target) {
        for (unsigned coordinate = 0; coordinate < kRank; ++coordinate) {
            const float kv = shared_key[target * kRank + coordinate];
            total_u_1 = ff_add_product(
                total_u_1, u[vector_base + source_1 * kRank + coordinate], kv);
            total_h_1 = ff_add_product(
                total_h_1, h[vector_base + source_1 * kRank + coordinate], kv);
        }
    }
    FloatFloat prefix_u_0{0.0f, 0.0f};
    FloatFloat prefix_h_0{0.0f, 0.0f};
    FloatFloat prefix_u_1{0.0f, 0.0f};
    FloatFloat prefix_h_1{0.0f, 0.0f};
    constexpr unsigned block_stride = kChunk * kOmegaTile * 2;
    constexpr unsigned panel_stride =
        kOmegaTiles * kOmegaTiles * block_stride;
#pragma unroll 1
    for (unsigned coordinate = 0; coordinate < kRank; ++coordinate) {
        const float kv = shared_key[target * kRank + coordinate];
        FloatFloat local_j{0.0f, 0.0f};
        FloatFloat local_d{0.0f, 0.0f};
#define PACKET_OMEGA_PAIR_SLOT(SOURCE, WEIGHT, PU, PH, TU, TH)                  \
        do {                                                                     \
            if ((SOURCE) <= target) {                                            \
                const float uv = u[vector_base + (SOURCE) * kRank + coordinate]; \
                const float hv = h[vector_base + (SOURCE) * kRank + coordinate]; \
                const FloatFloat current_u = ff_product(uv, kv);                 \
                const FloatFloat current_h = ff_product(hv, kv);                 \
                const FloatFloat suffix_u = ff_add(                              \
                    ff_add((TU), ff_negate(PU)), ff_negate(current_u));           \
                const FloatFloat suffix_h = ff_add(                              \
                    ff_add((TH), ff_negate(PH)), ff_negate(current_h));           \
                const FloatFloat j_inner = ff_add(                               \
                    ff_multiply((PU), h_lower - h_upper),                        \
                    ff_multiply(suffix_u, h_upper - h_lower));                   \
                local_j = ff_add(local_j, ff_multiply(                           \
                    ff_multiply(j_inner, uv), (WEIGHT)));                        \
                local_d = ff_add(local_d, local_driven_omega_action(             \
                    (PU), (PH), suffix_u, suffix_h, uv, hv,                      \
                    r_lower, r_upper, (WEIGHT)));                               \
                (PU) = ff_add((PU), current_u);                                 \
                (PH) = ff_add((PH), current_h);                                 \
            }                                                                    \
        } while (0)
        PACKET_OMEGA_PAIR_SLOT(
            source_0, weight_0, prefix_u_0, prefix_h_0, total_u_0, total_h_0);
        PACKET_OMEGA_PAIR_SLOT(
            source_1, weight_1, prefix_u_1, prefix_h_1, total_u_1, total_h_1);
#undef PACKET_OMEGA_PAIR_SLOT
        FloatFloat local = ff_add(
            reduce_group8(local_j), reduce_group8(local_d));
        const unsigned output_tile = coordinate / kOmegaTile;
        const unsigned local_row = coordinate % kOmegaTile;
        const unsigned partial = program * panel_stride
            + (output_tile * kOmegaTiles + lane) * block_stride
            + (target * kOmegaTile + local_row) * 2;
        FloatFloat boundary{workspace[partial], workspace[partial + 1]};
        boundary = reduce_group8(boundary);
        if (lane == 0) {
            FloatFloat action = ff_add(boundary, ff_multiply(local, 0.5f));
            const unsigned index = vector_base + target * kRank + coordinate;
            omega[index] = __fadd_rn(action.hi, action.lo);
        }
    }
    __syncwarp();
    float tau = 0.0f;
    float norm = 0.0f;
    for (unsigned coordinate = lane; coordinate < kRank; coordinate += 8) {
        const unsigned index = vector_base + target * kRank + coordinate;
        const float kv = shared_key[target * kRank + coordinate];
        tau = fmaf(erase[index] * kv, kv, tau);
        norm = fmaf(omega[index], omega[index], norm);
    }
    tau = reduce_group8(tau);
    norm = reduce_group8(norm);
    tau = __shfl_sync(0xffffffffu, tau, 0, 8);
    norm = __shfl_sync(0xffffffffu, norm, 0, 8);
    const float scale = tau * (2.0f - tau) * skew[scalar]
        * rsqrtf(1.0f + norm);
    const unsigned rhs_base = program * kChunk * kDualRhs * kRank
        + target * kDualRhs * kRank;
    for (unsigned coordinate = lane; coordinate < kRank; coordinate += 8) {
        const unsigned index = vector_base + target * kRank + coordinate;
        dual_rhs[rhs_base + coordinate] = erase[index]
            * shared_key[target * kRank + coordinate] + scale * omega[index];
        dual_rhs[rhs_base + kRank + coordinate] = query[index];
    }
}

struct DualShared {
    float* z_u;
    float* weights;
    float* rhs;
    float* intermediate;
    float* boundary_j_col;
    float* boundary_d_col;
    float* u_coordinate;
    float* h_coordinate;
    float* alpha;
    float* coefficient;
};

__device__ __forceinline__ DualShared partition_dual(float* storage) {
    DualShared shared{};
    shared.z_u = storage;
    storage += kDualRhs * kPacket;
    shared.weights = storage;
    storage += kPacket;
    shared.rhs = storage;
    storage += kChunk * kDualRhs * kRank;
    shared.intermediate = storage;
    storage += kChunk * kDualRhs * kRank;
    shared.boundary_j_col = storage;
    storage += kRank;
    shared.boundary_d_col = storage;
    storage += kRank;
    shared.u_coordinate = storage;
    storage += kChunk;
    shared.h_coordinate = storage;
    storage += kChunk;
    shared.alpha = storage;
    storage += kChunk;
    shared.coefficient = storage;
    return shared;
}

constexpr size_t dual_shared_bytes() {
    return sizeof(float) * (
        kDualRhs * kPacket + kPacket + 2 * kChunk * kDualRhs * kRank
        + 2 * kRank + 3 * kChunk + kCoefficients * kChunk);
}

template<bool SourceLower>
__device__ __forceinline__ void packet_transpose2_phase(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ saved_lower,
    DualShared shared,
    unsigned program) {
    const unsigned vector_base = program * kChunk * kRank;
    const unsigned boundary_base = program * kRank * kRank;

#pragma unroll 1
    for (unsigned step = 0; step < kRank; ++step) {
        const unsigned coordinate = SourceLower ? kRank - 1 - step : step;
        if (threadIdx.x < kRank) {
            const unsigned row = threadIdx.x;
            shared.boundary_j_col[row] =
                boundary_j[boundary_base + row * kRank + coordinate];
            shared.boundary_d_col[row] =
                boundary_d[boundary_base + row * kRank + coordinate];
        }
        if (threadIdx.x < kChunk) {
            const unsigned source = threadIdx.x;
            shared.u_coordinate[source] = u[vector_base + source * kRank + coordinate];
            shared.h_coordinate[source] = h[vector_base + source * kRank + coordinate];
        }
        __syncthreads();

        if (threadIdx.x < kChunk * kDualRhs * 4) {
            const unsigned item = threadIdx.x >> 2;
            const unsigned lane = threadIdx.x & 3;
            const unsigned target = item / kDualRhs;
            const unsigned rhs_index = item % kDualRhs;
            constexpr unsigned h_index = SourceLower ? 0 : 2;
            constexpr unsigned r_index = SourceLower ? 1 : 3;
            const float h_scale = shared.coefficient[target * kCoefficients + h_index];
            const float r_scale = shared.coefficient[target * kCoefficients + r_index];
            float j_col = 0.0f;
            float d_col = 0.0f;
            if constexpr (SourceLower) {
                for (unsigned row = coordinate + 1 + lane; row < kRank; row += 4) {
                    const float rhs_value = input[
                        (target * kDualRhs + rhs_index) * kRank + row];
                    j_col = fmaf(shared.boundary_j_col[row], rhs_value, j_col);
                    d_col = fmaf(shared.boundary_d_col[row], rhs_value, d_col);
                }
            } else {
                for (unsigned row = lane; row < coordinate; row += 4) {
                    const float rhs_value = input[
                        (target * kDualRhs + rhs_index) * kRank + row];
                    j_col = fmaf(shared.boundary_j_col[row], rhs_value, j_col);
                    d_col = fmaf(shared.boundary_d_col[row], rhs_value, d_col);
                }
            }
            float local = 0.0f;
            for (unsigned source = lane; source <= target; source += 4) {
                const unsigned packet = source * kChunk + target;
                const float z_u = shared.z_u[rhs_index * kPacket + packet];
                const float factor = h_scale * shared.u_coordinate[source]
                    + r_scale * shared.h_coordinate[source];
                local = fmaf(shared.weights[packet] * factor, z_u, local);
            }
            const float boundary = shared.alpha[target]
                * (h_scale * j_col + r_scale * d_col);
            const float action = reduce_group4(boundary + local);
            if (lane == 0) {
                const unsigned output_index =
                    (target * kDualRhs + rhs_index) * kRank + coordinate;
                float value = input[output_index] + action;
                if constexpr (SourceLower) {
                    saved_lower[
                        program * kChunk * kDualRhs * kRank + output_index] = value;
                    value *= diagonal[vector_base + target * kRank + coordinate];
                }
                output[output_index] = value;
            }
        }
        __syncthreads();

        for (unsigned index = threadIdx.x; index < kDualRhs * kPacket;
             index += kThreads) {
            const unsigned rhs_index = index / kPacket;
            const unsigned packet = index % kPacket;
            const unsigned source = packet / kChunk;
            const unsigned target = packet % kChunk;
            if (source <= target) {
                const float rhs_value = input[
                    (target * kDualRhs + rhs_index) * kRank + coordinate];
                shared.z_u[index] = fmaf(
                    shared.u_coordinate[source], rhs_value, shared.z_u[index]);
            }
        }
        __syncthreads();
    }
}

__launch_bounds__(kThreads, 1)
__global__ void packet_dual2_kernel(
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ u,
    const float* __restrict__ h,
    const float* __restrict__ weights,
    const float* __restrict__ alpha,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ rhs,
    float* __restrict__ output,
    float* __restrict__ saved_lower,
    unsigned programs) {
    extern __shared__ float shared_storage[];
    DualShared shared = partition_dual(shared_storage);
    const unsigned program = blockIdx.x;
    if (program >= programs) {
        return;
    }
    const unsigned packet_base = program * kPacket;
    const unsigned rhs_base = program * kChunk * kDualRhs * kRank;
    const unsigned scalar_base = program * kChunk;
    for (unsigned index = threadIdx.x; index < kDualRhs * kPacket;
         index += kThreads) {
        shared.z_u[index] = 0.0f;
    }
    for (unsigned index = threadIdx.x; index < kPacket; index += kThreads) {
        shared.weights[index] = weights[packet_base + index];
    }
    for (unsigned index = threadIdx.x; index < kChunk * kDualRhs * kRank;
         index += kThreads) {
        shared.rhs[index] = rhs[rhs_base + index];
    }
    if (threadIdx.x < kChunk) {
        shared.alpha[threadIdx.x] = alpha[scalar_base + threadIdx.x];
    }
    for (unsigned index = threadIdx.x; index < kChunk * kCoefficients;
         index += kThreads) {
        shared.coefficient[index] = coefficient[scalar_base * kCoefficients + index];
    }
    __syncthreads();
    packet_transpose2_phase<true>(
        boundary_j, boundary_d, u, h, weights, alpha, coefficient,
        diagonal, shared.rhs, shared.intermediate, saved_lower, shared, program);

    for (unsigned index = threadIdx.x; index < kDualRhs * kPacket;
         index += kThreads) {
        shared.z_u[index] = 0.0f;
    }
    __syncthreads();
    packet_transpose2_phase<false>(
        boundary_j, boundary_d, u, h, weights, alpha, coefficient,
        diagonal, shared.intermediate, output + rhs_base, nullptr, shared, program);
}

void check_inputs(
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& weights,
    const at::Tensor& alpha,
    const at::Tensor& coefficient,
    const at::Tensor& diagonal,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& skew,
    const at::Tensor& omega) {
    TORCH_CHECK(
        boundary_j.is_cuda() && boundary_j.scalar_type() == at::kFloat
            && boundary_j.is_contiguous() && boundary_j.dim() == 3
            && boundary_j.size(1) == kRank && boundary_j.size(2) == kRank,
        "boundary_j must be contiguous CUDA FP32 [P,128,128]");
    const auto programs = boundary_j.size(0);
    const auto check = [&](const at::Tensor& tensor) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.get_device() == boundary_j.get_device()
                && tensor.scalar_type() == at::kFloat && tensor.is_contiguous(),
            "all packet-frame inputs must be contiguous CUDA FP32 on one device");
    };
    for (const auto& tensor : {
             boundary_d, u, h, weights, alpha, coefficient, diagonal,
             key, erase, query, skew, omega}) {
        check(tensor);
    }
    TORCH_CHECK(boundary_d.sizes() == boundary_j.sizes(), "boundary_d shape mismatch");
    TORCH_CHECK(
        u.sizes() == at::IntArrayRef({programs, kChunk, kRank}),
        "u must be [P,16,128]");
    TORCH_CHECK(
        h.sizes() == u.sizes() && key.sizes() == u.sizes()
            && erase.sizes() == u.sizes() && query.sizes() == u.sizes()
            && diagonal.sizes() == u.sizes() && omega.sizes() == u.sizes(),
        "packed vector shape mismatch");
    TORCH_CHECK(
        weights.sizes() == at::IntArrayRef({programs, kChunk, kChunk}),
        "weights must be source-major [P,16,16]");
    TORCH_CHECK(
        alpha.sizes() == at::IntArrayRef({programs, kChunk}),
        "alpha must be [P,16]");
    TORCH_CHECK(
        coefficient.sizes() == at::IntArrayRef({programs, kChunk, kCoefficients}),
        "coefficient must be [P,16,4]");
    TORCH_CHECK(
        skew.sizes() == at::IntArrayRef({programs, kChunk}),
        "skew must be [P,16]");
}

template<typename Kernel>
void set_dynamic_shared(Kernel kernel, size_t bytes) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(bytes)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
packet_frame128_cuda(
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& weights,
    const at::Tensor& alpha,
    const at::Tensor& coefficient,
    const at::Tensor& diagonal,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& skew,
    at::Tensor& omega) {
    check_inputs(
        boundary_j, boundary_d, u, h, weights, alpha, coefficient,
        diagonal, key, erase, query, skew, omega);
    const unsigned programs = static_cast<unsigned>(boundary_j.size(0));
    c10::cuda::CUDAGuard guard(boundary_j.device());
    const auto stream = at::cuda::getCurrentCUDAStream();

    auto write_direction = at::empty_like(key);
    auto lower_solution = at::empty_like(key);
    set_dynamic_shared(packet_primal_kernel, triangular_shared_bytes());
    packet_primal_kernel<<<programs, kThreads, triangular_shared_bytes(), stream>>>(
        boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), weights.data_ptr<float>(),
        alpha.data_ptr<float>(), coefficient.data_ptr<float>(),
        diagonal.data_ptr<float>(), key.data_ptr<float>(),
        write_direction.data_ptr<float>(), lower_solution.data_ptr<float>(), programs);

    auto dual_rhs = at::empty(
        {static_cast<int64_t>(programs), kChunk, kDualRhs, kRank}, key.options());
    auto omega_workspace = at::empty(
        {static_cast<int64_t>(programs), kOmegaTiles, kOmegaTiles,
         kChunk, kOmegaTile, 2}, key.options());
    packet_omega_pair_boundary_kernel<<<
        dim3(programs, kOmegaPairs), kThreads, 0, stream>>>(
        boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        key.data_ptr<float>(), alpha.data_ptr<float>(),
        coefficient.data_ptr<float>(), omega_workspace.data_ptr<float>(), programs);
    packet_omega_pair_local_kernel<<<programs, kLocalThreads, 0, stream>>>(
        omega_workspace.data_ptr<float>(),
        u.data_ptr<float>(), h.data_ptr<float>(), weights.data_ptr<float>(),
        coefficient.data_ptr<float>(),
        key.data_ptr<float>(), erase.data_ptr<float>(),
        query.data_ptr<float>(), skew.data_ptr<float>(), omega.data_ptr<float>(),
        dual_rhs.data_ptr<float>(), programs);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(write_direction, lower_solution, dual_rhs);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(causallsso, m) {
    m.def(
        "packet_frame128(Tensor boundary_j, Tensor boundary_d, Tensor u, Tensor h, "
        "Tensor weights, Tensor alpha, Tensor coefficient, Tensor diagonal, "
        "Tensor key, Tensor erase, Tensor query, Tensor skew, Tensor(a!) omega) "
        "-> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl("packet_frame128", &packet_frame128_cuda);
}
