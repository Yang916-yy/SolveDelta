#include "solvedelta_c32.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <tuple>

#include "solvedelta_paired_action.cuh"


namespace {

constexpr int kRank = 128;
constexpr int kChunk = 32;
constexpr int kTile = 16;
constexpr int kComponents = 4;
constexpr int kDualRhs = 2;
constexpr int kWYWarps = 8;
constexpr int kThreads = kChunk * kTile;
constexpr int kEntriesPerThread = 16;
constexpr int kStrictEntries = kRank * (kRank - 1) / 2;
constexpr double kRadius = 1.0 / 8.0;

__device__ __forceinline__ int vector_index(
    int batch_index,
    int token,
    int head,
    int coordinate,
    int length,
    int heads) {
    return ((batch_index * length + token) * heads + head) * kRank
        + coordinate;
}

__device__ __forceinline__ void decode_panel(
    int panel,
    int heads,
    int chunks,
    int& batch_index,
    int& head,
    int& chunk) {
    chunk = panel % chunks;
    const int head_batch = panel / chunks;
    head = head_batch % heads;
    batch_index = head_batch / heads;
}

__device__ __forceinline__ void decode_lower_entry(
    int entry,
    int& row,
    int& column) {
    row = static_cast<int>(
        (1.0f + sqrtf(1.0f + 8.0f * static_cast<float>(entry))) * 0.5f);
    while (row * (row - 1) / 2 > entry) --row;
    while ((row + 1) * row / 2 <= entry) ++row;
    column = entry - row * (row - 1) / 2;
}

__device__ __forceinline__ void reduce_warp(float& value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
}

template <typename vector_t>
__global__ __launch_bounds__(kThreads, 1) void radial_kernel(
    const vector_t* __restrict__ u,
    const vector_t* __restrict__ h,
    const float* __restrict__ geometry_log_decay,
    const float* __restrict__ geometry_strength,
    const float* __restrict__ boundary_m,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    float* __restrict__ alpha0,
    float* __restrict__ inverse_mass_output,
    float* __restrict__ coefficient,
    float* __restrict__ radial_q2,
    float* __restrict__ radial_norm,
    float* __restrict__ diagonal,
    int length,
    int heads,
    int chunks,
    int panels) {
    __shared__ float vectors[2][kChunk * kRank];
    __shared__ float decay[kChunk];
    __shared__ float inverse_mass[kChunk];
    __shared__ float reduction[2][16];

    const int panel = blockIdx.x;
    const int side = blockIdx.y;
    if (panel >= panels) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    const int matrix_base = panel * kRank * kRank;

    for (int index = threadIdx.x;
         index < 2 * kChunk * kRank;
         index += blockDim.x) {
        const int which = index / (kChunk * kRank);
        const int local = index % (kChunk * kRank);
        const int target = local / kRank;
        const int coordinate = local % kRank;
        float value = 0.0f;
        if (target < valid_count) {
            const int source_index = vector_index(
                batch_index,
                token_start + target,
                head,
                coordinate,
                length,
                heads);
            value = static_cast<float>(
                which == 0 ? u[source_index] : h[source_index]);
        }
        vectors[which][local] = value;
    }

    if (threadIdx.x == 0) {
        float mass = boundary_m[panel];
        float first_alpha = 0.0f;
#pragma unroll
        for (int target = 0; target < kChunk; ++target) {
            const bool valid = target < valid_count;
            float factor = 1.0f;
            if (valid) {
                const int scalar_index =
                    (batch_index * length + token_start + target) * heads + head;
                factor = expf(geometry_log_decay[scalar_index]);
                mass = fmaf(factor, mass, 1.0f);
            }
            const float weight = valid ? 1.0f / mass : 0.0f;
            decay[target] = factor;
            inverse_mass[target] = weight;
            if (target == 0) first_alpha = factor * weight;
            if (side == 0) {
                inverse_mass_output[panel * kChunk + target] = weight;
            }
        }
        if (side == 0) alpha0[panel] = first_alpha;
    }
    __syncthreads();

    int rows[kEntriesPerThread];
    int columns[kEntriesPerThread];
    bool active[kEntriesPerThread];
    float moment_j[kEntriesPerThread];
    float moment_d[kEntriesPerThread];
#pragma unroll
    for (int slot = 0; slot < kEntriesPerThread; ++slot) {
        const int entry = threadIdx.x + slot * blockDim.x;
        active[slot] = entry < kStrictEntries;
        int lower_row = 1;
        int lower_column = 0;
        if (active[slot]) decode_lower_entry(entry, lower_row, lower_column);
        rows[slot] = side == 0 ? lower_row : lower_column;
        columns[slot] = side == 0 ? lower_column : lower_row;
        moment_j[slot] = active[slot]
            ? boundary_j[matrix_base + rows[slot] * kRank + columns[slot]]
            : 0.0f;
        moment_d[slot] = active[slot]
            ? boundary_d[matrix_base + rows[slot] * kRank + columns[slot]]
            : 0.0f;
    }

    float diagonal_j = 0.0f;
    float diagonal_d = 0.0f;
    const bool owns_diagonal = side == 0 && threadIdx.x < kRank;
    if (owns_diagonal) {
        diagonal_j = boundary_j[
            matrix_base + threadIdx.x * (kRank + 1)];
        diagonal_d = boundary_d[
            matrix_base + threadIdx.x * (kRank + 1)];
    }

    const float strength = geometry_strength[head];
    constexpr float radius = static_cast<float>(kRadius);
#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        const bool valid = target < valid_count;
        float norm_j = 0.0f;
        float norm_d = 0.0f;
        if (valid) {
            const float factor = decay[target];
            const float weight = inverse_mass[target];
#pragma unroll
            for (int slot = 0; slot < kEntriesPerThread; ++slot) {
                if (active[slot]) {
                    const float row_u =
                        vectors[0][target * kRank + rows[slot]];
                    moment_j[slot] = fmaf(
                        factor,
                        moment_j[slot],
                        row_u * vectors[0][target * kRank + columns[slot]]);
                    moment_d[slot] = fmaf(
                        factor,
                        moment_d[slot],
                        row_u * vectors[1][target * kRank + columns[slot]]);
                    const float normalized_j = moment_j[slot] * weight;
                    const float normalized_d = moment_d[slot] * weight;
                    norm_j = fmaf(normalized_j, normalized_j, norm_j);
                    norm_d = fmaf(normalized_d, normalized_d, norm_d);
                }
            }
            if (owns_diagonal) {
                const int coordinate = threadIdx.x;
                const float token_u =
                    vectors[0][target * kRank + coordinate];
                diagonal_j = fmaf(factor, diagonal_j, token_u * token_u);
                diagonal_d = fmaf(
                    factor,
                    diagonal_d,
                    token_u * vectors[1][target * kRank + coordinate]);
                const float x_j = strength * (
                    diagonal_j * weight
                    - 1.0f / static_cast<float>(kRank));
                const float x_d = strength * diagonal_d * weight;
                diagonal[(panel * kChunk + target) * kRank + coordinate] =
                    expf(
                        radius * tanhf(x_j / radius)
                        + radius * tanhf(x_d / radius));
            }
        } else if (owns_diagonal) {
            diagonal[(panel * kChunk + target) * kRank + threadIdx.x] = 1.0f;
        }

        reduce_warp(norm_j);
        reduce_warp(norm_d);
        if (lane == 0) {
            reduction[0][warp] = norm_j;
            reduction[1][warp] = norm_d;
        }
        __syncthreads();
        if (warp == 0) {
            float block_j = lane < 16 ? reduction[0][lane] : 0.0f;
            float block_d = lane < 16 ? reduction[1][lane] : 0.0f;
            reduce_warp(block_j);
            reduce_warp(block_d);
            if (lane == 0) {
                const int output =
                    (panel * kChunk + target) * kComponents + side * 2;
                const double strength_squared =
                    static_cast<double>(strength)
                    * static_cast<double>(strength);
                const double scaled_j = valid
                    ? strength_squared * static_cast<double>(block_j)
                    : 0.0;
                const double scaled_d = valid
                    ? strength_squared * static_cast<double>(block_d)
                    : 0.0;
                const double q2_j = kRadius * kRadius + scaled_j;
                const double q2_d = kRadius * kRadius + scaled_d;
                coefficient[output] = valid
                    ? static_cast<float>(
                        static_cast<double>(strength) * kRadius
                        / sqrt(q2_j))
                    : 0.0f;
                coefficient[output + 1] = valid
                    ? static_cast<float>(
                        static_cast<double>(strength) * kRadius
                        / sqrt(q2_d))
                    : 0.0f;
                radial_q2[output] = static_cast<float>(q2_j);
                radial_q2[output + 1] = static_cast<float>(q2_d);
                radial_norm[output] = valid ? block_j : 0.0f;
                radial_norm[output + 1] = valid ? block_d : 0.0f;
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__device__ __forceinline__ float load_chunk_vector(
    const scalar_t* source,
    int batch_index,
    int token_start,
    int target,
    int head,
    int coordinate,
    int valid_count,
    int length,
    int heads) {
    if (target >= valid_count) return 0.0f;
    return source[vector_index(
        batch_index,
        token_start + target,
        head,
        coordinate,
        length,
        heads)];
}

__device__ __forceinline__ int dual_vector_index(
    int batch_index,
    int token,
    int head,
    int route,
    int coordinate,
    int length,
    int heads) {
    return ((((batch_index * length + token) * heads + head) * kDualRhs
             + route) * kRank) + coordinate;
}

struct ActionPanelLayout {
    int batch_index;
    int head;
    int token_start;
    int valid_count;
    int length;
    int heads;

    __device__ __forceinline__ int vector(int target, int coordinate) const {
        return vector_index(
            batch_index,
            token_start + target,
            head,
            coordinate,
            length,
            heads);
    }

    __device__ __forceinline__ int dual(
        int target,
        int route,
        int coordinate) const {
        return dual_vector_index(
            batch_index,
            token_start + target,
            head,
            route,
            coordinate,
            length,
            heads);
    }
};

struct ActionGeometryView {
    const at::BFloat16* u_data;
    const at::BFloat16* h_data;
    const float* boundary_j_data;
    const float* boundary_d_data;
    const float* inverse_mass_data;
    const float* coefficient_data;
    const float* alpha0_data;
    ActionPanelLayout layout;
    int panel;

    __device__ __forceinline__ at::BFloat16 u(
        int source,
        int coordinate) const {
        return u_data[layout.vector(source, coordinate)];
    }

    __device__ __forceinline__ at::BFloat16 h(
        int source,
        int coordinate) const {
        return h_data[layout.vector(source, coordinate)];
    }

    __device__ __forceinline__ float boundary_j(
        int row,
        int column) const {
        return boundary_j_data[
            (panel * kRank + row) * kRank + column];
    }

    __device__ __forceinline__ float boundary_d(
        int row,
        int column) const {
        return boundary_d_data[
            (panel * kRank + row) * kRank + column];
    }

    __device__ __forceinline__ float inverse_mass(int target) const {
        return inverse_mass_data[panel * kChunk + target];
    }

    __device__ __forceinline__ float coefficient(
        int target,
        int component) const {
        return coefficient_data[
            (panel * kChunk + target) * kComponents + component];
    }

    __device__ __forceinline__ float alpha0() const {
        return alpha0_data[panel];
    }

    __device__ __forceinline__ int valid_count() const {
        return layout.valid_count;
    }
};

struct KeyActionInput {
    const at::BFloat16* key;
    ActionPanelLayout layout;

    __device__ __forceinline__ float load(
        int target,
        int,
        int coordinate) const {
        return static_cast<float>(key[layout.vector(target, coordinate)]);
    }
};

struct OriginalDualActionInput {
    const at::BFloat16* key;
    const at::BFloat16* erase;
    const at::BFloat16* query;
    ActionPanelLayout layout;

    __device__ __forceinline__ float load(
        int target,
        int route,
        int coordinate) const {
        const int index = layout.vector(target, coordinate);
        return route == 0
            ? static_cast<float>(erase[index])
                * static_cast<float>(key[index])
            : static_cast<float>(query[index]);
    }
};

struct Fp32VectorActionView {
    float* data;
    ActionPanelLayout layout;

    __device__ __forceinline__ float load(
        int target,
        int,
        int coordinate) const {
        return data[layout.vector(target, coordinate)];
    }

    __device__ __forceinline__ void store(
        int target,
        int,
        int coordinate,
        float value) const {
        data[layout.vector(target, coordinate)] = value;
    }
};

struct Bf16VectorActionView {
    at::BFloat16* data;
    ActionPanelLayout layout;

    __device__ __forceinline__ void store(
        int target,
        int,
        int coordinate,
        float value) const {
        data[layout.vector(target, coordinate)] = at::BFloat16(value);
    }
};

struct DiagonalPrimalActionInput {
    const at::BFloat16* data;
    const float* diagonal;
    ActionPanelLayout layout;
    int panel;

    __device__ __forceinline__ float load(
        int target,
        int,
        int coordinate) const {
        const float scale = static_cast<float>(at::BFloat16(diagonal[
            (panel * kChunk + target) * kRank + coordinate]));
        return static_cast<float>(data[layout.vector(target, coordinate)]) / scale;
    }
};

struct ScaledDualActionOutput {
    at::BFloat16* data;
    const float* diagonal;
    ActionPanelLayout layout;
    int panel;

    __device__ __forceinline__ void store(
        int target,
        int route,
        int coordinate,
        float value) const {
        const float scale = static_cast<float>(at::BFloat16(diagonal[
            (panel * kChunk + target) * kRank + coordinate]));
        data[layout.dual(target, route, coordinate)] =
            at::BFloat16(value * scale);
    }
};

struct Fp32DualActionView {
    float* data;
    ActionPanelLayout layout;

    __device__ __forceinline__ float load(
        int target,
        int route,
        int coordinate) const {
        return data[layout.dual(target, route, coordinate)];
    }

    __device__ __forceinline__ void store(
        int target,
        int route,
        int coordinate,
        float value) const {
        data[layout.dual(target, route, coordinate)] = value;
    }
};

struct Bf16DualActionView {
    at::BFloat16* data;
    ActionPanelLayout layout;

    __device__ __forceinline__ float load(
        int target,
        int route,
        int coordinate) const {
        return static_cast<float>(
            data[layout.dual(target, route, coordinate)]);
    }
};

struct FinalDualActionOutput {
    at::BFloat16* erase_dual;
    at::BFloat16* query_dual;
    ActionPanelLayout layout;

    __device__ __forceinline__ void store(
        int target,
        int route,
        int coordinate,
        float value) const {
        const int index = layout.vector(target, coordinate);
        if (route == 0) {
            erase_dual[index] = at::BFloat16(value);
        } else {
            query_dual[index] = at::BFloat16(value);
        }
    }
};

struct GradPrimalActionInput {
    const float* grad_d;
    ActionPanelLayout layout;

    __device__ __forceinline__ float load(
        int target,
        int,
        int coordinate) const {
        const int index = layout.vector(target, coordinate);
        return grad_d[index];
    }
};

struct GradDualActionInput {
    const float* grad_e;
    const float* grad_chi;
    ActionPanelLayout layout;

    __device__ __forceinline__ float load(
        int target,
        int route,
        int coordinate) const {
        const int index = layout.vector(target, coordinate);
        return route == 0
            ? grad_e[index]
            : grad_chi[index];
    }
};

struct FinalGradientDualOutput {
    const at::BFloat16* key;
    const at::BFloat16* erase;
    const float* lower_rhs;
    at::BFloat16* grad_key;
    at::BFloat16* grad_erase;
    at::BFloat16* grad_query;
    ActionPanelLayout layout;

    __device__ __forceinline__ void store(
        int target,
        int route,
        int coordinate,
        float value) const {
        const int index = layout.vector(target, coordinate);
        if (route == 0) {
            grad_key[index] = at::BFloat16(
                lower_rhs[index]
                + static_cast<float>(erase[index]) * value);
            grad_erase[index] = at::BFloat16(
                static_cast<float>(key[index]) * value);
        } else {
            grad_query[index] = at::BFloat16(value);
        }
    }
};

__global__ __launch_bounds__(kThreads, 2) void mixed_frame_kernel(
    const at::BFloat16* __restrict__ u,
    const at::BFloat16* __restrict__ h,
    const at::BFloat16* __restrict__ key,
    const at::BFloat16* __restrict__ erase,
    const at::BFloat16* __restrict__ query,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ alpha0,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    at::BFloat16* __restrict__ d,
    at::BFloat16* __restrict__ e,
    at::BFloat16* __restrict__ chi,
    at::BFloat16* __restrict__ lower_primal,
    at::BFloat16* __restrict__ lower_dual_scaled,
    const float* __restrict__ inclusive_decay,
    const at::BFloat16* __restrict__ write,
    const at::BFloat16* __restrict__ value,
    float* __restrict__ W,
    float* __restrict__ A_qd,
    at::BFloat16* __restrict__ Q_gamma,
    at::BFloat16* __restrict__ D_tail,
    float* __restrict__ G_last,
    at::BFloat16* __restrict__ Y,
    at::BFloat16* __restrict__ U_z,
    int length,
    int heads,
    int chunks,
    int value_dim,
    int panels) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<
        solvedelta::paired_action::PairedActionShared*>(storage);
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    const ActionPanelLayout layout{
        batch_index,
        head,
        token_start,
        valid_count,
        length,
        heads};
    const ActionGeometryView geometry{
        u,
        h,
        boundary_j,
        boundary_d,
        inverse_mass,
        coefficient,
        alpha0,
        layout,
        panel};
    const KeyActionInput key_input{key, layout};
    const Bf16VectorActionView lower_primal_view{lower_primal, layout};
    const OriginalDualActionInput original_dual{
        key, erase, query, layout};
    const ScaledDualActionOutput lower_dual_output{
        lower_dual_scaled, diagonal, layout, panel};
    solvedelta::paired_action::run_paired_action<false, false>(
        shared,
        geometry,
        key_input,
        lower_primal_view,
        original_dual,
        lower_dual_output);

    const DiagonalPrimalActionInput upper_primal_input{
        lower_primal, diagonal, layout, panel};
    const Bf16VectorActionView final_primal{d, layout};
    const Bf16DualActionView upper_dual_input{
        lower_dual_scaled, layout};
    const FinalDualActionOutput final_dual{
        e, chi, layout};
    solvedelta::paired_action::run_paired_action<true, false>(
        shared,
        geometry,
        upper_primal_input,
        final_primal,
        upper_dual_input,
        final_dual);

    __syncthreads();
    float* alpha = reinterpret_cast<float*>(&shared);
    for (int local = threadIdx.x;
         local < kChunk * kRank;
         local += blockDim.x) {
        const int token = local / kRank;
        const int coordinate = local % kRank;
        float gate = 0.0f;
        if (token < valid_count) {
            const int index = layout.vector(token, coordinate);
            const float current = inclusive_decay[index];
            const float previous = token == 0
                ? 0.0f
                : inclusive_decay[layout.vector(token - 1, coordinate)];
            gate = expf(current - previous);
        }
        alpha[local] = gate;
    }
    for (int local = threadIdx.x;
         local < kChunk * kChunk;
         local += blockDim.x) {
        const int row = local / kChunk;
        const int column = local % kChunk;
        const int pair = (panel * kChunk + row) * kChunk + column;
        W[pair] = row == column ? 1.0f : 0.0f;
        A_qd[pair] = 0.0f;
    }
    __syncthreads();

    for (int local = threadIdx.x;
         local < valid_count * kRank;
         local += blockDim.x) {
        const int target = local / kRank;
        const int coordinate = local % kRank;
        const int index = layout.vector(target, coordinate);
        const int last = layout.vector(valid_count - 1, coordinate);
        const float gate = inclusive_decay[index];
        const float final_gate = inclusive_decay[last];
        Q_gamma[index] = at::BFloat16(
            static_cast<float>(chi[index]) * expf(gate));
        D_tail[index] = at::BFloat16(
            static_cast<float>(d[index]) * expf(final_gate - gate));
        if (target == 0) {
            G_last[panel * kRank + coordinate] = final_gate;
        }
    }

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp < kWYWarps) {
#pragma unroll
        for (int group = 0; group < kChunk / kWYWarps; ++group) {
            const int target = group * kWYWarps + warp;
            if (target >= valid_count) continue;
            float erase_value[4];
            float query_value[4];
            float ratio[4];
#pragma unroll
            for (int route = 0; route < 4; ++route) {
                const int coordinate = lane + route * 32;
                const int index = layout.vector(target, coordinate);
                erase_value[route] = static_cast<float>(e[index]);
                query_value[route] = static_cast<float>(chi[index]);
                ratio[route] = 1.0f;
            }

            for (int source = target; source >= 0; --source) {
                float edit_dot = 0.0f;
                float query_dot = 0.0f;
#pragma unroll
                for (int route = 0; route < 4; ++route) {
                    const int coordinate = lane + route * 32;
                    const int source_index = layout.vector(source, coordinate);
                    const float decayed_d = ratio[route]
                        * static_cast<float>(d[source_index]);
                    edit_dot += erase_value[route] * decayed_d;
                    query_dot += query_value[route] * decayed_d;
                }
                reduce_warp(edit_dot);
                reduce_warp(query_dot);
                if (lane == 0) {
                    const int pair =
                        (panel * kChunk + target) * kChunk + source;
                    if (source < target) W[pair] += edit_dot;
                    A_qd[pair] = query_dot;
                }
                if (source > 0) {
#pragma unroll
                    for (int route = 0; route < 4; ++route) {
                        const int coordinate = lane + route * 32;
                        ratio[route] *= alpha[source * kRank + coordinate];
                    }
                }
            }
        }
    }
    __syncthreads();

    float* shared_W = reinterpret_cast<float*>(&shared);
    for (int index = threadIdx.x;
         index < kChunk * kChunk;
         index += blockDim.x) {
        shared_W[index] = W[panel * kChunk * kChunk + index];
    }
    __syncthreads();
    const int rhs = threadIdx.x;
    const int rhs_count = kRank + value_dim;
    if (rhs < rhs_count) {
        float solution[kChunk];
#pragma unroll 1
        for (int row = 0; row < kChunk; ++row) {
            float current = 0.0f;
            if (row < valid_count) {
                if (rhs < kRank) {
                    const int index = layout.vector(row, rhs);
                    current = static_cast<float>(e[index])
                        * expf(inclusive_decay[index]);
                } else {
                    const int coordinate = rhs - kRank;
                    const int index =
                        (((layout.batch_index * length
                           + layout.token_start + row) * heads
                          + layout.head) * value_dim) + coordinate;
                    current = static_cast<float>(at::BFloat16(
                        static_cast<float>(write[index])
                        * static_cast<float>(value[index])));
                }
                for (int column = 0; column < row; ++column) {
                    current = fmaf(
                        -shared_W[row * kChunk + column],
                        solution[column],
                        current);
                }
            }
            solution[row] = current;
        }
        for (int row = 0; row < valid_count; ++row) {
            if (rhs < kRank) {
                Y[layout.vector(row, rhs)] = at::BFloat16(solution[row]);
            } else {
                const int coordinate = rhs - kRank;
                const int index =
                    (((layout.batch_index * length
                       + layout.token_start + row) * heads
                      + layout.head) * value_dim) + coordinate;
                U_z[index] = at::BFloat16(solution[row]);
            }
        }
    }
}

__global__ __launch_bounds__(kThreads, 2)
void mixed_frame_adjoint_kernel(
    const at::BFloat16* __restrict__ u,
    const at::BFloat16* __restrict__ h,
    const at::BFloat16* __restrict__ key,
    const at::BFloat16* __restrict__ erase,
    const at::BFloat16* __restrict__ query,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const at::BFloat16* __restrict__ lower_primal,
    const at::BFloat16* __restrict__ lower_dual_scaled,
    const at::BFloat16* __restrict__ d,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    const float* __restrict__ alpha0,
    const float* __restrict__ grad_d,
    const float* __restrict__ grad_e,
    const float* __restrict__ grad_chi,
    at::BFloat16* __restrict__ grad_key,
    at::BFloat16* __restrict__ grad_erase,
    at::BFloat16* __restrict__ grad_query,
    float* __restrict__ upper_primal,
    float* __restrict__ upper_dual_output,
    float* __restrict__ grad_log_diagonal,
    at::BFloat16* __restrict__ descriptor_bundle,
    int length,
    int heads,
    int chunks,
    int panels) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<
        solvedelta::paired_action::PairedActionShared*>(storage);
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    const ActionPanelLayout layout{
        batch_index,
        head,
        token_start,
        valid_count,
        length,
        heads};

    for (int index = threadIdx.x;
         index < valid_count * kRank;
         index += blockDim.x) {
        const int target = index / kRank;
        const int coordinate = index % kRank;
        const int vector = layout.vector(target, coordinate);
        const int descriptor_base =
            (((3 * panels + panel) * kChunk + target) * 3) * kRank
            + coordinate;
        descriptor_bundle[descriptor_base] = d[vector];
        descriptor_bundle[descriptor_base + kRank] = at::BFloat16(grad_e[vector]);
        descriptor_bundle[descriptor_base + 2 * kRank] = at::BFloat16(grad_chi[vector]);
    }
    __syncthreads();

    const ActionGeometryView geometry{
        u,
        h,
        boundary_j,
        boundary_d,
        inverse_mass,
        coefficient,
        alpha0,
        layout,
        panel};
    const GradPrimalActionInput grad_primal_input{grad_d, layout};
    const Fp32VectorActionView upper_primal_view{
        upper_primal, layout};
    const GradDualActionInput grad_dual_input{grad_e, grad_chi, layout};
    const Fp32DualActionView upper_dual_view{
        upper_dual_output, layout};
    solvedelta::paired_action::run_paired_action<true, true>(
        shared,
        geometry,
        grad_primal_input,
        upper_primal_view,
        grad_dual_input,
        upper_dual_view);

    for (int index = threadIdx.x;
         index < valid_count * kRank;
         index += blockDim.x) {
        const int target = index / kRank;
        const int coordinate = index % kRank;
        const int vector = layout.vector(target, coordinate);
        const int descriptor_base =
            (((2 * panels + panel) * kChunk + target) * 3) * kRank
            + coordinate;
        descriptor_bundle[descriptor_base] = at::BFloat16(-upper_primal[vector]);
        descriptor_bundle[descriptor_base + kRank] =
            lower_dual_scaled[layout.dual(target, 0, coordinate)];
        descriptor_bundle[descriptor_base + 2 * kRank] =
            lower_dual_scaled[layout.dual(target, 1, coordinate)];
    }
    __syncthreads();

    for (int index = threadIdx.x;
         index < valid_count * kRank;
         index += blockDim.x) {
        const int target = index / kRank;
        const int coordinate = index % kRank;
        const int vector = layout.vector(target, coordinate);
        const int dual0 = layout.dual(target, 0, coordinate);
        const int dual1 = layout.dual(target, 1, coordinate);
        const float scale = static_cast<float>(at::BFloat16(diagonal[
            (panel * kChunk + target) * kRank + coordinate]));
        const float primal = upper_primal[vector];
        const float direct0 = upper_dual_output[dual0];
        const float direct1 = upper_dual_output[dual1];
        upper_primal[vector] = primal / scale;
        upper_dual_output[dual0] = direct0 * scale;
        upper_dual_output[dual1] = direct1 * scale;
        grad_log_diagonal[vector] =
            -primal * (static_cast<float>(lower_primal[vector]) / scale)
            + static_cast<float>(lower_dual_scaled[dual0]) * direct0
            + static_cast<float>(lower_dual_scaled[dual1]) * direct1;
        const int descriptor_base =
            (((1 * panels + panel) * kChunk + target) * 3) * kRank
            + coordinate;
        descriptor_bundle[descriptor_base] = lower_primal[vector];
        descriptor_bundle[descriptor_base + kRank] = at::BFloat16(
            upper_dual_output[dual0]);
        descriptor_bundle[descriptor_base + 2 * kRank] = at::BFloat16(
            upper_dual_output[dual1]);
    }
    __syncthreads();

    const Fp32VectorActionView lower_primal_input{upper_primal, layout};
    const Fp32DualActionView lower_dual_view{
        upper_dual_output, layout};

    for (int index = threadIdx.x;
         index < valid_count * kRank;
         index += blockDim.x) {
        const int target = index / kRank;
        const int coordinate = index % kRank;
        const int vector = layout.vector(target, coordinate);
        const int descriptor_base =
            ((panel * kChunk + target) * 3) * kRank + coordinate;
        descriptor_bundle[descriptor_base] = at::BFloat16(-upper_primal[vector]);
        descriptor_bundle[descriptor_base + kRank] = at::BFloat16(
            static_cast<float>(erase[vector]) * static_cast<float>(key[vector]));
        descriptor_bundle[descriptor_base + 2 * kRank] = query[vector];
    }
    __syncthreads();

    const FinalGradientDualOutput final_gradient{
        key,
        erase,
        upper_primal,
        grad_key,
        grad_erase,
        grad_query,
        layout};
    solvedelta::paired_action::run_paired_action<false, true>(
        shared,
        geometry,
        lower_primal_input,
        lower_primal_input,
        lower_dual_view,
        final_gradient);
}

void check_fp32_cuda_contiguous(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(
        tensor.get_device() == reference.get_device(),
        name,
        " must share one CUDA device");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be FP32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

C32PrepareForwardResult c32_solvedelta_prepare_forward_cuda(
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& geometry_log_decay,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& geometry_strength,
    const at::Tensor& boundary_m,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& inclusive_decay,
    const at::Tensor& write,
    const at::Tensor& value) {
    TORCH_CHECK(
        u.is_cuda() && u.scalar_type() == at::kBFloat16 && u.is_contiguous(),
        "u must be contiguous BF16 CUDA");
    TORCH_CHECK(
        u.dim() == 4 && u.size(3) == kRank,
        "u must be [B,T,H,128]");
    const int64_t batch = u.size(0);
    const int64_t length = u.size(1);
    const int64_t heads = u.size(2);
    TORCH_CHECK(batch > 0 && length > 0 && heads > 0,
                "B, T, and H must be positive");
    const int64_t chunks = (length - 1) / kChunk + 1;
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{&h, "h"},
             {&key, "key"},
             {&erase, "erase"},
             {&query, "query"}}) {
        TORCH_CHECK(
            named.first->is_cuda()
                && named.first->get_device() == u.get_device()
                && named.first->scalar_type() == at::kBFloat16
                && named.first->is_contiguous(),
            named.second,
            " must be contiguous BF16 on the u device");
    }
    TORCH_CHECK(h.sizes() == u.sizes() && query.sizes() == u.sizes(),
                "h/query shape mismatch");
    TORCH_CHECK(
        key.sizes() == at::IntArrayRef({batch, length, heads, 1, kRank})
            && erase.sizes() == key.sizes(),
        "key/erase must be [B,T,H,1,128]");
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{
                 &geometry_log_decay, "geometry_log_decay"},
             {&geometry_strength, "geometry_strength"},
             {&boundary_m, "boundary_m"},
             {&boundary_j, "boundary_J"},
             {&boundary_d, "boundary_D"},
             {&inclusive_decay, "inclusive_decay"}}) {
        check_fp32_cuda_contiguous(*named.first, u, named.second);
    }
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{&write, "write"},
             {&value, "value"}}) {
        TORCH_CHECK(
            named.first->is_cuda()
                && named.first->get_device() == u.get_device()
                && named.first->scalar_type() == at::kBFloat16
                && named.first->is_contiguous(),
            named.second,
            " must be contiguous BF16 on the u device");
    }
    TORCH_CHECK(
        geometry_log_decay.sizes()
            == at::IntArrayRef({batch, length, heads}),
        "geometry_log_decay must be [B,T,H]");
    TORCH_CHECK(
        inclusive_decay.sizes() == u.sizes(),
        "inclusive_decay must be [B,T,H,128]");
    TORCH_CHECK(
        write.dim() == 5 && write.size(3) == 1
            && value.sizes() == write.sizes(),
        "write/value must be [B,T,H,1,d_v]");
    const int64_t value_dim = write.size(4);
    TORCH_CHECK(
        value_dim > 0 && value_dim <= kRank,
        "native d_v must be in [1,128]");
    TORCH_CHECK(
        geometry_strength.sizes() == at::IntArrayRef({heads}),
        "geometry_strength must be [H]");
    TORCH_CHECK(
        boundary_m.sizes() == at::IntArrayRef({batch, heads, chunks}),
        "boundary_m must be [B,H,N]");
    TORCH_CHECK(
        boundary_j.sizes()
                == at::IntArrayRef(
                    {batch, heads, chunks, kRank, kRank})
            && boundary_d.sizes() == boundary_j.sizes(),
        "boundary_J/D must be [B,H,N,128,128]");
    constexpr int64_t max_index = std::numeric_limits<int>::max();
    TORCH_CHECK(
        length <= max_index && heads <= max_index && chunks <= max_index,
        "length, heads, and chunks must fit the resident int32 launch ABI");
    TORCH_CHECK(
        boundary_m.numel() <= max_index,
        "panel count must fit the resident int32 index space");
    TORCH_CHECK(
        u.numel() <= max_index / kDualRhs,
        "vector tensors exceed the resident int32 index space");
    TORCH_CHECK(
        boundary_j.numel() <= max_index,
        "boundary matrices exceed the resident int32 index space");

    c10::cuda::CUDAGuard guard(u.device());
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, u.get_device()));
    TORCH_CHECK(
        properties.major == 12 && properties.minor == 0,
        "the resident mixed frame contains only the SM120 specialization; got SM",
        properties.major,
        properties.minor);
    const int64_t panels64 = boundary_m.numel();
    const int panels = static_cast<int>(panels64);
    const auto fp32_options = u.options().dtype(at::kFloat);
    auto d = at::empty_like(query);
    auto e = at::empty_like(query);
    auto chi = at::empty_like(query);
    auto lower_primal = at::empty_like(u);
    auto lower_dual_scaled = at::empty(
        {batch, length, heads, kDualRhs, kRank}, u.options());
    auto inverse_mass = at::empty(
        {batch, heads, chunks, kChunk}, fp32_options);
    auto coefficient = at::empty(
        {panels64, kChunk, kComponents}, fp32_options);
    auto radial_q2 = at::empty_like(coefficient);
    auto radial_norm = at::empty_like(coefficient);
    auto diagonal = at::empty(
        {panels64, kChunk, kRank}, fp32_options);
    auto alpha0 = at::empty({panels64}, fp32_options);
    auto W = at::empty(
        {batch, heads, chunks, kChunk, kChunk}, fp32_options);
    auto A_qd = at::empty_like(W);
    auto Q_gamma = at::empty_like(u);
    auto D_tail = at::empty_like(u);
    auto G_last = at::empty(
        {batch, heads, chunks, kRank}, fp32_options);
    auto Y = at::empty_like(u);
    auto U_z = at::empty(
        {batch, length, heads, value_dim}, u.options());
    const auto stream = at::cuda::getCurrentCUDAStream();

    radial_kernel<at::BFloat16><<<dim3(panels, 2), kThreads, 0, stream>>>(
        u.data_ptr<at::BFloat16>(),
        h.data_ptr<at::BFloat16>(),
        geometry_log_decay.data_ptr<float>(),
        geometry_strength.data_ptr<float>(),
        boundary_m.data_ptr<float>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        alpha0.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        coefficient.data_ptr<float>(),
        radial_q2.data_ptr<float>(),
        radial_norm.data_ptr<float>(),
        diagonal.data_ptr<float>(),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        mixed_frame_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sizeof(solvedelta::paired_action::PairedActionShared))));
    mixed_frame_kernel<<<
        panels,
        kThreads,
        sizeof(solvedelta::paired_action::PairedActionShared),
        stream>>>(
        u.data_ptr<at::BFloat16>(),
        h.data_ptr<at::BFloat16>(),
        key.data_ptr<at::BFloat16>(),
        erase.data_ptr<at::BFloat16>(),
        query.data_ptr<at::BFloat16>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        alpha0.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        coefficient.data_ptr<float>(),
        diagonal.data_ptr<float>(),
        d.data_ptr<at::BFloat16>(),
        e.data_ptr<at::BFloat16>(),
        chi.data_ptr<at::BFloat16>(),
        lower_primal.data_ptr<at::BFloat16>(),
        lower_dual_scaled.data_ptr<at::BFloat16>(),
        inclusive_decay.data_ptr<float>(),
        write.data_ptr<at::BFloat16>(),
        value.data_ptr<at::BFloat16>(),
        W.data_ptr<float>(),
        A_qd.data_ptr<float>(),
        Q_gamma.data_ptr<at::BFloat16>(),
        D_tail.data_ptr<at::BFloat16>(),
        G_last.data_ptr<float>(),
        Y.data_ptr<at::BFloat16>(),
        U_z.data_ptr<at::BFloat16>(),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        static_cast<int>(value_dim),
        panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        W,
        A_qd,
        Q_gamma,
        D_tail,
        G_last,
        Y,
        U_z,
        d,
        e,
        chi,
        lower_primal,
        lower_dual_scaled,
        inverse_mass,
        coefficient,
        radial_q2,
        radial_norm,
        diagonal,
        alpha0};
}


C32FrameActionsBackwardResult c32_frame_actions_backward_cuda(
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& lower_primal,
    const at::Tensor& lower_dual_scaled,
    const at::Tensor& d,
    const at::Tensor& inverse_mass,
    const at::Tensor& coefficient,
    const at::Tensor& diagonal,
    const at::Tensor& alpha0,
    const at::Tensor& grad_d,
    const at::Tensor& grad_e,
    const at::Tensor& grad_chi) {
    TORCH_CHECK(
        u.is_cuda() && u.scalar_type() == at::kBFloat16 && u.is_contiguous(),
        "u must be contiguous BF16 CUDA");
    TORCH_CHECK(
        u.dim() == 4 && u.size(3) == kRank,
        "u must be [B,T,H,128]");
    const int64_t batch = u.size(0);
    const int64_t length = u.size(1);
    const int64_t heads = u.size(2);
    TORCH_CHECK(batch > 0 && length > 0 && heads > 0,
                "B, T, and H must be positive");
    const int64_t chunks = (length - 1) / kChunk + 1;
    const int64_t panels64 = batch * heads * chunks;
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{&h, "h"},
             {&key, "key"},
             {&erase, "erase"},
             {&query, "query"},
             }) {
        TORCH_CHECK(
            named.first->is_cuda()
                && named.first->get_device() == u.get_device()
                && named.first->scalar_type() == at::kBFloat16
                && named.first->is_contiguous(),
            named.second,
            " must be contiguous BF16 on the u device");
    }
    TORCH_CHECK(h.sizes() == u.sizes() && query.sizes() == u.sizes(),
                "h/query shape mismatch");
    TORCH_CHECK(
        key.sizes()
                == at::IntArrayRef({batch, length, heads, 1, kRank})
            && erase.sizes() == key.sizes(),
        "key/erase must be [B,T,H,1,128]");
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{
                 &boundary_j, "boundary_J"},
             {&boundary_d, "boundary_D"},
             {&inverse_mass, "inverse_mass"},
             {&coefficient, "radial_scale"},
             {&diagonal, "diagonal"},
             {&alpha0, "alpha0"},
             {&grad_d, "grad_d"},
             {&grad_e, "grad_e"},
             {&grad_chi, "grad_chi"}}) {
        check_fp32_cuda_contiguous(*named.first, u, named.second);
    }
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{
                 &lower_primal, "lower_primal"},
             {&lower_dual_scaled, "lower_dual_scaled"},
             {&d, "d"}}) {
        TORCH_CHECK(
            named.first->is_cuda()
                && named.first->get_device() == u.get_device()
                && named.first->scalar_type() == at::kBFloat16
                && named.first->is_contiguous(),
            named.second,
            " must be contiguous BF16 on the u device");
    }
    TORCH_CHECK(
        boundary_j.sizes()
                == at::IntArrayRef(
                    {batch, heads, chunks, kRank, kRank})
            && boundary_d.sizes() == boundary_j.sizes(),
        "boundary_J/D must be [B,H,N,128,128]");
    TORCH_CHECK(
        lower_primal.sizes() == u.sizes(),
        "lower_primal must be [B,T,H,128]");
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{
                 &d, "d"},
             {&grad_d, "grad_d"},
             {&grad_e, "grad_e"},
             {&grad_chi, "grad_chi"}}) {
        TORCH_CHECK(
            named.first->sizes() == u.sizes(),
            named.second,
            " must be [B,T,H,128]");
    }
    TORCH_CHECK(
        lower_dual_scaled.sizes()
                == at::IntArrayRef(
                    {batch, length, heads, kDualRhs, kRank}),
        "lower_dual_scaled must be [B,T,H,2,128]");
    TORCH_CHECK(
        inverse_mass.sizes()
                == at::IntArrayRef({batch, heads, chunks, kChunk}),
        "inverse_mass must be [B,H,N,32]");
    TORCH_CHECK(
        coefficient.sizes()
                == at::IntArrayRef({panels64, kChunk, kComponents}),
        "radial_scale must be [P,32,4]");
    TORCH_CHECK(
        diagonal.sizes()
                == at::IntArrayRef({panels64, kChunk, kRank}),
        "diagonal must be [P,32,128]");
    TORCH_CHECK(
        alpha0.sizes() == at::IntArrayRef({panels64}),
        "alpha0 must be [P]");
    constexpr int64_t max_index = std::numeric_limits<int>::max();
    TORCH_CHECK(
        length <= max_index && heads <= max_index && chunks <= max_index
            && panels64 <= max_index,
        "resident action dimensions must fit int32");
    TORCH_CHECK(
        u.numel() <= max_index / kDualRhs
            && boundary_j.numel() <= max_index,
        "resident action tensors exceed the int32 index space");

    c10::cuda::CUDAGuard guard(u.device());
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, u.get_device()));
    TORCH_CHECK(
        properties.major == 12 && properties.minor == 0,
        "the resident action backward contains only the SM120 specialization; got SM",
        properties.major,
        properties.minor);
    const int panels = static_cast<int>(panels64);
    auto grad_key = at::empty_like(key);
    auto grad_erase = at::empty_like(erase);
    auto grad_query = at::empty_like(u);
    const auto fp32_options = u.options().dtype(at::kFloat);
    auto upper_primal = at::empty(u.sizes(), fp32_options);
    auto upper_dual_output = at::empty(
        {batch, length, heads, kDualRhs, kRank}, fp32_options);
    auto grad_log_diagonal = at::empty_like(upper_primal);
    auto descriptor_bundle = at::empty(
        {4, panels64, kChunk, 3, kRank}, u.options());
    const auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        mixed_frame_adjoint_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sizeof(solvedelta::paired_action::PairedActionShared))));
    mixed_frame_adjoint_kernel<<<
        panels,
        kThreads,
        sizeof(solvedelta::paired_action::PairedActionShared),
        stream>>>(
        u.data_ptr<at::BFloat16>(),
        h.data_ptr<at::BFloat16>(),
        key.data_ptr<at::BFloat16>(),
        erase.data_ptr<at::BFloat16>(),
        query.data_ptr<at::BFloat16>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        lower_primal.data_ptr<at::BFloat16>(),
        lower_dual_scaled.data_ptr<at::BFloat16>(),
        d.data_ptr<at::BFloat16>(),
        inverse_mass.data_ptr<float>(),
        coefficient.data_ptr<float>(),
        diagonal.data_ptr<float>(),
        alpha0.data_ptr<float>(),
        grad_d.data_ptr<float>(),
        grad_e.data_ptr<float>(),
        grad_chi.data_ptr<float>(),
        grad_key.data_ptr<at::BFloat16>(),
        grad_erase.data_ptr<at::BFloat16>(),
        grad_query.data_ptr<at::BFloat16>(),
        upper_primal.data_ptr<float>(),
        upper_dual_output.data_ptr<float>(),
        grad_log_diagonal.data_ptr<float>(),
        descriptor_bundle.data_ptr<at::BFloat16>(),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_key,
        grad_erase,
        grad_query,
        grad_log_diagonal,
        descriptor_bundle};
}
