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

__device__ __forceinline__ void reduce_warp(float& value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
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
    const at::Half* u_data;
    const at::BFloat16* h_data;
    const float* boundary_j_data;
    const float* boundary_d_data;
    const float* inverse_mass_data;
    const float* coefficient_data;
    const float* alpha0_data;
    ActionPanelLayout layout;
    int panel;

    __device__ __forceinline__ at::Half u(
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
    const at::Half* key;
    ActionPanelLayout layout;

    __device__ __forceinline__ float load(
        int target,
        int,
        int coordinate) const {
        return static_cast<float>(key[layout.vector(target, coordinate)]);
    }
};

struct OriginalDualActionInput {
    const at::Half* key;
    const at::BFloat16* erase;
    const at::Half* query;
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

struct Fp16VectorActionView {
    at::Half* data;
    ActionPanelLayout layout;

    __device__ __forceinline__ void store(
        int target,
        int,
        int coordinate,
        float value) const {
        data[layout.vector(target, coordinate)] = at::Half(value);
    }
};

struct DiagonalPrimalActionInput {
    const at::Half* data;
    const float* diagonal;
    ActionPanelLayout layout;
    int panel;

    __device__ __forceinline__ float load(
        int target,
        int,
        int coordinate) const {
        const float scale = diagonal[
            (panel * kChunk + target) * kRank + coordinate];
        return static_cast<float>(data[layout.vector(target, coordinate)]) / scale;
    }
};

struct ScaledDualActionOutput {
    at::Half* data;
    const float* diagonal;
    ActionPanelLayout layout;
    int panel;

    __device__ __forceinline__ void store(
        int target,
        int route,
        int coordinate,
        float value) const {
        const float scale = diagonal[
            (panel * kChunk + target) * kRank + coordinate];
        data[layout.dual(target, route, coordinate)] =
            at::Half(value * scale);
    }
};

struct Fp32DualActionView {
    float* data;
    ActionPanelLayout layout;

    __device__ __forceinline__ void store(
        int target,
        int route,
        int coordinate,
        float value) const {
        data[layout.dual(target, route, coordinate)] = value;
    }
};

struct Fp16DualActionView {
    at::Half* data;
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
    at::Half* erase_dual;
    at::Half* query_dual;
    ActionPanelLayout layout;

    __device__ __forceinline__ void store(
        int target,
        int route,
        int coordinate,
        float value) const {
        const int index = layout.vector(target, coordinate);
        if (route == 0) {
            erase_dual[index] = at::Half(value);
        } else {
            query_dual[index] = at::Half(value);
        }
    }
};

struct GradPrimalActionInput {
    const float* data;
    ActionPanelLayout layout;

    __device__ __forceinline__ float load(
        int target,
        int,
        int coordinate) const {
        const int index = layout.vector(target, coordinate);
        return data[index];
    }
};

struct GradDualActionInput {
    const float* data;
    ActionPanelLayout layout;

    __device__ __forceinline__ float load(
        int target,
        int route,
        int coordinate) const {
        return data[layout.dual(target, route, coordinate)];
    }
};

struct FinalGradientDualOutput {
    const at::Half* key;
    const at::BFloat16* erase;
    const float* lower_rhs;
    float* grad_key;
    at::BFloat16* grad_erase;
    float* grad_query;
    ActionPanelLayout layout;

    __device__ __forceinline__ void store(
        int target,
        int route,
        int coordinate,
        float value) const {
        const int index = layout.vector(target, coordinate);
        if (route == 0) {
            grad_key[index] = lower_rhs[index]
                + static_cast<float>(erase[index]) * value;
            grad_erase[index] = at::BFloat16(
                static_cast<float>(key[index]) * value);
        } else {
            grad_query[index] = value;
        }
    }
};

__global__ __launch_bounds__(kThreads, 2) void mixed_frame_kernel(
    const at::Half* __restrict__ u,
    const at::BFloat16* __restrict__ h,
    const at::Half* __restrict__ key,
    const at::BFloat16* __restrict__ erase,
    const at::Half* __restrict__ query,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ alpha0,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ coefficient,
    const float* __restrict__ diagonal,
    at::Half* __restrict__ d,
    at::Half* __restrict__ e,
    at::Half* __restrict__ chi,
    at::Half* __restrict__ lower_primal,
    at::Half* __restrict__ lower_dual_scaled,
    const float* __restrict__ inclusive_decay,
    float* __restrict__ W,
    float* __restrict__ A_qd,
    at::Half* __restrict__ Q_gamma,
    at::Half* __restrict__ D_tail,
    float* __restrict__ G_last,
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
    const Fp16VectorActionView lower_primal_view{lower_primal, layout};
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
    const Fp16VectorActionView final_primal{d, layout};
    const Fp16DualActionView upper_dual_input{
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
        Q_gamma[index] = at::Half(
            static_cast<float>(chi[index]) * expf(gate));
        D_tail[index] = at::Half(
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
}

template <bool Upper>
__device__ __forceinline__ float strict_left(
    int route,
    int target,
    int coordinate,
    const ActionPanelLayout& layout,
    const at::Half* key,
    const at::BFloat16* erase,
    const at::Half* query,
    const at::Half* lower_dual_scaled,
    const float* action_primal) {
    const int vector = layout.vector(target, coordinate);
    if constexpr (Upper) {
        return route == 0
            ? -action_primal[vector]
            : static_cast<float>(lower_dual_scaled[
                layout.dual(target, route - 1, coordinate)]);
    }
    if (route == 0) return -action_primal[vector];
    if (route == 1) {
        return static_cast<float>(erase[vector])
            * static_cast<float>(key[vector]);
    }
    return static_cast<float>(query[vector]);
}

template <bool Upper>
__device__ __forceinline__ float strict_right(
    int route,
    int target,
    int coordinate,
    const ActionPanelLayout& layout,
    const at::Half* d,
    const at::Half* lower_primal,
    const float* original_dual,
    const float* action_dual) {
    if constexpr (Upper) {
        return route == 0
            ? static_cast<float>(d[layout.vector(target, coordinate)])
            : original_dual[layout.dual(target, route - 1, coordinate)];
    }
    return route == 0
        ? static_cast<float>(lower_primal[
            layout.vector(target, coordinate)])
        : action_dual[layout.dual(target, route - 1, coordinate)];
}

__global__ __launch_bounds__(kThreads, 2)
void upper_frame_adjoint_kernel(
    const at::Half* __restrict__ u,
    const at::BFloat16* __restrict__ h,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ coefficient,
    const float* __restrict__ alpha0,
    const float* __restrict__ frame_primal,
    const float* __restrict__ frame_dual,
    float* __restrict__ upper_primal,
    float* __restrict__ upper_dual_output,
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
    const GradPrimalActionInput grad_primal_input{frame_primal, layout};
    const Fp32VectorActionView upper_primal_view{
        upper_primal, layout};
    const GradDualActionInput grad_dual_input{frame_dual, layout};
    const Fp32DualActionView upper_dual_view{
        upper_dual_output, layout};
    solvedelta::paired_action::run_paired_action<true, true>(
        shared,
        geometry,
        grad_primal_input,
        upper_primal_view,
        grad_dual_input,
        upper_dual_view);
}

__global__ void diagonal_frame_adjoint_kernel(
    const at::Half* __restrict__ lower_primal,
    const at::Half* __restrict__ lower_dual_scaled,
    const float* __restrict__ diagonal,
    float* __restrict__ upper_primal,
    float* __restrict__ upper_dual,
    float* __restrict__ grad_log_diagonal,
    int length,
    int heads,
    int chunks,
    int panels) {
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    const ActionPanelLayout layout{
        batch_index, head, token_start, valid_count, length, heads};

    for (int index = threadIdx.x; index < valid_count * kRank;
         index += blockDim.x) {
        const int target = index / kRank;
        const int coordinate = index % kRank;
        const int vector = layout.vector(target, coordinate);
        const int dual0 = layout.dual(target, 0, coordinate);
        const int dual1 = layout.dual(target, 1, coordinate);
        const float scale = diagonal[
            (panel * kChunk + target) * kRank + coordinate];
        const float primal = upper_primal[vector];
        const float direct0 = upper_dual[dual0];
        const float direct1 = upper_dual[dual1];
        upper_primal[vector] = primal / scale;
        upper_dual[dual0] = direct0 * scale;
        upper_dual[dual1] = direct1 * scale;
        grad_log_diagonal[vector] =
            -primal * (static_cast<float>(lower_primal[vector]) / scale)
            + static_cast<float>(lower_dual_scaled[dual0]) * direct0
            + static_cast<float>(lower_dual_scaled[dual1]) * direct1;
    }
}

__global__ __launch_bounds__(kThreads, 2)
void lower_frame_adjoint_kernel(
    const at::Half* __restrict__ u,
    const at::BFloat16* __restrict__ h,
    const at::Half* __restrict__ key,
    const at::BFloat16* __restrict__ erase,
    const float* __restrict__ boundary_j,
    const float* __restrict__ boundary_d,
    const float* __restrict__ inverse_mass,
    const float* __restrict__ coefficient,
    const float* __restrict__ alpha0,
    float* __restrict__ upper_primal,
    const float* __restrict__ upper_dual,
    float* __restrict__ grad_key,
    at::BFloat16* __restrict__ grad_erase,
    float* __restrict__ grad_query,
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
        batch_index, head, token_start, valid_count, length, heads};
    const ActionGeometryView geometry{
        u, h, boundary_j, boundary_d, inverse_mass, coefficient,
        alpha0, layout, panel};

    const Fp32VectorActionView lower_primal_input{upper_primal, layout};
    const GradDualActionInput lower_dual_view{upper_dual, layout};

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

template <bool Upper>
__global__ __launch_bounds__(256)
void strict_boundary_matrix_stream_kernel(
    const at::Half* __restrict__ key,
    const at::BFloat16* __restrict__ erase,
    const at::Half* __restrict__ query,
    const at::Half* __restrict__ d,
    const at::Half* __restrict__ lower_primal,
    const at::Half* __restrict__ lower_dual_scaled,
    const float* __restrict__ original_dual,
    const float* __restrict__ action_primal,
    const float* __restrict__ action_dual,
    const float* __restrict__ theta,
    const float* __restrict__ coefficient,
    float* __restrict__ grad_boundary_j,
    float* __restrict__ grad_boundary_d,
    int length,
    int heads,
    int chunks,
    int panels) {
    const int panel = blockIdx.x;
    const int tile_pair = blockIdx.y;
    if (panel >= panels) return;
    const int row_block = tile_pair / (kRank / kTile);
    const int column_block = tile_pair % (kRank / kTile);
    const int local = threadIdx.x;
    const int row = row_block * kTile + local / kTile;
    const int column = column_block * kTile + local % kTile;
    if (!((Upper && row < column) || (!Upper && row > column))) return;
    int batch_index;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch_index, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    const ActionPanelLayout layout{
        batch_index, head, token_start, valid_count, length, heads};
    float result_j = 0.0f;
    float result_d = 0.0f;
    for (int target = 0; target < valid_count; ++target) {
        const int scalar = panel * kChunk + target;
        const int component = Upper ? 2 : 0;
        const float theta_j = theta[scalar]
            * coefficient[scalar * kComponents + component];
        const float theta_d = theta[scalar]
            * coefficient[scalar * kComponents + component + 1];
#pragma unroll
        for (int route = 0; route < 3; ++route) {
            const float product = strict_left<Upper>(
                route, target, row, layout,
                key, erase, query, lower_dual_scaled, action_primal)
                * strict_right<Upper>(
                    route, target, column, layout,
                    d, lower_primal, original_dual, action_dual);
            result_j = fmaf(theta_j, product, result_j);
            result_d = fmaf(theta_d, product, result_d);
        }
    }
    const int output = (panel * kRank + row) * kRank + column;
    atomicAdd(grad_boundary_j + output, result_j);
    atomicAdd(grad_boundary_d + output, result_d);
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
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& inverse_mass,
    const at::Tensor& coefficient,
    const at::Tensor& diagonal,
    const at::Tensor& alpha0,
    const at::Tensor& inclusive_decay) {
    TORCH_CHECK(
        u.is_cuda() && u.scalar_type() == at::kHalf && u.is_contiguous(),
        "u must be contiguous FP16 CUDA");
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
             std::pair<const at::Tensor*, const char*>{&key, "key"},
             {&query, "query"}}) {
        TORCH_CHECK(
            named.first->is_cuda()
                && named.first->get_device() == u.get_device()
                && named.first->scalar_type() == at::kHalf
                && named.first->is_contiguous(),
            named.second,
            " must be contiguous FP16 on the u device");
    }
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{&h, "h"},
             {&erase, "erase"}}) {
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
                 &boundary_j, "boundary_J"},
             {&boundary_d, "boundary_D"},
             {&inverse_mass, "inverse_mass"},
             {&coefficient, "radial_scale"},
             {&diagonal, "diagonal"},
             {&alpha0, "alpha0"},
             {&inclusive_decay, "inclusive_decay"}}) {
        check_fp32_cuda_contiguous(*named.first, u, named.second);
    }
    TORCH_CHECK(
        inclusive_decay.sizes() == u.sizes(),
        "inclusive_decay must be [B,T,H,128]");
    TORCH_CHECK(
        boundary_j.sizes()
                == at::IntArrayRef(
                    {batch, heads, chunks, kRank, kRank})
            && boundary_d.sizes() == boundary_j.sizes(),
        "boundary_J/D must be [B,H,N,128,128]");
    const int64_t panels64 = batch * heads * chunks;
    TORCH_CHECK(
        inverse_mass.sizes() == at::IntArrayRef({panels64, kChunk}),
        "inverse_mass must be [P,32]");
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
        length <= max_index && heads <= max_index && chunks <= max_index,
        "length, heads, and chunks must fit the resident int32 launch ABI");
    TORCH_CHECK(
        panels64 <= max_index,
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
    const int panels = static_cast<int>(panels64);
    const auto fp32_options = u.options().dtype(at::kFloat);
    auto d = at::empty_like(query);
    auto e = at::empty_like(query);
    auto chi = at::empty_like(query);
    auto lower_primal = at::empty_like(u);
    auto lower_dual_scaled = at::empty(
        {batch, length, heads, kDualRhs, kRank}, u.options());
    auto W = at::empty(
        {batch, heads, chunks, kChunk, kChunk}, fp32_options);
    auto A_qd = at::empty_like(W);
    auto Q_gamma = at::empty_like(u);
    auto D_tail = at::empty_like(u);
    auto G_last = at::empty(
        {batch, heads, chunks, kRank}, fp32_options);
    const auto stream = at::cuda::getCurrentCUDAStream();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        mixed_frame_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sizeof(solvedelta::paired_action::PairedActionShared))));
    mixed_frame_kernel<<<
        panels,
        kThreads,
        sizeof(solvedelta::paired_action::PairedActionShared),
        stream>>>(
        u.data_ptr<at::Half>(),
        h.data_ptr<at::BFloat16>(),
        key.data_ptr<at::Half>(),
        erase.data_ptr<at::BFloat16>(),
        query.data_ptr<at::Half>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        alpha0.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        coefficient.data_ptr<float>(),
        diagonal.data_ptr<float>(),
        d.data_ptr<at::Half>(),
        e.data_ptr<at::Half>(),
        chi.data_ptr<at::Half>(),
        lower_primal.data_ptr<at::Half>(),
        lower_dual_scaled.data_ptr<at::Half>(),
        inclusive_decay.data_ptr<float>(),
        W.data_ptr<float>(),
        A_qd.data_ptr<float>(),
        Q_gamma.data_ptr<at::Half>(),
        D_tail.data_ptr<at::Half>(),
        G_last.data_ptr<float>(),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        W,
        A_qd,
        Q_gamma,
        D_tail,
        G_last,
        d,
        e,
        chi,
        lower_primal,
        lower_dual_scaled};
}


namespace {

struct FrameBackwardShape {
    int batch;
    int length;
    int heads;
    int chunks;
    int panels;
};

FrameBackwardShape validate_frame_backward(
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
    const at::Tensor& theta,
    const at::Tensor& diagonal,
    const at::Tensor& alpha0,
    const at::Tensor& frame_primal,
    const at::Tensor& frame_dual) {
    TORCH_CHECK(
        u.is_cuda() && u.scalar_type() == at::kHalf && u.is_contiguous(),
        "u must be contiguous FP16 CUDA");
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
             std::pair<const at::Tensor*, const char*>{&key, "key"},
             {&query, "query"},
             }) {
        TORCH_CHECK(
            named.first->is_cuda()
                && named.first->get_device() == u.get_device()
                && named.first->scalar_type() == at::kHalf
                && named.first->is_contiguous(),
            named.second,
            " must be contiguous FP16 on the u device");
    }
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{&h, "h"},
             {&erase, "erase"}}) {
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
             {&theta, "theta"},
             {&diagonal, "diagonal"},
             {&alpha0, "alpha0"},
             {&frame_primal, "frame_primal"},
             {&frame_dual, "frame_dual"}}) {
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
                && named.first->scalar_type() == at::kHalf
                && named.first->is_contiguous(),
            named.second,
            " must be contiguous FP16 on the u device");
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
             {&frame_primal, "frame_primal"}}) {
        TORCH_CHECK(
            named.first->sizes() == u.sizes(),
            named.second,
            " must be [B,T,H,128]");
    }
    TORCH_CHECK(
        frame_dual.sizes()
            == at::IntArrayRef(
                {batch, length, heads, kDualRhs, kRank}),
        "frame_dual must be [B,T,H,2,128]");
    TORCH_CHECK(
        lower_dual_scaled.sizes()
                == at::IntArrayRef(
                    {batch, length, heads, kDualRhs, kRank}),
        "lower_dual_scaled must be [B,T,H,2,128]");
    TORCH_CHECK(
        inverse_mass.sizes()
                == at::IntArrayRef({panels64, kChunk}),
        "inverse_mass must be [P,32]");
    TORCH_CHECK(
        coefficient.sizes()
                == at::IntArrayRef({panels64, kChunk, kComponents}),
        "radial_scale must be [P,32,4]");
    TORCH_CHECK(
        theta.sizes() == at::IntArrayRef({panels64, kChunk}),
        "theta must be [P,32]");
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

    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, u.get_device()));
    TORCH_CHECK(
        properties.major == 12 && properties.minor == 0,
        "the resident action backward contains only the SM120 specialization; got SM",
        properties.major,
        properties.minor);
    return {
        static_cast<int>(batch),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        static_cast<int>(panels64)};
}

}  // namespace

C32FrameUpperBackwardResult c32_frame_upper_backward_cuda(
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
    const at::Tensor& theta,
    const at::Tensor& diagonal,
    const at::Tensor& alpha0,
    const at::Tensor& frame_primal,
    const at::Tensor& frame_dual) {
    const auto shape = validate_frame_backward(
        u, h, key, erase, query, boundary_j, boundary_d, lower_primal,
        lower_dual_scaled, d, inverse_mass, coefficient, theta, diagonal,
        alpha0, frame_primal, frame_dual);
    c10::cuda::CUDAGuard guard(u.device());
    const auto fp32_options = u.options().dtype(at::kFloat);
    auto upper_primal = at::empty_like(frame_primal);
    auto upper_dual = at::empty_like(frame_dual);
    auto grad_boundary_j = at::zeros_like(boundary_j);
    auto grad_boundary_d = at::zeros_like(boundary_d);
    const auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        upper_frame_adjoint_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sizeof(solvedelta::paired_action::PairedActionShared))));
    upper_frame_adjoint_kernel<<<
        shape.panels,
        kThreads,
        sizeof(solvedelta::paired_action::PairedActionShared),
        stream>>>(
        u.data_ptr<at::Half>(),
        h.data_ptr<at::BFloat16>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        coefficient.data_ptr<float>(),
        alpha0.data_ptr<float>(),
        frame_primal.data_ptr<float>(),
        frame_dual.data_ptr<float>(),
        upper_primal.data_ptr<float>(),
        upper_dual.data_ptr<float>(),
        shape.length,
        shape.heads,
        shape.chunks,
        shape.panels);

    const dim3 matrix_grid(
        shape.panels, (kRank / kTile) * (kRank / kTile));
    strict_boundary_matrix_stream_kernel<true><<<
        matrix_grid, 256, 0, stream>>>(
        key.data_ptr<at::Half>(), erase.data_ptr<at::BFloat16>(),
        query.data_ptr<at::Half>(), d.data_ptr<at::Half>(),
        lower_primal.data_ptr<at::Half>(),
        lower_dual_scaled.data_ptr<at::Half>(),
        frame_dual.data_ptr<float>(), upper_primal.data_ptr<float>(),
        upper_dual.data_ptr<float>(), theta.data_ptr<float>(),
        coefficient.data_ptr<float>(), grad_boundary_j.data_ptr<float>(),
        grad_boundary_d.data_ptr<float>(), shape.length,
        shape.heads, shape.chunks, shape.panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {upper_primal, upper_dual, grad_boundary_j, grad_boundary_d};
}

C32FrameLowerBackwardResult c32_frame_lower_backward_cuda(
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
    const at::Tensor& theta,
    const at::Tensor& diagonal,
    const at::Tensor& alpha0,
    const at::Tensor& upper_primal,
    const at::Tensor& upper_dual,
    const at::Tensor& grad_boundary_j,
    const at::Tensor& grad_boundary_d) {
    const auto shape = validate_frame_backward(
        u, h, key, erase, query, boundary_j, boundary_d, lower_primal,
        lower_dual_scaled, d, inverse_mass, coefficient, theta, diagonal,
        alpha0, upper_primal, upper_dual);
    check_fp32_cuda_contiguous(grad_boundary_j, u, "grad_boundary_J");
    check_fp32_cuda_contiguous(grad_boundary_d, u, "grad_boundary_D");
    TORCH_CHECK(
        grad_boundary_j.sizes() == boundary_j.sizes()
            && grad_boundary_d.sizes() == boundary_d.sizes(),
        "boundary gradient shape mismatch");
    c10::cuda::CUDAGuard guard(u.device());
    const auto fp32_options = u.options().dtype(at::kFloat);
    auto grad_key = at::empty(key.sizes(), fp32_options);
    auto grad_erase = at::empty_like(erase);
    auto grad_query = at::empty(u.sizes(), fp32_options);
    auto grad_log_diagonal = at::empty_like(upper_primal);
    const auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        lower_frame_adjoint_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sizeof(solvedelta::paired_action::PairedActionShared))));
    diagonal_frame_adjoint_kernel<<<shape.panels, 256, 0, stream>>>(
        lower_primal.data_ptr<at::Half>(),
        lower_dual_scaled.data_ptr<at::Half>(), diagonal.data_ptr<float>(),
        upper_primal.data_ptr<float>(), upper_dual.data_ptr<float>(),
        grad_log_diagonal.data_ptr<float>(), shape.length,
        shape.heads, shape.chunks, shape.panels);
    lower_frame_adjoint_kernel<<<
        shape.panels,
        kThreads,
        sizeof(solvedelta::paired_action::PairedActionShared),
        stream>>>(
        u.data_ptr<at::Half>(), h.data_ptr<at::BFloat16>(),
        key.data_ptr<at::Half>(), erase.data_ptr<at::BFloat16>(),
        boundary_j.data_ptr<float>(), boundary_d.data_ptr<float>(),
        inverse_mass.data_ptr<float>(), coefficient.data_ptr<float>(),
        alpha0.data_ptr<float>(), upper_primal.data_ptr<float>(),
        upper_dual.data_ptr<float>(), grad_key.data_ptr<float>(),
        grad_erase.data_ptr<at::BFloat16>(), grad_query.data_ptr<float>(),
        shape.length, shape.heads, shape.chunks, shape.panels);

    const dim3 matrix_grid(
        shape.panels, (kRank / kTile) * (kRank / kTile));
    strict_boundary_matrix_stream_kernel<false><<<
        matrix_grid, 256, 0, stream>>>(
        key.data_ptr<at::Half>(), erase.data_ptr<at::BFloat16>(),
        query.data_ptr<at::Half>(), d.data_ptr<at::Half>(),
        lower_primal.data_ptr<at::Half>(),
        lower_dual_scaled.data_ptr<at::Half>(),
        upper_dual.data_ptr<float>(), upper_primal.data_ptr<float>(),
        upper_dual.data_ptr<float>(), theta.data_ptr<float>(),
        coefficient.data_ptr<float>(), grad_boundary_j.data_ptr<float>(),
        grad_boundary_d.data_ptr<float>(), shape.length,
        shape.heads, shape.chunks, shape.panels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_key,
        grad_erase,
        grad_query,
        grad_log_diagonal,
        grad_boundary_j,
        grad_boundary_d,
        upper_dual,
        upper_primal};
}
