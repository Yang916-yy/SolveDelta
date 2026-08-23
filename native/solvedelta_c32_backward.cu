#include "solvedelta_c32.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <limits>


namespace {

constexpr int kRank = 128;
constexpr int kChunk = 32;
constexpr int kThreads = 256;
constexpr float kRadius = 0.125f;
constexpr int kPanelElements = kRank * kChunk;
constexpr int kScoreElements = kChunk * kChunk;
constexpr int kComponents = 4;

// Three wide vector panels are reused as factor-action scratch and then as the
// two diagonal chart cotangents. The 4*C*C area first holds masked moment
// Grams and is overwritten by the radial projection coupling coefficients.
constexpr int kOffsetX = 0;
constexpr int kOffsetY = kOffsetX + kPanelElements;
constexpr int kOffsetDiagonal = kOffsetY + kPanelElements;
constexpr int kOffsetWeights = kOffsetDiagonal + kPanelElements;
constexpr int kOffsetPair = kOffsetWeights + kScoreElements;
constexpr int kOffsetWeightBar = kOffsetPair + kComponents * kScoreElements;
constexpr int kOffsetLambda = kOffsetWeightBar + kScoreElements;
constexpr int kOffsetInverseMass = kOffsetLambda + kChunk;
constexpr int kOffsetBeta = kOffsetInverseMass + kChunk;
constexpr int kOffsetAlpha = kOffsetBeta + kChunk;
constexpr int kOffsetRadialA = kOffsetAlpha + kChunk;
constexpr int kOffsetRadialQ2 = kOffsetRadialA + kComponents * kChunk;
constexpr int kOffsetRadialP = kOffsetRadialQ2 + kComponents * kChunk;
constexpr int kOffsetZeta = kOffsetRadialP + kComponents * kChunk;
constexpr int kOffsetABoundary = kOffsetZeta + kComponents * kChunk;
constexpr int kOffsetBoundarySource =
    kOffsetABoundary + kComponents * kChunk;
constexpr int kOffsetBoundaryNorm =
    kOffsetBoundarySource + kComponents * kChunk;
constexpr int kOffsetAlphaBar = kOffsetBoundaryNorm + kComponents;
constexpr int kOffsetProjectionBoundary = kOffsetAlphaBar + kChunk;
constexpr int kOffsetProjectionBoundarySelf =
    kOffsetProjectionBoundary + kComponents * kChunk;
constexpr int kOffsetReduction =
    kOffsetProjectionBoundarySelf + kComponents;
constexpr int kSharedFloats = kOffsetReduction + 8;

static_assert(kSharedFloats * sizeof(float) < 96 * 1024);


struct PanelContext {
    const float* u;
    const float* h;
    const float* key;
    const float* erase;
    const float* query;
    const float* lower_primal;
    const float* lower_dual_scaled;
    const float* write_direction;
    const float* grad_write;
    const float* grad_erase;
    const float* grad_query;
    const float* temporary;
    const float* factor_key_bar;
    const float* boundary_j;
    const float* boundary_d;
    int batch_index;
    int head;
    int token_start;
    int valid_count;
    int length;
    int heads;
    int panel;
    int vector_elements;
};


__device__ __forceinline__ int vector_base(
    const PanelContext& context,
    int local_token) {
    const int token = context.token_start + local_token;
    return ((context.batch_index * context.length + token) * context.heads
            + context.head)
        * kRank;
}


__device__ __forceinline__ const float* temporary_vector(
    const PanelContext& context,
    int component,
    int local_token) {
    return context.temporary + component * context.vector_elements
        + vector_base(context, local_token);
}


__device__ __forceinline__ const float* dual_saved_vector(
    const PanelContext& context,
    int rhs,
    int local_token) {
    const int token = context.token_start + local_token;
    return context.lower_dual_scaled
        + (((context.batch_index * context.length + token) * context.heads
            + context.head) * 2 + rhs) * kRank;
}


__device__ __forceinline__ float block_sum(float value, float* reduction) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) reduction[warp] = value;
    __syncthreads();
    float total = threadIdx.x < 8 ? reduction[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
        if (lane == 0) reduction[0] = total;
    }
    __syncthreads();
    return reduction[0];
}


template<bool Driven, bool Lower>
__device__ void compute_radial_parameters(
    const PanelContext& context,
    const float* lambda,
    const float* inverse_mass,
    float strength,
    float* radial_a,
    float* radial_q2,
    float* reduction) {
    float norm[kChunk];
#pragma unroll
    for (int target = 0; target < kChunk; ++target) norm[target] = 0.0f;

    const float* boundary = Driven ? context.boundary_d : context.boundary_j;
    const int boundary_base = context.panel * kRank * kRank;
    for (int index = threadIdx.x; index < kRank * kRank;
         index += blockDim.x) {
        const int row = index / kRank;
        const int column = index % kRank;
        const bool selected = Lower ? row > column : row < column;
        if (!selected) continue;
        float state = boundary[boundary_base + index];
#pragma unroll
        for (int target = 0; target < kChunk; ++target) {
            if (target >= context.valid_count) continue;
            const int source_base = vector_base(context, target);
            const float left = context.u[source_base + row];
            const float right = Driven
                ? context.h[source_base + column]
                : context.u[source_base + column];
            state = fmaf(left, right, lambda[target] * state);
            const float normalized = state * inverse_mass[target];
            norm[target] = fmaf(normalized, normalized, norm[target]);
        }
    }

#pragma unroll
    for (int target = 0; target < kChunk; ++target) {
        const float total = block_sum(norm[target], reduction);
        if (threadIdx.x == 0) {
            const float q2 = kRadius * kRadius
                + strength * strength * total;
            radial_q2[target] = q2;
            radial_a[target] = target < context.valid_count
                ? kRadius * strength * rsqrtf(q2)
                : 0.0f;
        }
    }
}


template<bool Lower, bool Transpose, bool Solve>
__device__ void apply_chart_factor(
    const PanelContext& context,
    float* x,
    float* y,
    const float* weights,
    const float* alpha,
    const float* radial_a) {
    const int target = threadIdx.x;
    if (target >= context.valid_count || target >= kChunk) return;

    float dot_u[kChunk];
    float dot_h[kChunk];
#pragma unroll
    for (int source = 0; source < kChunk; ++source) {
        dot_u[source] = 0.0f;
        dot_h[source] = 0.0f;
    }

    constexpr bool ascending = Lower != Transpose;
    constexpr int component = Lower ? 0 : 2;
    const int boundary_base = context.panel * kRank * kRank;

#pragma unroll
    for (int step = 0; step < kRank; ++step) {
        const int row = ascending ? step : kRank - 1 - step;
        float action = 0.0f;
        for (int previous = 0; previous < step; ++previous) {
            const int column = ascending ? previous : kRank - 1 - previous;
            const int matrix_index = Transpose
                ? column * kRank + row
                : row * kRank + column;
            const float entry =
                radial_a[(component + 0) * kChunk + target]
                    * context.boundary_j[boundary_base + matrix_index]
                + radial_a[(component + 1) * kChunk + target]
                    * context.boundary_d[boundary_base + matrix_index];
            action = fmaf(
                alpha[target] * entry,
                x[column * kChunk + target],
                action);
        }

        const float coefficient_h = radial_a[component * kChunk + target];
        const float coefficient_r = radial_a[(component + 1) * kChunk + target];
        for (int source = 0; source <= target; ++source) {
            const int source_base = vector_base(context, source);
            float local;
            if constexpr (Transpose) {
                local = (
                    coefficient_h * context.u[source_base + row]
                    + coefficient_r * context.h[source_base + row]
                ) * dot_u[source];
            } else {
                local = context.u[source_base + row] * (
                    coefficient_h * dot_u[source]
                    + coefficient_r * dot_h[source]);
            }
            action = fmaf(weights[source * kChunk + target], local, action);
        }

        const float rhs = x[row * kChunk + target];
        const float result = Solve ? rhs - action : rhs + action;
        if constexpr (Solve) x[row * kChunk + target] = result;
        else y[row * kChunk + target] = result;

        const float vector = Solve ? result : rhs;
        for (int source = 0; source <= target; ++source) {
            const int source_base = vector_base(context, source);
            dot_u[source] = fmaf(
                context.u[source_base + row], vector, dot_u[source]);
            if constexpr (!Transpose) {
                dot_h[source] = fmaf(
                    context.h[source_base + row], vector, dot_h[source]);
            }
        }
    }
}


__device__ __forceinline__ float a_left_value(
    const PanelContext& context,
    bool lower,
    int term,
    int target,
    int coordinate) {
    const int base = vector_base(context, target);
    if (lower) {
        if (term == 0) return -context.factor_key_bar[base + coordinate];
        if (term == 1) {
            return context.erase[base + coordinate]
                * context.key[base + coordinate];
        }
        return context.query[base + coordinate];
    }
    if (term == 0) return -temporary_vector(context, 0, target)[coordinate];
    if (term == 1) return dual_saved_vector(context, 0, target)[coordinate];
    return dual_saved_vector(context, 1, target)[coordinate];
}


__device__ __forceinline__ float a_right_value(
    const PanelContext& context,
    bool lower,
    int term,
    int target,
    int coordinate) {
    const int base = vector_base(context, target);
    if (lower) {
        if (term == 0) return context.lower_primal[base + coordinate];
        if (term == 1) return temporary_vector(context, 1, target)[coordinate];
        return temporary_vector(context, 2, target)[coordinate];
    }
    if (term == 0) return context.write_direction[base + coordinate];
    if (term == 1) return context.grad_erase[base + coordinate];
    return context.grad_query[base + coordinate];
}


__device__ float a_dense_inner(
    const PanelContext& context,
    int target,
    const float* dense,
    bool lower) {
    const int boundary_base = context.panel * kRank * kRank;
    float total = 0.0f;
    for (int term = 0; term < 3; ++term) {
        for (int row = 0; row < kRank; ++row) {
            const float left = a_left_value(
                context, lower, term, target, row);
            const int begin = lower ? 0 : row + 1;
            const int end = lower ? row : kRank;
            for (int column = begin; column < end; ++column) {
                total = fmaf(
                    left * a_right_value(
                        context, lower, term, target, column),
                    dense[boundary_base + row * kRank + column],
                    total);
            }
        }
    }
    return total;
}


__device__ float a_outer_inner(
    const PanelContext& context,
    int target,
    const float* outer_left,
    const float* outer_right,
    bool lower) {
    float total = 0.0f;
    for (int term = 0; term < 3; ++term) {
        float partial = 0.0f;
        if (lower) {
            for (int row = 0; row < kRank; ++row) {
                total = fmaf(
                    a_left_value(context, true, term, target, row)
                        * outer_left[row],
                    partial,
                    total);
                partial = fmaf(
                    a_right_value(context, true, term, target, row),
                    outer_right[row],
                    partial);
            }
        } else {
            for (int row = kRank - 1; row >= 0; --row) {
                total = fmaf(
                    a_left_value(context, false, term, target, row)
                        * outer_left[row],
                    partial,
                    total);
                partial = fmaf(
                    a_right_value(context, false, term, target, row),
                    outer_right[row],
                    partial);
            }
        }
    }
    return total;
}


__device__ float boundary_source_inner(
    const PanelContext& context,
    int component,
    int source) {
    const bool lower = component < 2;
    const bool driven = component & 1;
    const float* boundary = driven ? context.boundary_d : context.boundary_j;
    const int boundary_base = context.panel * kRank * kRank;
    const int source_base = vector_base(context, source);
    float total = 0.0f;
    for (int row = 0; row < kRank; ++row) {
        const float left = context.u[source_base + row];
        const int begin = lower ? 0 : row + 1;
        const int end = lower ? row : kRank;
        for (int column = begin; column < end; ++column) {
            const float right = driven
                ? context.h[source_base + column]
                : context.u[source_base + column];
            total = fmaf(
                boundary[boundary_base + row * kRank + column],
                left * right,
                total);
        }
    }
    return total;
}


__device__ float source_pair_inner(
    const PanelContext& context,
    int component,
    int left_source,
    int right_source) {
    const bool lower = component < 2;
    const bool driven = component & 1;
    const int left_base = vector_base(context, left_source);
    const int right_base = vector_base(context, right_source);
    float prefix = 0.0f;
    float total = 0.0f;
    if (lower) {
        for (int row = 0; row < kRank; ++row) {
            total = fmaf(
                context.u[left_base + row] * context.u[right_base + row],
                prefix,
                total);
            const float left_right = driven
                ? context.h[left_base + row]
                : context.u[left_base + row];
            const float right_right = driven
                ? context.h[right_base + row]
                : context.u[right_base + row];
            prefix = fmaf(left_right, right_right, prefix);
        }
    } else {
        for (int row = kRank - 1; row >= 0; --row) {
            total = fmaf(
                context.u[left_base + row] * context.u[right_base + row],
                prefix,
                total);
            const float left_right = driven
                ? context.h[left_base + row]
                : context.u[left_base + row];
            const float right_right = driven
                ? context.h[right_base + row]
                : context.u[right_base + row];
            prefix = fmaf(left_right, right_right, prefix);
        }
    }
    return total;
}


__device__ void add_a_action(
    const PanelContext& context,
    bool lower,
    bool transpose,
    int target,
    const float* vector,
    float scale,
    float* output) {
    for (int term = 0; term < 3; ++term) {
        float running = 0.0f;
        const bool ascending = lower != transpose;
        for (int step = 0; step < kRank; ++step) {
            const int row = ascending ? step : kRank - 1 - step;
            const float left = transpose
                ? a_right_value(context, lower, term, target, row)
                : a_left_value(context, lower, term, target, row);
            output[row] = fmaf(scale * left, running, output[row]);
            const float right = transpose
                ? a_left_value(context, lower, term, target, row)
                : a_right_value(context, lower, term, target, row);
            running = fmaf(right, vector[row], running);
        }
    }
}


__device__ void add_boundary_action(
    const float* boundary,
    bool lower,
    bool transpose,
    const float* vector,
    float scale,
    float* output) {
    const bool ascending = lower != transpose;
    for (int step = 0; step < kRank; ++step) {
        const int row = ascending ? step : kRank - 1 - step;
        float action = 0.0f;
        for (int previous = 0; previous < step; ++previous) {
            const int column = ascending ? previous : kRank - 1 - previous;
            const int index = transpose
                ? column * kRank + row
                : row * kRank + column;
            action = fmaf(boundary[index], vector[column], action);
        }
        output[row] = fmaf(scale, action, output[row]);
    }
}


__device__ void add_source_action(
    const PanelContext& context,
    bool driven,
    bool lower,
    bool transpose,
    int source,
    const float* vector,
    float scale,
    float* output) {
    const int source_base = vector_base(context, source);
    float running = 0.0f;
    const bool ascending = lower != transpose;
    for (int step = 0; step < kRank; ++step) {
        const int row = ascending ? step : kRank - 1 - step;
        const float primal = context.u[source_base + row];
        const float driven_value = driven
            ? context.h[source_base + row]
            : primal;
        const float left = transpose ? driven_value : primal;
        output[row] = fmaf(scale * left, running, output[row]);
        const float right = transpose ? primal : driven_value;
        running = fmaf(right, vector[row], running);
    }
}


__global__ __launch_bounds__(kThreads, 1) void c32_frame_backward_kernel(
    const float* u,
    const float* h,
    const float* geometry_log_decay,
    const float* key,
    const float* erase,
    const float* query,
    const float* geometry_strength,
    const float* boundary_m,
    const float* boundary_j,
    const float* boundary_d,
    const float* inverse_mass_saved,
    const float* lower_primal,
    const float* lower_dual_scaled,
    const float* write_direction,
    const float* grad_write,
    const float* grad_erase,
    const float* grad_query,
    float* temporary,
    float* grad_u,
    float* grad_h,
    float* grad_geometry_log_decay,
    float* grad_key,
    float* grad_erase_input,
    float* grad_query_input,
    float* strength_partial,
    float* grad_boundary_m,
    float* grad_boundary_j,
    float* grad_boundary_d,
    int batch,
    int length,
    int heads,
    int chunks,
    int vector_elements) {
    extern __shared__ float shared[];
    float* x = shared + kOffsetX;
    float* y = shared + kOffsetY;
    float* diagonal = shared + kOffsetDiagonal;
    float* weights = shared + kOffsetWeights;
    float* pair = shared + kOffsetPair;
    float* weight_bar = shared + kOffsetWeightBar;
    float* lambda = shared + kOffsetLambda;
    float* inverse_mass = shared + kOffsetInverseMass;
    float* beta = shared + kOffsetBeta;
    float* alpha = shared + kOffsetAlpha;
    float* radial_a = shared + kOffsetRadialA;
    float* radial_q2 = shared + kOffsetRadialQ2;
    float* radial_p = shared + kOffsetRadialP;
    float* zeta = shared + kOffsetZeta;
    float* a_boundary = shared + kOffsetABoundary;
    float* boundary_source = shared + kOffsetBoundarySource;
    float* boundary_norm = shared + kOffsetBoundaryNorm;
    float* alpha_bar = shared + kOffsetAlphaBar;
    float* projection_boundary = shared + kOffsetProjectionBoundary;
    float* projection_boundary_self =
        shared + kOffsetProjectionBoundarySelf;
    float* reduction = shared + kOffsetReduction;

    const int panel = blockIdx.x;
    const int chunk = panel % chunks;
    const int head_batch = panel / chunks;
    const int head = head_batch % heads;
    const int batch_index = head_batch / heads;
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, max(0, length - token_start));
    const int boundary_base = panel * kRank * kRank;
    const float strength = geometry_strength[head];

    PanelContext context{
        u, h, key, erase, query, lower_primal, lower_dual_scaled,
        write_direction, grad_write, grad_erase, grad_query, temporary,
        grad_key, boundary_j, boundary_d, batch_index, head, token_start,
        valid_count, length, heads, panel, vector_elements};

    if (threadIdx.x == 0) {
        for (int target = 0; target < kChunk; ++target) {
            if (target >= valid_count) {
                lambda[target] = 0.0f;
                inverse_mass[target] = 0.0f;
                beta[target] = 0.0f;
                alpha[target] = 0.0f;
                continue;
            }
            const int token = token_start + target;
            const int scalar_index =
                (batch_index * length + token) * heads + head;
            lambda[target] = expf(geometry_log_decay[scalar_index]);
            inverse_mass[target] = inverse_mass_saved[panel * kChunk + target];
            beta[target] = 1.0f - inverse_mass[target];
            alpha[target] = target == 0
                ? lambda[target] * inverse_mass[target]
                : beta[target] * alpha[target - 1];
            for (int source = 0; source < target; ++source) {
                weights[source * kChunk + target] =
                    beta[target] * weights[source * kChunk + target - 1];
            }
            weights[target * kChunk + target] = inverse_mass[target];
        }
    }
    for (int index = threadIdx.x; index < kScoreElements;
         index += blockDim.x) {
        const int source = index / kChunk;
        const int target = index % kChunk;
        if (source > target || target >= valid_count) weights[index] = 0.0f;
    }
    __syncthreads();

    compute_radial_parameters<false, true>(
        context, lambda, inverse_mass, strength,
        radial_a + 0 * kChunk, radial_q2 + 0 * kChunk, reduction);
    compute_radial_parameters<true, true>(
        context, lambda, inverse_mass, strength,
        radial_a + 1 * kChunk, radial_q2 + 1 * kChunk, reduction);
    compute_radial_parameters<false, false>(
        context, lambda, inverse_mass, strength,
        radial_a + 2 * kChunk, radial_q2 + 2 * kChunk, reduction);
    compute_radial_parameters<true, false>(
        context, lambda, inverse_mass, strength,
        radial_a + 3 * kChunk, radial_q2 + 3 * kChunk, reduction);

    if (threadIdx.x < kRank) {
        const int row = threadIdx.x;
        float state_j = boundary_j[boundary_base + row * kRank + row];
        float state_d = boundary_d[boundary_base + row * kRank + row];
        for (int target = 0; target < valid_count; ++target) {
            const int base = vector_base(context, target);
            state_j = fmaf(
                u[base + row], u[base + row], lambda[target] * state_j);
            state_d = fmaf(
                u[base + row], h[base + row], lambda[target] * state_d);
            const float x_h = strength * (
                state_j * inverse_mass[target]
                - 1.0f / static_cast<float>(kRank));
            const float x_r = strength * state_d * inverse_mass[target];
            diagonal[row * kChunk + target] = expf(
                kRadius * tanhf(x_h / kRadius)
                + kRadius * tanhf(x_r / kRadius));
        }
        for (int target = valid_count; target < kChunk; ++target) {
            diagonal[row * kChunk + target] = 1.0f;
        }
    }
    __syncthreads();

    // Primal path: U^-T grad_d, then L^-T D^-1 U^-T grad_d.
    for (int index = threadIdx.x; index < kPanelElements;
         index += blockDim.x) {
        const int row = index / kChunk;
        const int target = index % kChunk;
        x[index] = target < valid_count
            ? grad_write[vector_base(context, target) + row]
            : 0.0f;
    }
    __syncthreads();
    apply_chart_factor<false, true, true>(
        context, x, y, weights, alpha, radial_a);
    __syncthreads();
    for (int index = threadIdx.x; index < kPanelElements;
         index += blockDim.x) {
        const int row = index / kChunk;
        const int target = index % kChunk;
        if (target < valid_count) {
            temporary[0 * vector_elements + vector_base(context, target) + row]
                = x[index];
            x[index] /= diagonal[index];
        }
    }
    __syncthreads();
    apply_chart_factor<true, true, true>(
        context, x, y, weights, alpha, radial_a);
    __syncthreads();
    for (int index = threadIdx.x; index < kPanelElements;
         index += blockDim.x) {
        const int row = index / kChunk;
        const int target = index % kChunk;
        if (target < valid_count) {
            grad_key[vector_base(context, target) + row] = x[index];
        }
    }
    __syncthreads();

    // Two dual paths: U grad_output, D scaling, then L action.
    for (int rhs = 0; rhs < 2; ++rhs) {
        const float* output_gradient = rhs == 0 ? grad_erase : grad_query;
        for (int index = threadIdx.x; index < kPanelElements;
             index += blockDim.x) {
            const int row = index / kChunk;
            const int target = index % kChunk;
            x[index] = target < valid_count
                ? output_gradient[vector_base(context, target) + row]
                : 0.0f;
        }
        __syncthreads();
        apply_chart_factor<false, false, false>(
            context, x, y, weights, alpha, radial_a);
        __syncthreads();
        for (int index = threadIdx.x; index < kPanelElements;
             index += blockDim.x) {
            const int row = index / kChunk;
            const int target = index % kChunk;
            x[index] = diagonal[index] * y[index];
            if (target < valid_count) {
                temporary[(rhs + 1) * vector_elements
                    + vector_base(context, target) + row] = x[index];
            }
        }
        __syncthreads();
        apply_chart_factor<true, false, false>(
            context, x, y, weights, alpha, radial_a);
        __syncthreads();
        for (int index = threadIdx.x; index < kPanelElements;
             index += blockDim.x) {
            const int row = index / kChunk;
            const int target = index % kChunk;
            if (target >= valid_count) continue;
            const int base = vector_base(context, target);
            if (rhs == 0) {
                grad_erase_input[base + row] = y[index];
            } else {
                grad_query_input[base + row] = y[index];
            }
        }
        __syncthreads();
    }

    // Diagonal chart cotangents overwrite the no-longer-needed factor panels.
    float diagonal_strength = 0.0f;
    if (threadIdx.x < kRank) {
        const int row = threadIdx.x;
        float state_j = boundary_j[boundary_base + row * kRank + row];
        float state_d = boundary_d[boundary_base + row * kRank + row];
        for (int target = 0; target < valid_count; ++target) {
            const int base = vector_base(context, target);
            state_j = fmaf(
                u[base + row], u[base + row], lambda[target] * state_j);
            state_d = fmaf(
                u[base + row], h[base + row], lambda[target] * state_d);
            const float z_h = state_j * inverse_mass[target]
                - 1.0f / static_cast<float>(kRank);
            const float z_r = state_d * inverse_mass[target];
            const float tanh_h = tanhf(strength * z_h / kRadius);
            const float tanh_r = tanhf(strength * z_r / kRadius);
            const float inv_diagonal =
                1.0f / diagonal[row * kChunk + target];
            const float grad_log_diagonal =
                -temporary_vector(context, 0, target)[row]
                    * lower_primal[base + row] * inv_diagonal
                + dual_saved_vector(context, 0, target)[row]
                    * temporary_vector(context, 1, target)[row]
                    * inv_diagonal
                + dual_saved_vector(context, 1, target)[row]
                    * temporary_vector(context, 2, target)[row]
                    * inv_diagonal;
            const float derivative_h = 1.0f - tanh_h * tanh_h;
            const float derivative_r = 1.0f - tanh_r * tanh_r;
            x[row * kChunk + target] =
                strength * derivative_h * grad_log_diagonal;
            y[row * kChunk + target] =
                strength * derivative_r * grad_log_diagonal;
            diagonal_strength = fmaf(
                grad_log_diagonal,
                derivative_h * z_h + derivative_r * z_r,
                diagonal_strength);
        }
    }
    diagonal_strength = block_sum(diagonal_strength, reduction);

    // Masked boundary/source Grams needed by the radial projection and by the
    // alpha/weight scalar pullback.
    if (threadIdx.x < kComponents) {
        const int component = threadIdx.x;
        const bool lower = component < 2;
        const bool driven = component & 1;
        const float* boundary = driven ? boundary_d : boundary_j;
        float total = 0.0f;
        for (int row = 0; row < kRank; ++row) {
            const int begin = lower ? 0 : row + 1;
            const int end = lower ? row : kRank;
            for (int column = begin; column < end; ++column) {
                const float value =
                    boundary[boundary_base + row * kRank + column];
                total = fmaf(value, value, total);
            }
        }
        boundary_norm[component] = total;
    }
    for (int job = threadIdx.x; job < kComponents * kChunk;
         job += blockDim.x) {
        const int component = job / kChunk;
        const int source = job % kChunk;
        boundary_source[job] = source < valid_count
            ? boundary_source_inner(context, component, source)
            : 0.0f;
    }
    for (int job = threadIdx.x;
         job < kComponents * kScoreElements;
         job += blockDim.x) {
        const int component = job / kScoreElements;
        const int pair_index = job % kScoreElements;
        const int left_source = pair_index / kChunk;
        const int right_source = pair_index % kChunk;
        pair[job] = left_source < valid_count && right_source < valid_count
            ? source_pair_inner(
                context, component, left_source, right_source)
            : 0.0f;
    }
    __syncthreads();

    if (threadIdx.x < kChunk) {
        const int target = threadIdx.x;
        if (target < valid_count) {
            for (int component = 0; component < kComponents; ++component) {
                const bool lower = component < 2;
                const bool driven = component & 1;
                const float* boundary = driven ? boundary_d : boundary_j;
                const float boundary_inner =
                    a_dense_inner(context, target, boundary, lower);
                a_boundary[component * kChunk + target] = boundary_inner;
                float inner = alpha[target] * boundary_inner;
                for (int source = 0; source <= target; ++source) {
                    const int base = vector_base(context, source);
                    inner = fmaf(
                        weights[source * kChunk + target],
                        a_outer_inner(
                            context,
                            target,
                            u + base,
                            (driven ? h : u) + base,
                            lower),
                        inner);
                }
                zeta[component * kChunk + target] = inner;
                const float a = radial_a[component * kChunk + target];
                radial_p[component * kChunk + target] =
                    -a * strength * strength
                    / radial_q2[component * kChunk + target] * inner;
            }
        } else {
            for (int component = 0; component < kComponents; ++component) {
                a_boundary[component * kChunk + target] = 0.0f;
                zeta[component * kChunk + target] = 0.0f;
                radial_p[component * kChunk + target] = 0.0f;
            }
        }
    }
    __syncthreads();

    // zeta shares storage with the scalar coefficient reverse below. Contract
    // the strength derivative before that scratch is reused.
    if (threadIdx.x == 0) {
        float radial_strength = 0.0f;
        for (int component = 0; component < kComponents; ++component) {
            for (int target = 0; target < valid_count; ++target) {
                const float q2 = radial_q2[component * kChunk + target];
                radial_strength += kRadius * kRadius * kRadius
                    * zeta[component * kChunk + target]
                    / (q2 * sqrtf(q2));
            }
        }
        strength_partial[panel] = diagonal_strength + radial_strength;
    }

    // Local derivatives with respect to the normalized boundary coefficient
    // alpha_t and every source coefficient w_{s,t}.
    if (threadIdx.x < kChunk) {
        const int target = threadIdx.x;
        float value = 0.0f;
        if (target < valid_count) {
            for (int component = 0; component < kComponents; ++component) {
                float z_boundary =
                    alpha[target] * boundary_norm[component];
                for (int source = 0; source <= target; ++source) {
                    z_boundary = fmaf(
                        weights[source * kChunk + target],
                        boundary_source[component * kChunk + source],
                        z_boundary);
                }
                value = fmaf(
                    radial_a[component * kChunk + target],
                    a_boundary[component * kChunk + target],
                    value);
                value = fmaf(
                    radial_p[component * kChunk + target],
                    z_boundary,
                    value);
            }
            const float* boundary_j_panel = boundary_j + boundary_base;
            const float* boundary_d_panel = boundary_d + boundary_base;
            for (int row = 0; row < kRank; ++row) {
                value = fmaf(
                    x[row * kChunk + target],
                    boundary_j_panel[row * kRank + row],
                    value);
                value = fmaf(
                    y[row * kChunk + target],
                    boundary_d_panel[row * kRank + row],
                    value);
            }
        }
        alpha_bar[target] = value;
    }
    for (int job = threadIdx.x; job < kScoreElements;
         job += blockDim.x) {
        const int source = job / kChunk;
        const int target = job % kChunk;
        float value = 0.0f;
        if (source <= target && target < valid_count) {
            const int source_base = vector_base(context, source);
            for (int component = 0; component < kComponents; ++component) {
                const bool lower = component < 2;
                const bool driven = component & 1;
                const float a_inner = a_outer_inner(
                    context,
                    target,
                    u + source_base,
                    (driven ? h : u) + source_base,
                    lower);
                float z_inner = alpha[target]
                    * boundary_source[component * kChunk + source];
                for (int previous = 0; previous <= target; ++previous) {
                    z_inner = fmaf(
                        weights[previous * kChunk + target],
                        pair[component * kScoreElements
                            + previous * kChunk + source],
                        z_inner);
                }
                value = fmaf(
                    radial_a[component * kChunk + target], a_inner, value);
                value = fmaf(
                    radial_p[component * kChunk + target], z_inner, value);
            }
            for (int row = 0; row < kRank; ++row) {
                value = fmaf(
                    x[row * kChunk + target],
                    u[source_base + row] * u[source_base + row],
                    value);
                value = fmaf(
                    y[row * kChunk + target],
                    u[source_base + row] * h[source_base + row],
                    value);
            }
        }
        weight_bar[job] = value;
    }
    __syncthreads();

    // Reverse the normalized moment recurrence.  Contracting each local chart
    // cotangent with (outer_t - state_{t-1}) avoids separately differentiating
    // alpha and w: those two terms nearly cancel when a large boundary moment
    // is cancelled by the first local outer product.  The scalar accumulation
    // is deliberately FP64; it is tiny (C^3 scalars per panel), deterministic,
    // and never divides by lambda, including exp(-1000) == 0.
    if (threadIdx.x == 0) {
        double inverse_bar[kChunk];
        inverse_bar[0] = 0.0;
        for (int target = 1; target < valid_count; ++target) {
            double value = 0.0;
            double future_scale = 1.0;
            for (int local_loss = target;
                 local_loss < valid_count;
                 ++local_loss) {
                double previous_inner =
                    static_cast<double>(alpha[target - 1])
                    * static_cast<double>(alpha_bar[local_loss]);
                for (int source = 0; source < target; ++source) {
                    const int weight_index =
                        source * kChunk + local_loss;
                    previous_inner = fma(
                        static_cast<double>(
                            weights[source * kChunk + target - 1]),
                        static_cast<double>(weight_bar[weight_index]),
                        previous_inner);
                }
                const int local_index =
                    target * kChunk + local_loss;
                const double local_inner =
                    static_cast<double>(weight_bar[local_index]);
                value = fma(
                    future_scale, local_inner - previous_inner, value);
                if (local_loss + 1 < valid_count) {
                    future_scale *=
                        static_cast<double>(beta[local_loss + 1]);
                }
            }
            inverse_bar[target] = value;
        }

        double mass_carry = 0.0;
        for (int target = valid_count - 1; target >= 1; --target) {
            const double inverse =
                static_cast<double>(inverse_mass[target]);
            const double mass_bar = mass_carry
                - inverse_bar[target] * inverse * inverse;
            const double previous_mass =
                1.0 / static_cast<double>(inverse_mass[target - 1]);
            const double lambda_bar = mass_bar * previous_mass;
            mass_carry = static_cast<double>(lambda[target]) * mass_bar;
            const int token = token_start + target;
            const int scalar_index =
                (batch_index * length + token) * heads + head;
            grad_geometry_log_decay[scalar_index] = static_cast<float>(
                static_cast<double>(lambda[target]) * lambda_bar);
        }

        // The first decay and boundary mass have exact raw-state formulas that
        // remain valid at boundary_m == 0. They also avoid the division by the
        // boundary mass that made the old alpha/w pullback singular there.
        const double boundary_mass =
            static_cast<double>(boundary_m[panel]);
        double boundary_mass_bar = 0.0;
        double first_decay_bar = 0.0;
        for (int local_loss = 0;
             local_loss < valid_count;
             ++local_loss) {
            const double boundary_inner =
                static_cast<double>(alpha_bar[local_loss]);
            double state_inner =
                static_cast<double>(alpha[local_loss]) * boundary_inner;
            for (int source = 0; source <= local_loss; ++source) {
                const int weight_index =
                    source * kChunk + local_loss;
                state_inner = fma(
                    static_cast<double>(weights[weight_index]),
                    static_cast<double>(weight_bar[weight_index]),
                    state_inner);
            }
            const double boundary_coefficient =
                static_cast<double>(alpha[local_loss]);
            boundary_mass_bar = fma(
                -boundary_coefficient, state_inner, boundary_mass_bar);
            first_decay_bar = fma(
                boundary_coefficient,
                boundary_inner - boundary_mass * state_inner,
                first_decay_bar);
        }
        const int first_scalar_index =
            (batch_index * length + token_start) * heads + head;
        grad_geometry_log_decay[first_scalar_index] =
            static_cast<float>(first_decay_bar);
        grad_boundary_m[panel] = static_cast<float>(boundary_mass_bar);
    }
    __syncthreads();

    // Projection coefficients shared by the dense boundary and all local
    // source actions. pair is no longer needed and becomes the CxC coupling.
    for (int job = threadIdx.x; job < kComponents * kChunk;
         job += blockDim.x) {
        const int component = job / kChunk;
        const int source = job % kChunk;
        float value = 0.0f;
        if (source < valid_count) {
            for (int target = source; target < valid_count; ++target) {
                value = fmaf(
                    weights[source * kChunk + target]
                        * radial_p[component * kChunk + target],
                    alpha[target],
                    value);
            }
        }
        projection_boundary[job] = value;
    }
    if (threadIdx.x < kComponents) {
        const int component = threadIdx.x;
        float value = 0.0f;
        for (int target = 0; target < valid_count; ++target) {
            value = fmaf(
                alpha[target] * alpha[target],
                radial_p[component * kChunk + target],
                value);
        }
        projection_boundary_self[component] = value;
    }
    for (int job = threadIdx.x;
         job < kComponents * kScoreElements;
         job += blockDim.x) {
        const int component = job / kScoreElements;
        const int pair_index = job % kScoreElements;
        const int source = pair_index / kChunk;
        const int previous = pair_index % kChunk;
        float value = 0.0f;
        const int first = max(source, previous);
        for (int target = first; target < valid_count; ++target) {
            value = fmaf(
                weights[source * kChunk + target]
                    * weights[previous * kChunk + target],
                radial_p[component * kChunk + target],
                value);
        }
        pair[job] = value;
    }
    __syncthreads();

    // Each lane owns one source token, so vector accumulation is deterministic
    // and requires no atomics.
    if (threadIdx.x < valid_count) {
        const int source = threadIdx.x;
        const int source_base = vector_base(context, source);
        float* grad_u_source = grad_u + source_base;
        float* grad_h_source = grad_h + source_base;
        for (int row = 0; row < kRank; ++row) {
            grad_u_source[row] = 0.0f;
            grad_h_source[row] = 0.0f;
        }

        for (int target = source; target < valid_count; ++target) {
            const float weight = weights[source * kChunk + target];
            add_a_action(
                context, true, false, target, u + source_base,
                weight * radial_a[0 * kChunk + target], grad_u_source);
            add_a_action(
                context, true, true, target, u + source_base,
                weight * radial_a[0 * kChunk + target], grad_u_source);
            add_a_action(
                context, false, false, target, u + source_base,
                weight * radial_a[2 * kChunk + target], grad_u_source);
            add_a_action(
                context, false, true, target, u + source_base,
                weight * radial_a[2 * kChunk + target], grad_u_source);

            add_a_action(
                context, true, false, target, h + source_base,
                weight * radial_a[1 * kChunk + target], grad_u_source);
            add_a_action(
                context, false, false, target, h + source_base,
                weight * radial_a[3 * kChunk + target], grad_u_source);
            add_a_action(
                context, true, true, target, u + source_base,
                weight * radial_a[1 * kChunk + target], grad_h_source);
            add_a_action(
                context, false, true, target, u + source_base,
                weight * radial_a[3 * kChunk + target], grad_h_source);
        }

        for (int component = 0; component < kComponents; ++component) {
            const bool lower = component < 2;
            const bool driven = component & 1;
            const float* boundary = (driven ? boundary_d : boundary_j)
                + boundary_base;
            if (!driven) {
                add_boundary_action(
                    boundary, lower, false, u + source_base,
                    projection_boundary[component * kChunk + source],
                    grad_u_source);
                add_boundary_action(
                    boundary, lower, true, u + source_base,
                    projection_boundary[component * kChunk + source],
                    grad_u_source);
            } else {
                add_boundary_action(
                    boundary, lower, false, h + source_base,
                    projection_boundary[component * kChunk + source],
                    grad_u_source);
                add_boundary_action(
                    boundary, lower, true, u + source_base,
                    projection_boundary[component * kChunk + source],
                    grad_h_source);
            }
            for (int previous = 0; previous < valid_count; ++previous) {
                const float coefficient = pair[
                    component * kScoreElements
                    + source * kChunk + previous];
                if (!driven) {
                    add_source_action(
                        context, false, lower, false, previous,
                        u + source_base, coefficient, grad_u_source);
                    add_source_action(
                        context, false, lower, true, previous,
                        u + source_base, coefficient, grad_u_source);
                } else {
                    add_source_action(
                        context, true, lower, false, previous,
                        h + source_base, coefficient, grad_u_source);
                    add_source_action(
                        context, true, lower, true, previous,
                        u + source_base, coefficient, grad_h_source);
                }
            }
        }

        for (int row = 0; row < kRank; ++row) {
            float diagonal_h = 0.0f;
            float diagonal_r = 0.0f;
            for (int target = source; target < valid_count; ++target) {
                diagonal_h = fmaf(
                    weights[source * kChunk + target],
                    x[row * kChunk + target],
                    diagonal_h);
                diagonal_r = fmaf(
                    weights[source * kChunk + target],
                    y[row * kChunk + target],
                    diagonal_r);
            }
            grad_u_source[row] = fmaf(
                2.0f * diagonal_h, u[source_base + row],
                fmaf(diagonal_r, h[source_base + row], grad_u_source[row]));
            grad_h_source[row] = fmaf(
                diagonal_r, u[source_base + row], grad_h_source[row]);
        }
    }
    __syncthreads();

    // Dense boundary outputs are unavoidable contract gradients, but only one
    // such pair exists per chunk. No per-token dense cotangent is staged.
    for (int index = threadIdx.x; index < kRank * kRank;
         index += blockDim.x) {
        const int row = index / kRank;
        const int column = index % kRank;
        float value_j = 0.0f;
        float value_d = 0.0f;
        if (row == column) {
            for (int target = 0; target < valid_count; ++target) {
                value_j = fmaf(
                    alpha[target], x[row * kChunk + target], value_j);
                value_d = fmaf(
                    alpha[target], y[row * kChunk + target], value_d);
            }
        } else {
            const bool lower = row > column;
            const int component_j = lower ? 0 : 2;
            const int component_d = lower ? 1 : 3;
            for (int target = 0; target < valid_count; ++target) {
                float a_entry = 0.0f;
                for (int term = 0; term < 3; ++term) {
                    a_entry = fmaf(
                        a_left_value(context, lower, term, target, row),
                        a_right_value(
                            context, lower, term, target, column),
                        a_entry);
                }
                value_j = fmaf(
                    alpha[target]
                        * radial_a[component_j * kChunk + target],
                    a_entry,
                    value_j);
                value_d = fmaf(
                    alpha[target]
                        * radial_a[component_d * kChunk + target],
                    a_entry,
                    value_d);
            }
            value_j = fmaf(
                projection_boundary_self[component_j],
                boundary_j[boundary_base + index],
                value_j);
            value_d = fmaf(
                projection_boundary_self[component_d],
                boundary_d[boundary_base + index],
                value_d);
            for (int source = 0; source < valid_count; ++source) {
                const int source_base = vector_base(context, source);
                const float left = u[source_base + row];
                value_j = fmaf(
                    projection_boundary[component_j * kChunk + source],
                    left * u[source_base + column],
                    value_j);
                value_d = fmaf(
                    projection_boundary[component_d * kChunk + source],
                    left * h[source_base + column],
                    value_d);
            }
        }
        grad_boundary_j[boundary_base + index] = value_j;
        grad_boundary_d[boundary_base + index] = value_d;
    }
    __syncthreads();

    // Finish the direct b = erase * key pullback only after factor_key_bar has
    // served as the exact L cotangent vector throughout the chart reverse.
    for (int index = threadIdx.x; index < kPanelElements;
         index += blockDim.x) {
        const int row = index / kChunk;
        const int target = index % kChunk;
        if (target >= valid_count) continue;
        const int base = vector_base(context, target);
        const float grad_b = grad_erase_input[base + row];
        grad_key[base + row] = fmaf(
            grad_b, erase[base + row], grad_key[base + row]);
        grad_erase_input[base + row] = grad_b * key[base + row];
    }
}


__global__ void reduce_strength_kernel(
    const float* partial,
    float* output,
    int batch,
    int heads,
    int chunks) {
    const int head = blockIdx.x;
    if (threadIdx.x != 0) return;
    float total = 0.0f;
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int panel = (batch_index * heads + head) * chunks + chunk;
            total += partial[panel];
        }
    }
    output[head] = total;
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


C32BackwardResult c32_frame_backward_cuda(
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
    const at::Tensor& inverse_mass,
    const at::Tensor& lower_primal,
    const at::Tensor& lower_dual_scaled,
    const at::Tensor& write_direction,
    const at::Tensor& grad_write_direction,
    const at::Tensor& grad_erase_direction,
    const at::Tensor& grad_query) {
    TORCH_CHECK(
        u.is_cuda() && u.scalar_type() == at::kFloat && u.is_contiguous(),
        "u must be contiguous FP32 CUDA");
    TORCH_CHECK(
        u.dim() == 4 && u.size(3) == kRank,
        "u must be [B,T,H,128]");
    const int64_t batch = u.size(0);
    const int64_t length = u.size(1);
    const int64_t heads = u.size(2);
    TORCH_CHECK(
        batch > 0 && length > 0 && heads > 0,
        "B, T, and H must be positive");
    const int64_t chunks = (length - 1) / kChunk + 1;

    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{&h, "h"},
             {&geometry_log_decay, "geometry_log_decay"},
             {&key, "key"},
             {&erase, "erase"},
             {&query, "query"},
             {&geometry_strength, "geometry_strength"},
             {&boundary_m, "boundary_m"},
             {&boundary_j, "boundary_J"},
             {&boundary_d, "boundary_D"},
             {&inverse_mass, "inverse_mass"},
             {&lower_primal, "lower_primal"},
             {&lower_dual_scaled, "lower_dual_scaled"},
             {&write_direction, "write_direction"},
             {&grad_write_direction, "grad_write_direction"},
             {&grad_erase_direction, "grad_erase_direction"},
             {&grad_query, "grad_query"}}) {
        check_fp32_cuda_contiguous(*named.first, u, named.second);
    }
    TORCH_CHECK(
        h.sizes() == u.sizes() && query.sizes() == u.sizes(),
        "h/query shape mismatch");
    TORCH_CHECK(
        geometry_log_decay.sizes()
            == at::IntArrayRef({batch, length, heads}),
        "geometry_log_decay must be [B,T,H]");
    TORCH_CHECK(
        key.sizes()
            == at::IntArrayRef({batch, length, heads, 1, kRank})
            && erase.sizes() == key.sizes(),
        "key/erase must be [B,T,H,1,128]");
    TORCH_CHECK(
        geometry_strength.sizes() == at::IntArrayRef({heads}),
        "geometry_strength must be [H]");
    TORCH_CHECK(
        boundary_m.sizes() == at::IntArrayRef({batch, heads, chunks}),
        "boundary_m must be [B,H,N]");
    TORCH_CHECK(
        boundary_j.sizes()
            == at::IntArrayRef({batch, heads, chunks, kRank, kRank})
            && boundary_d.sizes() == boundary_j.sizes(),
        "boundary_J/D must be [B,H,N,128,128]");
    TORCH_CHECK(
        inverse_mass.sizes()
            == at::IntArrayRef({batch, heads, chunks, kChunk}),
        "inverse_mass must be [B,H,N,32]");
    TORCH_CHECK(
        lower_primal.sizes() == key.sizes()
            && write_direction.sizes() == key.sizes()
            && grad_write_direction.sizes() == key.sizes()
            && grad_erase_direction.sizes() == key.sizes(),
        "primal/edit tensors must be [B,T,H,1,128]");
    TORCH_CHECK(
        lower_dual_scaled.sizes()
            == at::IntArrayRef({batch, length, heads, 2, kRank}),
        "lower_dual_scaled must be [B,T,H,2,128]");
    TORCH_CHECK(
        grad_query.sizes() == query.sizes(),
        "grad_query shape mismatch");

    constexpr int64_t max_index = std::numeric_limits<int>::max();
    TORCH_CHECK(
        length <= max_index && heads <= max_index && chunks <= max_index,
        "length, heads, and chunks must fit the native int32 launch ABI");
    TORCH_CHECK(
        boundary_m.numel() <= max_index,
        "panel count must fit the native int32 index space");
    TORCH_CHECK(
        u.numel() <= max_index / 3,
        "vector tensors exceed the three-panel temporary int32 index space");
    TORCH_CHECK(
        boundary_j.numel() <= max_index,
        "boundary matrices exceed the native int32 index space");

    c10::cuda::CUDAGuard guard(u.device());
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, u.get_device()));
    TORCH_CHECK(
        properties.major == 12 && properties.minor == 0,
        "the native C32 frame contains only the SM120 specialization; got SM",
        properties.major,
        properties.minor);
    auto grad_u = at::empty_like(u);
    auto grad_h = at::empty_like(h);
    auto grad_geometry_log_decay = at::empty_like(geometry_log_decay);
    auto grad_key = at::empty_like(key);
    auto grad_erase_input = at::empty_like(erase);
    auto grad_query_input = at::empty_like(query);
    auto grad_geometry_strength = at::empty_like(geometry_strength);
    auto grad_boundary_m = at::empty_like(boundary_m);
    auto grad_boundary_j = at::empty_like(boundary_j);
    auto grad_boundary_d = at::empty_like(boundary_d);

    const int64_t vector_elements = batch * length * heads * kRank;
    auto temporary = at::empty({3, vector_elements}, u.options());
    const int panels = static_cast<int>(boundary_m.numel());
    auto strength_partial = at::empty({panels}, u.options());
    const size_t shared_bytes = kSharedFloats * sizeof(float);
    auto kernel = c32_frame_backward_kernel;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    kernel<<<panels, kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        u.data_ptr<float>(),
        h.data_ptr<float>(),
        geometry_log_decay.data_ptr<float>(),
        key.data_ptr<float>(),
        erase.data_ptr<float>(),
        query.data_ptr<float>(),
        geometry_strength.data_ptr<float>(),
        boundary_m.data_ptr<float>(),
        boundary_j.data_ptr<float>(),
        boundary_d.data_ptr<float>(),
        inverse_mass.data_ptr<float>(),
        lower_primal.data_ptr<float>(),
        lower_dual_scaled.data_ptr<float>(),
        write_direction.data_ptr<float>(),
        grad_write_direction.data_ptr<float>(),
        grad_erase_direction.data_ptr<float>(),
        grad_query.data_ptr<float>(),
        temporary.data_ptr<float>(),
        grad_u.data_ptr<float>(),
        grad_h.data_ptr<float>(),
        grad_geometry_log_decay.data_ptr<float>(),
        grad_key.data_ptr<float>(),
        grad_erase_input.data_ptr<float>(),
        grad_query_input.data_ptr<float>(),
        strength_partial.data_ptr<float>(),
        grad_boundary_m.data_ptr<float>(),
        grad_boundary_j.data_ptr<float>(),
        grad_boundary_d.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        static_cast<int>(vector_elements));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    reduce_strength_kernel<<<
        static_cast<int>(heads), 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        strength_partial.data_ptr<float>(),
        grad_geometry_strength.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(heads),
        static_cast<int>(chunks));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        grad_u,
        grad_h,
        grad_geometry_log_decay,
        grad_key,
        grad_erase_input,
        grad_query_input,
        grad_geometry_strength,
        grad_boundary_m,
        grad_boundary_j,
        grad_boundary_d};
}
