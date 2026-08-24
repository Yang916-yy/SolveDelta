#include "solvedelta_c32.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <limits>


namespace {

constexpr int kRank = 128;
constexpr int kChunk = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;

__device__ __forceinline__ int vector_index(
    int batch,
    int token,
    int head,
    int coordinate,
    int length,
    int heads) {
    return ((batch * length + token) * heads + head) * kRank + coordinate;
}

__device__ __forceinline__ void decode_panel(
    int panel,
    int heads,
    int chunks,
    int& batch,
    int& head,
    int& chunk) {
    chunk = panel % chunks;
    const int head_batch = panel / chunks;
    head = head_batch % heads;
    batch = head_batch / heads;
}

__global__ void wy_solve_backward_kernel(
    const float* __restrict__ W,
    const at::BFloat16* __restrict__ Y,
    const at::BFloat16* __restrict__ U_z,
    const float* __restrict__ grad_Y,
    const float* __restrict__ grad_U_z,
    float* __restrict__ grad_E_gamma,
    float* __restrict__ grad_Z,
    float* __restrict__ grad_T,
    int length,
    int heads,
    int chunks,
    int value_dim,
    int panels) {
    extern __shared__ unsigned char storage[];
    float* shared_W = reinterpret_cast<float*>(storage);
    at::BFloat16* shared_X = reinterpret_cast<at::BFloat16*>(
        shared_W + kChunk * kChunk);
    float* shared_bar = reinterpret_cast<float*>(
        shared_X + kChunk * (kRank + value_dim));
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    int batch;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    const int rhs_count = kRank + value_dim;
    for (int index = threadIdx.x; index < kChunk * kChunk; index += blockDim.x) {
        shared_W[index] = W[panel * kChunk * kChunk + index];
    }
    __syncthreads();

    const int rhs = threadIdx.x;
    if (rhs < rhs_count) {
        float solution[kChunk];
#pragma unroll 1
        for (int reverse = 0; reverse < kChunk; ++reverse) {
            const int row = kChunk - 1 - reverse;
            float current = 0.0f;
            if (row < valid_count) {
                if (rhs < kRank) {
                    current = grad_Y[vector_index(
                        batch, token_start + row, head, rhs, length, heads)];
                } else {
                    const int coordinate = rhs - kRank;
                    const int index =
                        (((batch * length + token_start + row) * heads + head)
                         * value_dim) + coordinate;
                    current = grad_U_z[index];
                }
                for (int later = row + 1; later < valid_count; ++later) {
                    current = fmaf(
                        -shared_W[later * kChunk + row],
                        solution[later],
                        current);
                }
            }
            solution[row] = current;
            shared_bar[row * rhs_count + rhs] = current;
            if (row < valid_count) {
                if (rhs < kRank) {
                    grad_E_gamma[vector_index(
                        batch, token_start + row, head, rhs, length, heads)] = current;
                } else {
                    const int coordinate = rhs - kRank;
                    const int index =
                        (((batch * length + token_start + row) * heads + head)
                         * value_dim) + coordinate;
                    grad_Z[index] = current;
                }
            }
        }
    }
    for (int index = threadIdx.x;
         index < kChunk * rhs_count;
         index += blockDim.x) {
        const int row = index / rhs_count;
        const int column = index % rhs_count;
        at::BFloat16 bits(0.0f);
        if (row < valid_count) {
            if (column < kRank) {
                bits = Y[vector_index(
                    batch, token_start + row, head, column, length, heads)];
            } else {
                const int coordinate = column - kRank;
                const int offset =
                    (((batch * length + token_start + row) * heads + head)
                     * value_dim) + coordinate;
                bits = U_z[offset];
            }
        }
        shared_X[index] = bits;
    }
    __syncthreads();

    for (int pair = threadIdx.x; pair < kChunk * kChunk; pair += blockDim.x) {
        const int row = pair / kChunk;
        const int column = pair % kChunk;
        float gradient = 0.0f;
        if (row < valid_count && column < row) {
            for (int coordinate = 0; coordinate < rhs_count; ++coordinate) {
                gradient = fmaf(
                    -shared_bar[row * rhs_count + coordinate],
                    static_cast<float>(shared_X[column * rhs_count + coordinate]),
                    gradient);
            }
        }
        grad_T[panel * kChunk * kChunk + pair] = gradient;
    }
}

__global__ void wy_pair_backward_kernel(
    const at::BFloat16* __restrict__ d,
    const at::BFloat16* __restrict__ e,
    const at::BFloat16* __restrict__ chi,
    const float* __restrict__ inclusive_decay,
    const at::BFloat16* __restrict__ D_tail,
    const at::BFloat16* __restrict__ Q_gamma,
    const float* __restrict__ grad_T,
    const float* __restrict__ grad_A_qd,
    const float* __restrict__ grad_E_gamma,
    const float* __restrict__ grad_Q_gamma,
    const float* __restrict__ grad_D_tail,
    const float* __restrict__ grad_G_last,
    float* __restrict__ grad_d,
    float* __restrict__ grad_e,
    float* __restrict__ grad_chi,
    float* __restrict__ grad_G,
    int length,
    int heads,
    int chunks,
    int panels) {
    __shared__ float alpha[kChunk * kRank];
    const int panel = blockIdx.x;
    if (panel >= panels) return;
    int batch;
    int head;
    int chunk;
    decode_panel(panel, heads, chunks, batch, head, chunk);
    const int token_start = chunk * kChunk;
    const int valid_count = min(kChunk, length - token_start);
    for (int local = threadIdx.x; local < kChunk * kRank; local += blockDim.x) {
        const int token = local / kRank;
        const int coordinate = local % kRank;
        float value = 0.0f;
        if (token < valid_count) {
            const int index = vector_index(
                batch, token_start + token, head, coordinate, length, heads);
            const float current = inclusive_decay[index];
            const float previous = token == 0
                ? 0.0f
                : inclusive_decay[vector_index(
                    batch,
                    token_start + token - 1,
                    head,
                    coordinate,
                    length,
                    heads)];
            value = expf(current - previous);
        }
        alpha[local] = value;
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int group = 0; group < kChunk / kWarps; ++group) {
        const int token = group * kWarps + warp;
        if (token >= valid_count) continue;
        float gd[4];
        float ge[4];
        float gq[4];
        float gg[4];
        float d_value[4];
#pragma unroll
        for (int route = 0; route < 4; ++route) {
            const int coordinate = lane + route * 32;
            const int index = vector_index(
                batch, token_start + token, head, coordinate, length, heads);
            const float gate = inclusive_decay[index];
            const float exp_gate = expf(gate);
            const float final_gate = inclusive_decay[vector_index(
                batch,
                token_start + valid_count - 1,
                head,
                coordinate,
                length,
                heads)];
            const float tail_scale = expf(final_gate - gate);
            d_value[route] = static_cast<float>(d[index]);
            gd[route] = tail_scale * grad_D_tail[index];
            ge[route] = exp_gate * grad_E_gamma[index];
            gq[route] = exp_gate * grad_Q_gamma[index];
            gg[route] =
                grad_E_gamma[index] * exp_gate * static_cast<float>(e[index])
                + grad_Q_gamma[index] * static_cast<float>(Q_gamma[index])
                - grad_D_tail[index] * static_cast<float>(D_tail[index]);
            if (token == valid_count - 1) {
                float tail_gradient = grad_G_last[panel * kRank + coordinate];
                for (int source = 0; source < valid_count; ++source) {
                    const int source_index = vector_index(
                        batch,
                        token_start + source,
                        head,
                        coordinate,
                        length,
                        heads);
                    tail_gradient += grad_D_tail[source_index]
                        * static_cast<float>(D_tail[source_index]);
                }
                gg[route] += tail_gradient;
            }
        }

        float ratio[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        for (int source = token; source >= 0; --source) {
            const int pair = (panel * kChunk + token) * kChunk + source;
            const float g_t = source < token ? grad_T[pair] : 0.0f;
            const float g_a = grad_A_qd[pair];
#pragma unroll
            for (int route = 0; route < 4; ++route) {
                const int coordinate = lane + route * 32;
                const int row_index = vector_index(
                    batch, token_start + token, head, coordinate, length, heads);
                const int source_index = vector_index(
                    batch, token_start + source, head, coordinate, length, heads);
                const float p = ratio[route] * static_cast<float>(d[source_index]);
                const float bar_p =
                    g_t * static_cast<float>(e[row_index])
                    + g_a * static_cast<float>(chi[row_index]);
                ge[route] += g_t * p;
                gq[route] += g_a * p;
                gg[route] += p * bar_p;
            }
            if (source > 0) {
#pragma unroll
                for (int route = 0; route < 4; ++route) {
                    ratio[route] *= alpha[source * kRank + lane + route * 32];
                }
            }
        }

        ratio[0] = ratio[1] = ratio[2] = ratio[3] = 1.0f;
        for (int row = token; row < valid_count; ++row) {
            const int pair = (panel * kChunk + row) * kChunk + token;
            const float g_t = row > token ? grad_T[pair] : 0.0f;
            const float g_a = grad_A_qd[pair];
#pragma unroll
            for (int route = 0; route < 4; ++route) {
                const int coordinate = lane + route * 32;
                const int row_index = vector_index(
                    batch, token_start + row, head, coordinate, length, heads);
                const float bar_p =
                    g_t * static_cast<float>(e[row_index])
                    + g_a * static_cast<float>(chi[row_index]);
                gd[route] += ratio[route] * bar_p;
                gg[route] -= ratio[route] * d_value[route] * bar_p;
            }
            if (row + 1 < valid_count) {
#pragma unroll
                for (int route = 0; route < 4; ++route) {
                    ratio[route] *= alpha[(row + 1) * kRank + lane + route * 32];
                }
            }
        }

#pragma unroll
        for (int route = 0; route < 4; ++route) {
            const int coordinate = lane + route * 32;
            const int index = vector_index(
                batch, token_start + token, head, coordinate, length, heads);
            grad_d[index] = gd[route];
            grad_e[index] = ge[route];
            grad_chi[index] = gq[route];
            grad_G[index] = gg[route];
        }
    }
}

void check_bf16_vectors(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
    TORCH_CHECK(
        tensor.is_cuda()
            && tensor.get_device() == reference.get_device()
            && tensor.scalar_type() == at::kBFloat16
            && tensor.is_contiguous(),
        name,
        " must be contiguous BF16 CUDA on the shared device");
}

void check_fp32(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
    TORCH_CHECK(
        tensor.is_cuda()
            && tensor.get_device() == reference.get_device()
            && tensor.scalar_type() == at::kFloat
            && tensor.is_contiguous(),
        name,
        " must be contiguous FP32 CUDA on the shared device");
}

void check_sm120(const at::Tensor& reference) {
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, reference.get_device()));
    TORCH_CHECK(
        properties.major == 12 && properties.minor == 0,
        "the C32 WY specialization requires SM120; got SM",
        properties.major,
        properties.minor);
}

}  // namespace

C32WYSolveBackwardResult c32_wy_solve_backward_cuda(
    const at::Tensor& W,
    const at::Tensor& Y,
    const at::Tensor& U_z,
    const at::Tensor& grad_Y,
    const at::Tensor& grad_U_z) {
    check_bf16_vectors(Y, Y, "Y");
    check_bf16_vectors(U_z, Y, "U_z");
    check_fp32(W, Y, "W");
    check_fp32(grad_Y, Y, "grad_Y");
    check_fp32(grad_U_z, Y, "grad_U_z");
    const int64_t batch = Y.size(0);
    const int64_t length = Y.size(1);
    const int64_t heads = Y.size(2);
    const int64_t value_dim = U_z.size(3);
    const int64_t chunks = (length + kChunk - 1) / kChunk;
    TORCH_CHECK(Y.dim() == 4 && Y.size(3) == kRank, "Y must be [B,T,H,128]");
    TORCH_CHECK(U_z.sizes() == at::IntArrayRef({batch, length, heads, value_dim}), "U_z shape mismatch");
    TORCH_CHECK(grad_Y.sizes() == Y.sizes() && grad_U_z.sizes() == U_z.sizes(), "solve cotangent shape mismatch");
    TORCH_CHECK(value_dim > 0 && value_dim <= kRank, "native d_v must be in [1,128]");
    auto grad_E = at::empty(Y.sizes(), Y.options().dtype(at::kFloat));
    auto grad_Z = at::empty(U_z.sizes(), U_z.options().dtype(at::kFloat));
    auto grad_T = at::empty_like(W);
    const int64_t panels = batch * heads * chunks;
    const size_t shared_bytes =
        kChunk * kChunk * sizeof(float)
        + kChunk * (kRank + value_dim) * sizeof(at::BFloat16)
        + kChunk * (kRank + value_dim) * sizeof(float);
    c10::cuda::CUDAGuard guard(Y.device());
    check_sm120(Y);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        wy_solve_backward_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    wy_solve_backward_kernel<<<
        static_cast<int>(panels),
        kThreads,
        shared_bytes,
        at::cuda::getCurrentCUDAStream()>>>(
        W.data_ptr<float>(),
        Y.data_ptr<at::BFloat16>(),
        U_z.data_ptr<at::BFloat16>(),
        grad_Y.data_ptr<float>(),
        grad_U_z.data_ptr<float>(),
        grad_E.data_ptr<float>(),
        grad_Z.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        static_cast<int>(value_dim),
        static_cast<int>(panels));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_E, grad_Z, grad_T};
}

C32WYPairBackwardResult c32_wy_pair_backward_cuda(
    const at::Tensor& d,
    const at::Tensor& e,
    const at::Tensor& chi,
    const at::Tensor& inclusive_decay,
    const at::Tensor& D_tail,
    const at::Tensor& Q_gamma,
    const at::Tensor& grad_T,
    const at::Tensor& grad_A_qd,
    const at::Tensor& grad_E_gamma,
    const at::Tensor& grad_Q_gamma,
    const at::Tensor& grad_D_tail,
    const at::Tensor& grad_G_last) {
    check_bf16_vectors(d, d, "d");
    check_bf16_vectors(e, d, "e");
    check_bf16_vectors(chi, d, "chi");
    check_bf16_vectors(D_tail, d, "D_tail");
    check_bf16_vectors(Q_gamma, d, "Q_gamma");
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{&inclusive_decay, "inclusive_decay"},
             {&grad_T, "grad_T"},
             {&grad_A_qd, "grad_A_qd"},
             {&grad_E_gamma, "grad_E_gamma"},
             {&grad_Q_gamma, "grad_Q_gamma"},
             {&grad_D_tail, "grad_D_tail"},
             {&grad_G_last, "grad_G_last"}}) {
        check_fp32(*named.first, d, named.second);
    }
    TORCH_CHECK(d.dim() == 4 && d.size(3) == kRank, "d must be [B,T,H,128]");
    TORCH_CHECK(
        e.sizes() == d.sizes() && chi.sizes() == d.sizes()
            && D_tail.sizes() == d.sizes() && Q_gamma.sizes() == d.sizes(),
        "pair vector shape mismatch");
    const int64_t batch = d.size(0);
    const int64_t length = d.size(1);
    const int64_t heads = d.size(2);
    const int64_t chunks = (length + kChunk - 1) / kChunk;
    TORCH_CHECK(
        grad_T.sizes()
                == at::IntArrayRef({batch, heads, chunks, kChunk, kChunk})
            && grad_A_qd.sizes() == grad_T.sizes(),
        "pair cotangent shape mismatch");
    TORCH_CHECK(grad_E_gamma.sizes() == d.sizes() && grad_Q_gamma.sizes() == d.sizes() && grad_D_tail.sizes() == d.sizes(), "gauge cotangent shape mismatch");
    TORCH_CHECK(grad_G_last.sizes() == at::IntArrayRef({batch, heads, chunks, kRank}), "grad_G_last shape mismatch");
    auto fp32 = d.options().dtype(at::kFloat);
    auto grad_d = at::empty(d.sizes(), fp32);
    auto grad_e = at::empty(d.sizes(), fp32);
    auto grad_chi = at::empty(d.sizes(), fp32);
    auto grad_G = at::empty(d.sizes(), fp32);
    const int64_t panels = batch * heads * chunks;
    c10::cuda::CUDAGuard guard(d.device());
    check_sm120(d);
    wy_pair_backward_kernel<<<
        static_cast<int>(panels), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        d.data_ptr<at::BFloat16>(),
        e.data_ptr<at::BFloat16>(),
        chi.data_ptr<at::BFloat16>(),
        inclusive_decay.data_ptr<float>(),
        D_tail.data_ptr<at::BFloat16>(),
        Q_gamma.data_ptr<at::BFloat16>(),
        grad_T.data_ptr<float>(),
        grad_A_qd.data_ptr<float>(),
        grad_E_gamma.data_ptr<float>(),
        grad_Q_gamma.data_ptr<float>(),
        grad_D_tail.data_ptr<float>(),
        grad_G_last.data_ptr<float>(),
        grad_d.data_ptr<float>(),
        grad_e.data_ptr<float>(),
        grad_chi.data_ptr<float>(),
        grad_G.data_ptr<float>(),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        static_cast<int>(panels));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_d, grad_e, grad_chi, grad_G};
}
