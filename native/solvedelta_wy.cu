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

__device__ __forceinline__ int dual_vector_index(
    int batch,
    int token,
    int head,
    int route,
    int coordinate,
    int length,
    int heads) {
    return ((((batch * length + token) * heads + head) * 2 + route)
            * kRank) + coordinate;
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

__global__ void wy_pair_backward_kernel(
    const at::Half* __restrict__ d,
    const at::Half* __restrict__ e,
    const at::Half* __restrict__ chi,
    const float* __restrict__ inclusive_decay,
    const at::Half* __restrict__ D_tail,
    const at::Half* __restrict__ Q_gamma,
    const float* __restrict__ grad_T,
    const float* __restrict__ grad_A_qd,
    const float* __restrict__ grad_E_gamma,
    const float* __restrict__ grad_Q_gamma,
    const float* __restrict__ grad_D_tail,
    const float* __restrict__ grad_G_last,
    float* __restrict__ frame_primal,
    float* __restrict__ frame_dual,
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
            frame_primal[index] = gd[route];
            frame_dual[dual_vector_index(
                batch,
                token_start + token,
                head,
                0,
                coordinate,
                length,
                heads)] = ge[route];
            frame_dual[dual_vector_index(
                batch,
                token_start + token,
                head,
                1,
                coordinate,
                length,
                heads)] = gq[route];
            grad_G[index] = gg[route];
        }
    }
}

void check_fp16_vectors(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
    TORCH_CHECK(
        tensor.is_cuda()
            && tensor.get_device() == reference.get_device()
            && tensor.scalar_type() == at::kHalf
            && tensor.is_contiguous(),
        name,
        " must be contiguous FP16 CUDA on the shared device");
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

at::Tensor c32_wy_pair_backward_cuda(
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
    const at::Tensor& grad_G_last,
    const at::Tensor& frame_primal,
    const at::Tensor& frame_dual) {
    check_fp16_vectors(d, d, "d");
    check_fp16_vectors(e, d, "e");
    check_fp16_vectors(chi, d, "chi");
    check_fp16_vectors(D_tail, d, "D_tail");
    check_fp16_vectors(Q_gamma, d, "Q_gamma");
    for (const auto& named : {
             std::pair<const at::Tensor*, const char*>{&inclusive_decay, "inclusive_decay"},
             {&grad_T, "grad_T"},
             {&grad_A_qd, "grad_A_qd"},
             {&grad_E_gamma, "grad_E_gamma"},
             {&grad_Q_gamma, "grad_Q_gamma"},
             {&grad_D_tail, "grad_D_tail"},
             {&grad_G_last, "grad_G_last"},
             {&frame_primal, "frame_primal"},
             {&frame_dual, "frame_dual"}}) {
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
    TORCH_CHECK(frame_primal.sizes() == d.sizes(), "frame_primal shape mismatch");
    TORCH_CHECK(
        frame_dual.sizes()
            == at::IntArrayRef({batch, length, heads, 2, kRank}),
        "frame_dual shape mismatch");
    auto fp32 = d.options().dtype(at::kFloat);
    auto grad_G = at::empty(d.sizes(), fp32);
    const int64_t panels = batch * heads * chunks;
    c10::cuda::CUDAGuard guard(d.device());
    check_sm120(d);
    wy_pair_backward_kernel<<<
        static_cast<int>(panels), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        d.data_ptr<at::Half>(),
        e.data_ptr<at::Half>(),
        chi.data_ptr<at::Half>(),
        inclusive_decay.data_ptr<float>(),
        D_tail.data_ptr<at::Half>(),
        Q_gamma.data_ptr<at::Half>(),
        grad_T.data_ptr<float>(),
        grad_A_qd.data_ptr<float>(),
        grad_E_gamma.data_ptr<float>(),
        grad_Q_gamma.data_ptr<float>(),
        grad_D_tail.data_ptr<float>(),
        grad_G_last.data_ptr<float>(),
        frame_primal.data_ptr<float>(),
        frame_dual.data_ptr<float>(),
        grad_G.data_ptr<float>(),
        static_cast<int>(length),
        static_cast<int>(heads),
        static_cast<int>(chunks),
        static_cast<int>(panels));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_G;
}
