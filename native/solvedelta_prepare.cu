#include "solvedelta_c32.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <limits>


namespace {

constexpr int kValueThreads = 256;

__global__ void value_backward_kernel(
    const at::BFloat16* __restrict__ write,
    const at::BFloat16* __restrict__ value,
    const float* __restrict__ grad_z,
    at::BFloat16* __restrict__ grad_write,
    at::BFloat16* __restrict__ grad_value,
    int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float gradient = grad_z[index];
    grad_write[index] = at::BFloat16(
        gradient * static_cast<float>(value[index]));
    grad_value[index] = at::BFloat16(
        gradient * static_cast<float>(write[index]));
}

void check_bf16_value(
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

}  // namespace


C32PrepareBackwardResult c32_solvedelta_prepare_backward_cuda(
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& d,
    const at::Tensor& e,
    const at::Tensor& chi,
    const at::Tensor& lower_primal,
    const at::Tensor& lower_dual_scaled,
    const at::Tensor& inverse_mass,
    const at::Tensor& radial_scale,
    const at::Tensor& diagonal,
    const at::Tensor& alpha0,
    const at::Tensor& inclusive_decay,
    const at::Tensor& W,
    const at::Tensor& D_tail,
    const at::Tensor& Q_gamma,
    const at::Tensor& Y,
    const at::Tensor& U_z,
    const at::Tensor& write,
    const at::Tensor& value,
    const at::Tensor& grad_Y,
    const at::Tensor& grad_U_z,
    const at::Tensor& grad_A_qd,
    const at::Tensor& grad_Q_gamma,
    const at::Tensor& grad_D_tail,
    const at::Tensor& grad_G_last) {
    auto solve = c32_wy_solve_backward_cuda(
        W, Y, U_z, grad_Y, grad_U_z);
    auto pair = c32_wy_pair_backward_cuda(
        d,
        e,
        chi,
        inclusive_decay,
        D_tail,
        Q_gamma,
        std::get<2>(solve),
        grad_A_qd,
        std::get<0>(solve),
        grad_Q_gamma,
        grad_D_tail,
        grad_G_last);
    auto frame = c32_frame_actions_backward_cuda(
        u,
        h,
        key,
        erase,
        query,
        boundary_j,
        boundary_d,
        lower_primal,
        lower_dual_scaled,
        d,
        inverse_mass,
        radial_scale,
        diagonal,
        alpha0,
        std::get<0>(pair),
        std::get<1>(pair),
        std::get<2>(pair));

    check_bf16_value(write, d, "write");
    check_bf16_value(value, d, "value");
    TORCH_CHECK(
        write.sizes() == value.sizes()
            && write.numel() == std::get<1>(solve).numel(),
        "write/value and grad_Z shape mismatch");
    TORCH_CHECK(
        write.numel() <= std::numeric_limits<int>::max(),
        "write/value tensor exceeds the int32 index space");
    auto grad_write = at::empty_like(write);
    auto grad_value = at::empty_like(value);
    const int elements = static_cast<int>(write.numel());
    c10::cuda::CUDAGuard guard(write.device());
    value_backward_kernel<<<
        (elements + kValueThreads - 1) / kValueThreads,
        kValueThreads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        write.data_ptr<at::BFloat16>(),
        value.data_ptr<at::BFloat16>(),
        std::get<1>(solve).data_ptr<float>(),
        grad_write.data_ptr<at::BFloat16>(),
        grad_value.data_ptr<at::BFloat16>(),
        elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::tuple_cat(
        frame,
        std::make_tuple(
            std::get<3>(pair), grad_write, grad_value));
}


TORCH_LIBRARY(causallsso, m) {
    m.def("c32_solvedelta_prepare_forward(Tensor u, Tensor h, Tensor geometry_log_decay, Tensor key, Tensor erase, Tensor query, Tensor geometry_strength, Tensor boundary_m, Tensor boundary_J, Tensor boundary_D, Tensor inclusive_decay, Tensor write, Tensor value) -> (Tensor W, Tensor A_qd, Tensor Q_gamma, Tensor D_tail, Tensor G_last, Tensor Y, Tensor U_z, Tensor d, Tensor e, Tensor chi, Tensor lower_primal, Tensor lower_dual_scaled, Tensor inverse_mass, Tensor radial_scale, Tensor radial_q2, Tensor radial_norm, Tensor diagonal, Tensor alpha0)");
    m.def("c32_solvedelta_prepare_backward(Tensor u, Tensor h, Tensor key, Tensor erase, Tensor query, Tensor boundary_J, Tensor boundary_D, Tensor d, Tensor e, Tensor chi, Tensor lower_primal, Tensor lower_dual_scaled, Tensor inverse_mass, Tensor radial_scale, Tensor diagonal, Tensor alpha0, Tensor inclusive_decay, Tensor W, Tensor D_tail, Tensor Q_gamma, Tensor Y, Tensor U_z, Tensor write, Tensor value, Tensor grad_Y, Tensor grad_U_z, Tensor grad_A_qd, Tensor grad_Q_gamma, Tensor grad_D_tail, Tensor grad_G_last) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl(
        "c32_solvedelta_prepare_forward",
        &c32_solvedelta_prepare_forward_cuda);
    m.impl(
        "c32_solvedelta_prepare_backward",
        &c32_solvedelta_prepare_backward_cuda);
}
