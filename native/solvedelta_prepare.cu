#include "solvedelta_c32.h"

#include <torch/library.h>

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
    const at::Tensor& D_tail,
    const at::Tensor& Q_gamma,
    const at::Tensor& grad_T,
    const at::Tensor& grad_E_gamma,
    const at::Tensor& grad_A_qd,
    const at::Tensor& grad_Q_gamma,
    const at::Tensor& grad_D_tail,
    const at::Tensor& grad_G_last) {
    const auto fp32 = d.options().dtype(at::kFloat);
    auto frame_primal = at::empty(d.sizes(), fp32);
    auto frame_dual = at::empty(
        {d.size(0), d.size(1), d.size(2), 2, d.size(3)}, fp32);
    auto grad_G = c32_wy_pair_backward_cuda(
        d,
        e,
        chi,
        inclusive_decay,
        D_tail,
        Q_gamma,
        grad_T,
        grad_A_qd,
        grad_E_gamma,
        grad_Q_gamma,
        grad_D_tail,
        grad_G_last,
        frame_primal,
        frame_dual);
    // Each transpose traversal loads a source tile before overwriting it.
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
        frame_primal,
        frame_dual);

    return std::tuple_cat(
        frame,
        std::make_tuple(grad_G));
}


TORCH_LIBRARY(causallsso, m) {
    m.def("c32_solvedelta_prepare_forward(Tensor u, Tensor h, Tensor key, Tensor erase, Tensor query, Tensor boundary_J, Tensor boundary_D, Tensor inverse_mass, Tensor radial_scale, Tensor diagonal, Tensor alpha0, Tensor inclusive_decay) -> (Tensor W, Tensor A_qd, Tensor Q_gamma, Tensor D_tail, Tensor G_last, Tensor d, Tensor e, Tensor chi, Tensor lower_primal, Tensor lower_dual_scaled)");
    m.def("c32_solvedelta_prepare_backward(Tensor u, Tensor h, Tensor key, Tensor erase, Tensor query, Tensor boundary_J, Tensor boundary_D, Tensor d, Tensor e, Tensor chi, Tensor lower_primal, Tensor lower_dual_scaled, Tensor inverse_mass, Tensor radial_scale, Tensor diagonal, Tensor alpha0, Tensor inclusive_decay, Tensor D_tail, Tensor Q_gamma, Tensor grad_T, Tensor grad_E_gamma, Tensor grad_A_qd, Tensor grad_Q_gamma, Tensor grad_D_tail, Tensor grad_G_last) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl(
        "c32_solvedelta_prepare_forward",
        &c32_solvedelta_prepare_forward_cuda);
    m.impl(
        "c32_solvedelta_prepare_backward",
        &c32_solvedelta_prepare_backward_cuda);
}
