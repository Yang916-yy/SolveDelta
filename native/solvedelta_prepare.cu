#include "solvedelta_c32.h"

#include <torch/library.h>

C32WYPairBackwardResult c32_wy_pair_backward_stage_cuda(
    const at::Tensor& d,
    const at::Tensor& e,
    const at::Tensor& chi,
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
    return {frame_primal, frame_dual, grad_G};
}


TORCH_LIBRARY(causallsso, m) {
    m.def("c32_solvedelta_prepare_forward(Tensor u, Tensor h, Tensor key, Tensor erase, Tensor query, Tensor boundary_J, Tensor boundary_D, Tensor inverse_mass, Tensor radial_scale, Tensor diagonal, Tensor alpha0, Tensor inclusive_decay) -> (Tensor W, Tensor A_qd, Tensor Q_gamma, Tensor D_tail, Tensor G_last, Tensor d, Tensor e, Tensor chi, Tensor lower_primal, Tensor lower_dual_scaled)");
    m.def("c32_wy_pair_backward_stage(Tensor d, Tensor e, Tensor chi, Tensor inclusive_decay, Tensor D_tail, Tensor Q_gamma, Tensor grad_T, Tensor grad_E_gamma, Tensor grad_A_qd, Tensor grad_Q_gamma, Tensor grad_D_tail, Tensor grad_G_last) -> (Tensor frame_primal, Tensor frame_dual, Tensor grad_G)");
    m.def("c32_frame_upper_backward(Tensor u, Tensor h, Tensor key, Tensor erase, Tensor query, Tensor boundary_J, Tensor boundary_D, Tensor lower_primal, Tensor lower_dual_scaled, Tensor d, Tensor inverse_mass, Tensor radial_scale, Tensor theta, Tensor diagonal, Tensor alpha0, Tensor frame_primal, Tensor frame_dual) -> (Tensor upper_primal, Tensor upper_dual, Tensor grad_boundary_J, Tensor grad_boundary_D)");
    m.def("c32_frame_lower_backward(Tensor u, Tensor h, Tensor key, Tensor erase, Tensor query, Tensor boundary_J, Tensor boundary_D, Tensor lower_primal, Tensor lower_dual_scaled, Tensor d, Tensor inverse_mass, Tensor radial_scale, Tensor theta, Tensor diagonal, Tensor alpha0, Tensor(a!) upper_primal, Tensor(b!) upper_dual, Tensor(c!) grad_boundary_J, Tensor(d!) grad_boundary_D) -> (Tensor grad_key, Tensor grad_erase, Tensor grad_query, Tensor grad_log_diagonal, Tensor(c!) grad_boundary_J_out, Tensor(d!) grad_boundary_D_out, Tensor(b!) scaled_dual, Tensor(a!) lower_action_primal)");
}

TORCH_LIBRARY_IMPL(causallsso, CUDA, m) {
    m.impl(
        "c32_solvedelta_prepare_forward",
        &c32_solvedelta_prepare_forward_cuda);
    m.impl(
        "c32_wy_pair_backward_stage",
        &c32_wy_pair_backward_stage_cuda);
    m.impl(
        "c32_frame_upper_backward",
        &c32_frame_upper_backward_cuda);
    m.impl(
        "c32_frame_lower_backward",
        &c32_frame_lower_backward_cuda);
}
