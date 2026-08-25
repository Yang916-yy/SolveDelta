#pragma once

#include <ATen/ATen.h>

#include <tuple>


using C32WYPairBackwardResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor>;

using C32FrameUpperBackwardResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

using C32FrameLowerBackwardResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

using C32PrepareForwardResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

C32PrepareForwardResult c32_solvedelta_prepare_forward_cuda(
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& inverse_mass,
    const at::Tensor& radial_scale,
    const at::Tensor& diagonal,
    const at::Tensor& alpha0,
    const at::Tensor& inclusive_decay);

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
    const at::Tensor& radial_scale,
    const at::Tensor& theta,
    const at::Tensor& diagonal,
    const at::Tensor& alpha0,
    const at::Tensor& frame_primal,
    const at::Tensor& frame_dual);

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
    const at::Tensor& radial_scale,
    const at::Tensor& theta,
    const at::Tensor& diagonal,
    const at::Tensor& alpha0,
    const at::Tensor& upper_primal,
    const at::Tensor& upper_dual,
    const at::Tensor& grad_boundary_j,
    const at::Tensor& grad_boundary_d);

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
    const at::Tensor& frame_dual);
