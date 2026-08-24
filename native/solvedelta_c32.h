#pragma once

#include <ATen/ATen.h>

#include <tuple>


using C32FrameActionsBackwardResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

using C32WYSolveBackwardResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor>;

using C32WYPairBackwardResult = std::tuple<
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
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

using C32PrepareBackwardResult = std::tuple<
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
    const at::Tensor& value);

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
    const at::Tensor& radial_scale,
    const at::Tensor& diagonal,
    const at::Tensor& alpha0,
    const at::Tensor& grad_d,
    const at::Tensor& grad_e,
    const at::Tensor& grad_chi);

C32WYSolveBackwardResult c32_wy_solve_backward_cuda(
    const at::Tensor& W,
    const at::Tensor& Y,
    const at::Tensor& U_z,
    const at::Tensor& grad_Y,
    const at::Tensor& grad_U_z);

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
    const at::Tensor& grad_G_last);
