#pragma once

#include <ATen/ATen.h>

#include <tuple>


using C32ForwardResult = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

using C32BackwardResult = std::tuple<
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

C32ForwardResult c32_frame_forward_cuda(
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& geometry_log_decay,
    const at::Tensor& key,
    const at::Tensor& erase,
    const at::Tensor& query,
    const at::Tensor& geometry_strength,
    const at::Tensor& boundary_m,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d);

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
    const at::Tensor& lower_primal,
    const at::Tensor& lower_dual_scaled,
    const at::Tensor& write_direction,
    const at::Tensor& grad_write_direction,
    const at::Tensor& grad_erase_direction,
    const at::Tensor& grad_solved_query);
