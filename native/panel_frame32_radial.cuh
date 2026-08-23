#pragma once

#include <ATen/ATen.h>

#include <tuple>

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
panel_frame32_parameters_cuda(
    const at::Tensor& boundary_m,
    const at::Tensor& boundary_j,
    const at::Tensor& boundary_d,
    const at::Tensor& u,
    const at::Tensor& h,
    const at::Tensor& log_decay,
    const at::Tensor& strength,
    int64_t heads,
    int64_t chunks,
    int64_t length);
