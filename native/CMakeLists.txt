cmake_minimum_required(VERSION 3.25)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
project(causallsso_native LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(MATHDX_ROOT "" CACHE PATH "MathDx installation prefix")
find_package(
  mathdx REQUIRED COMPONENTS cublasdx CONFIG
  HINTS ${MATHDX_ROOT} $ENV{MATHDX_ROOT}
)

add_library(
  causallsso_mathdx SHARED
  mathdx_trsm.cu
  packet_frame.cu
  packet_frame_action_vjp.cu
  packet_frame_descriptor_vjp.cu
  packet_frame_rank5_vjp.cu
  packet_frame_radial_forward.cu
  packet_frame_radial_vjp.cu
  packet_frame_scalar_vjp.cu
  packet_frame_omega_vjp.cu
)
target_compile_features(causallsso_mathdx PRIVATE cxx_std_20)
target_compile_options(
  causallsso_mathdx PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xptxas=-v>
)
target_link_libraries(
  causallsso_mathdx PRIVATE
  ${TORCH_LIBRARIES}
  torch_cuda
  c10_cuda
  mathdx::cublasdx_fatbin
)
set_target_properties(
  causallsso_mathdx
  PROPERTIES
    PREFIX "lib"
    CUDA_ARCHITECTURES 120-real
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
