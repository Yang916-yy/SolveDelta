cmake_minimum_required(VERSION 3.25)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
project(causallsso_native LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CUTLASS_ROOT "/opt/nvidia-cutlass" CACHE PATH "CUTLASS installation root")

add_library(
  causallsso_chunk SHARED
  solvedelta_c32_forward.cu
  solvedelta_wy.cu
  solvedelta_prepare.cu
)
target_compile_features(causallsso_chunk PRIVATE cxx_std_20)
target_include_directories(
  causallsso_chunk PRIVATE
  "${CUTLASS_ROOT}/include"
)
target_compile_options(
  causallsso_chunk PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-extended-lambda;-Xptxas=-v>
)
target_link_libraries(
  causallsso_chunk PRIVATE
  ${TORCH_LIBRARIES}
  torch_cuda
  c10_cuda
)
set_target_properties(
  causallsso_chunk
  PROPERTIES
    PREFIX "lib"
    CUDA_ARCHITECTURES 120-real
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

option(
  CAUSALLSSO_BUILD_MATHDX_ORACLE
  "Build the optional MathDx r=128 validation oracle"
  OFF
)

if(CAUSALLSSO_BUILD_MATHDX_ORACLE)
  set(MATHDX_ROOT "" CACHE PATH "MathDx installation prefix")
  find_package(
    mathdx REQUIRED COMPONENTS cublasdx CONFIG
    HINTS ${MATHDX_ROOT} $ENV{MATHDX_ROOT}
  )

  add_library(causallsso_mathdx_oracle SHARED mathdx_trsm.cu)
  target_compile_features(causallsso_mathdx_oracle PRIVATE cxx_std_20)
  target_compile_options(
    causallsso_mathdx_oracle PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xptxas=-v>
  )
  target_link_libraries(
    causallsso_mathdx_oracle PRIVATE
    ${TORCH_LIBRARIES}
    torch_cuda
    c10_cuda
    mathdx::cublasdx_fatbin
  )
  set_target_properties(
    causallsso_mathdx_oracle
    PROPERTIES
      PREFIX "lib"
      CUDA_ARCHITECTURES 120-real
      CUDA_SEPARABLE_COMPILATION ON
      INTERPROCEDURAL_OPTIMIZATION ON
  )
endif()
