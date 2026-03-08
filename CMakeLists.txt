cmake_minimum_required(VERSION 3.18)

project(CUDA_Practice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 设置 CUDA 架构，自动检测或者指定
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# 包含 Common 头文件路径
include_directories(${CMAKE_SOURCE_DIR}/Common/include)

# 添加子目录
add_subdirectory(01_Basics/01_vector_add)
add_subdirectory(01_Basics/02_matrix_mul_naive)
add_subdirectory(01_Basics/03_matrix_mul_tiled)
add_subdirectory(02_Reduction/01_reduce_sum)
add_subdirectory(02_Reduction/02_reduce_optimized)
add_subdirectory(02_Reduction/03_dot_product)
add_subdirectory(03_Scan/01_prefix_sum)
add_subdirectory(03_Scan/02_segmented_scan)
add_subdirectory(04_GEMM_Optimization/01_tiled_gemm)
add_subdirectory(04_GEMM_Optimization/02_advanced_gemm)
add_subdirectory(05_LLM_Ops/01_softmax)
add_subdirectory(05_LLM_Ops/02_layernorm)
add_subdirectory(05_LLM_Ops/03_flash_attention)
add_subdirectory(06_Warp_Primitives/01_warp_shuffle)
add_subdirectory(06_Warp_Primitives/02_warp_reduce)
add_subdirectory(06_Warp_Primitives/03_warp_scan)
add_subdirectory(07_Quantization/01_fp16_gemm)
add_subdirectory(07_Quantization/02_int8_gemm)
add_subdirectory(07_Quantization/03_quant_dequant)
add_subdirectory(08_Advanced/01_cuda_graphs)
add_subdirectory(08_Advanced/02_multi_stream)
add_subdirectory(08_Advanced/03_pytorch_extension)

add_subdirectory(09_Tensor_Core/01_wmma_gemm)
add_subdirectory(09_Tensor_Core/02_mixed_precision)
add_subdirectory(10_Memory_Optimization/01_coalesced_access)
add_subdirectory(10_Memory_Optimization/02_bank_conflict)
add_subdirectory(10_Memory_Optimization/03_async_copy)

add_subdirectory(11_Inference_Optimization/01_kv_cache)
add_subdirectory(11_Inference_Optimization/02_kernel_fusion)
add_subdirectory(11_Inference_Optimization/03_dynamic_batching)

add_subdirectory(12_Standard_Libraries/01_cublas_gemm)
add_subdirectory(12_Standard_Libraries/02_cufft)
add_subdirectory(12_Standard_Libraries/03_thrust)

add_subdirectory(13_Performance_Analysis/01_occupancy)
add_subdirectory(13_Performance_Analysis/02_roofline)
add_subdirectory(13_Performance_Analysis/03_nsight_profiling)

add_subdirectory(04_GEMM_Optimization/03_register_tiling)
add_subdirectory(05_LLM_Ops/04_rope)
add_subdirectory(05_LLM_Ops/05_rmsnorm)
add_subdirectory(14_CUTLASS/01_cutlass_gemm)
add_subdirectory(14_CUTLASS/02_tensorop_gemm)
add_subdirectory(14_CUTLASS/03_cute_basics)

add_subdirectory(15_Multi_GPU/01_nccl_allreduce)
