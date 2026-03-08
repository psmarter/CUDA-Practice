add_executable(wmma_gemm wmma_gemm.cu)
# WMMA API 要求 sm_70+（RTX 4090 = sm_89）
set_target_properties(wmma_gemm PROPERTIES CUDA_ARCHITECTURES "89")
