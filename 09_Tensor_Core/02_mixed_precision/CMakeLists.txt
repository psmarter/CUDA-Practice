add_executable(mixed_precision mixed_precision.cu)
# WMMA API 要求 sm_70+（RTX 4090 = sm_89）
set_target_properties(mixed_precision PROPERTIES CUDA_ARCHITECTURES "89")
