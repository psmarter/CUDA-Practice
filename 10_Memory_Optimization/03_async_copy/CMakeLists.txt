add_executable(async_copy async_copy.cu)
# cuda::pipeline / memcpy_async 要求 sm_80+（RTX 4090 = sm_89）
set_target_properties(async_copy PROPERTIES CUDA_ARCHITECTURES "89")
