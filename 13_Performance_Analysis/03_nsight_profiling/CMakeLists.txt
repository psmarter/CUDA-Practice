add_executable(nsight_profiling nsight_profiling.cu)

# 编译选项：-lineinfo 让 Nsight Compute 支持源码级关联分析
target_compile_options(nsight_profiling PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# NVTX 标注库 (nvtxRangePush / nvtxRangePop)
if(WIN32)
    target_link_libraries(nsight_profiling nvToolsExt64_1)
else()
    target_link_libraries(nsight_profiling nvToolsExt)
endif()
