add_executable(cutlass_gemm cutlass_gemm.cu)

# CUTLASS 是 header-only 库，需要指定头文件路径
# 用户需要设置 CUTLASS_DIR 环境变量或通过 CMake 参数指定路径
# 例如: cmake .. -DCUTLASS_DIR=/path/to/cutlass
if(DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_DIR $ENV{CUTLASS_DIR})
endif()

if(CUTLASS_DIR)
    target_include_directories(cutlass_gemm PRIVATE
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include
    )
    message(STATUS "CUTLASS found at: ${CUTLASS_DIR}")
else()
    message(WARNING "CUTLASS_DIR not set. CUTLASS example will not compile. "
                    "Set CUTLASS_DIR to CUTLASS installation path.")
endif()

target_link_libraries(cutlass_gemm PRIVATE cublas)
