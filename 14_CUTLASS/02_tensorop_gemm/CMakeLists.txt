add_executable(tensorop_gemm tensorop_gemm.cu)

if(DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_DIR $ENV{CUTLASS_DIR})
endif()

if(CUTLASS_DIR)
    target_include_directories(tensorop_gemm PRIVATE
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include
    )
    message(STATUS "CUTLASS found at: ${CUTLASS_DIR}")
else()
    message(WARNING "CUTLASS_DIR not set. TensorOp example will not compile. "
                    "Set CUTLASS_DIR to CUTLASS installation path.")
endif()

target_link_libraries(tensorop_gemm PRIVATE cublas)
