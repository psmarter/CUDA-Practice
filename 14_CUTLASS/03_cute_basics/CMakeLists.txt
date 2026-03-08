add_executable(cute_basics cute_basics.cu)

if(DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_DIR $ENV{CUTLASS_DIR})
endif()

if(CUTLASS_DIR)
    target_include_directories(cute_basics PRIVATE
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include
    )
    message(STATUS "CuTe found at: ${CUTLASS_DIR}")
else()
    message(WARNING "CUTLASS_DIR not set. CuTe example will not compile. "
                    "Set CUTLASS_DIR to CUTLASS installation path.")
endif()

target_link_libraries(cute_basics PRIVATE cublas)
