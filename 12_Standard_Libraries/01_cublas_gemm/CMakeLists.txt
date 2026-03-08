add_executable(cublas_gemm cublas_gemm.cu)
target_link_libraries(cublas_gemm cublas cublasLt)
