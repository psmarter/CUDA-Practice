// CUTLASS Tensor Core GEMM 示例 - 使用 CUTLASS 实现基于 Tensor Core 的高性能混合精度 GEMM
// 展示 OpClassTensorOp 和混合精度计算 (FP16 输入，FP32 累加)
//
// 注意：编译本文件需要安装 CUTLASS 头文件库
#include <code_abbreviation.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#ifdef __has_include
  #if __has_include(<cutlass/gemm/device/gemm.h>)
    #define HAS_CUTLASS 1
  #else
    #define HAS_CUTLASS 0
  #endif
#else
  #define HAS_CUTLASS 0
#endif

#if HAS_CUTLASS
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#endif

// ========================= CPU 参考实现 =========================

// 混合精度 GEMM (CPU，手写)
void gemm_cpu_fp16(const half* A, const half* B, float* C, CInt M, CInt N, CInt K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

// ========================= 验证函数 =========================

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = 5e-2f) {
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配\n";
        return false;
    }

    int error_count = 0;
    float max_diff = 0.0f;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float gpu_v = gpu_result[i];
        float cpu_v = cpu_result[i];
        float diff = fabs(gpu_v - cpu_v);
        if (diff > max_diff) max_diff = diff;
        
        // Tensor Core 处理可能会带来精度的下降，因此放宽差异容忍比率
        if (diff > epsilon && (diff / (fabs(cpu_v) + 1e-5f)) > 1e-2f) {
            error_count++;
        }
    }

    if (error_count > 0) {
        cout << "✗ " << kernel_name << " FAILED: " << error_count << " 个元素超出误差阈值 (max_diff=" << max_diff << ")\n";
        return false;
    }

    cout << "✓ " << kernel_name << " PASSED (最大误差 " << max_diff << ")\n";
    return true;
}

// ========================= CUTLASS GEMM =========================

#if HAS_CUTLASS

// CUTLASS Tensor Core GEMM 封装（GPU，手写）
GpuTimingResult cutlass_tensorop_gemm_run(const std::vector<half>& h_A, const std::vector<half>& h_B, RMatrix h_C,
                                          CInt M, CInt N, CInt K, CInt iterations) {
    // 定义 CUTLASS GEMM 类型，使用 Tensor Core (OpClassTensorOp) 和混合精度
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,                     // ElementA
        cutlass::layout::RowMajor,           // LayoutA
        cutlass::half_t,                     // ElementB
        cutlass::layout::ColumnMajor,        // LayoutB
        float,                               // ElementOutput
        cutlass::layout::RowMajor,           // LayoutOutput
        float,                               // ElementAccumulator
        cutlass::arch::OpClassTensorOp,      // OperatorClass (使用 Tensor Core)
        cutlass::arch::Sm80                  // 架构 (Ampere 及以上)
    >;

    half *d_A = nullptr, *d_B = nullptr;
    PFloat d_C = nullptr;
    CSize size_A = M * K * sizeof(half);
    CSize size_B = K * N * sizeof(half);
    CSize size_C = M * N * FSIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    Gemm gemm_op;
    Gemm::Arguments args(
        {M, N, K},                                              // 问题规模
        {reinterpret_cast<cutlass::half_t*>(d_A), K},           // TensorA
        {reinterpret_cast<cutlass::half_t*>(d_B), K},           // TensorB
        {d_C, N},                                               // TensorC
        {d_C, N},                                               // TensorD (输出)
        {1.0f, 0.0f}                                            // alpha, beta
    );

    // 预热
    
    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    cutlass::Status status = gemm_op(args, workspace);
 if (status != cutlass::Status::kSuccess) { std::cout << "CUTLASS Error: " << cutlass::cutlassGetStatusString(status) << std::endl; }
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        
    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    cutlass::Status status = gemm_op(args, workspace);
 if (status != cutlass::Status::kSuccess) { std::cout << "CUTLASS Error: " << cutlass::cutlassGetStatusString(status) << std::endl; }
    }
    timerKernel.stop();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

#endif  // HAS_CUTLASS

// ========================= cuBLAS 对比 =========================

// cuBLAS GEMM - 混合精度（GPU，手写）
GpuTimingResult cublas_mixed_gemm_run(const std::vector<half>& h_A, const std::vector<half>& h_B, RMatrix h_C,
                                      CInt M, CInt N, CInt K, CInt iterations) {
    half *d_A = nullptr, *d_B = nullptr;
    PFloat d_C = nullptr;
    CSize size_A = M * K * sizeof(half);
    CSize size_B = K * N * sizeof(half);
    CSize size_C = M * N * FSIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    cublasHandle_t handle;
    cublasCreate(&handle);

    // 启用 Tensor Core 矩阵乘数学模式
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    float alpha = 1.0f, beta = 0.0f;

    // cublasGemmEx 支持混合精度计算
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K,
                 &beta,
                 d_C, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     d_B, CUDA_R_16F, N,
                     d_A, CUDA_R_16F, K,
                     &beta,
                     d_C, CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    timerKernel.stop();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

// ========================= 主函数 =========================

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt M = 2048;
    CInt N = 2048;
    CInt K = 2048;
    CInt iterations = 20;
    const double flops = 2.0 * M * N * K;

    printDeviceInfo();

    cout << "========================================\n";
    cout << "   CUTLASS Tensor Core GEMM 性能基准测试\n";
    cout << "========================================\n";
    cout << "矩阵维度：A(" << M << " x " << K << ") * B(" << K << " x " << N << ")\n";
    cout << "数据类型：FP16 输入，FP32 累加输出\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    Matrix h_C_cpu(M * N, 0.0f);
    Matrix h_C_cublas(M * N, 0.0f);

    srand(42);
    // 限定在小随机实数以防累加时严重的向下溢出和精度丢失
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(static_cast<float>(rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(static_cast<float>(rand() % 100) / 100.0f);

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    // gemm_cpu_fp16(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K); // Disabled
    cpuTimer.stop();
    cout << "CPU 执行时间：" << setw(8) << cpuTimer.elapsed_ms() << " ms\n\n";

#if HAS_CUTLASS
    // CUTLASS Tensor Core GEMM
    cout << "--- CUTLASS Tensor Core GEMM ---\n";
    Matrix h_C_cutlass(M * N, 0.0f);
    GpuTimingResult res_cutlass = cutlass_tensorop_gemm_run(h_A, h_B, h_C_cutlass, M, N, K, iterations);
    double cutlass_tflops = (flops / 1e12) / (res_cutlass.kernel_ms / 1000.0);
    cout << "Kernel 执行时间：" << setw(8) << res_cutlass.kernel_ms << " ms\n";
    cout << "计算性能：       " << setprecision(2) << cutlass_tflops << " TFLOPS\n\n";
#else
    cout << "[提示] CUTLASS 未安装，跳过 CUTLASS Tensor Core GEMM 测试\n";
    cout << "       请设置 CUTLASS_DIR 环境变量后重新编译\n\n";
#endif

    // cuBLAS 对比
    cout << "--- cuBLAS Tensor Core GEMM (对比基准) ---\n";
    GpuTimingResult res_cublas = cublas_mixed_gemm_run(h_A, h_B, h_C_cublas, M, N, K, iterations);
    double cublas_tflops = (flops / 1e12) / (res_cublas.kernel_ms / 1000.0);
    cout << "Kernel 执行时间：" << setw(8) << res_cublas.kernel_ms << " ms\n";
    cout << "计算性能：       " << setprecision(2) << cublas_tflops << " TFLOPS\n\n";

    // 性能对比
    cout << "--- 性能对比 ---\n";
#if HAS_CUTLASS
    cout << "CUTLASS GEMM:  " << setprecision(2) << cutlass_tflops << " TFLOPS\n";
#endif
    cout << "cuBLAS GEMM:   " << setprecision(2) << cublas_tflops << " TFLOPS\n";
#if HAS_CUTLASS
    cout << "CUTLASS/cuBLAS: " << setprecision(1) << (cutlass_tflops / cublas_tflops * 100.0) << "%\n";
#endif
    cout << "\n";

    // 验证
    cout << "--- 结果验证 ---\n";
    bool pass_cublas = verify_results(h_C_cublas, h_C_cpu, "cuBLAS Tensor Core GEMM");
#if HAS_CUTLASS
    bool pass_cutlass = verify_results(h_C_cutlass, h_C_cpu, "CUTLASS Tensor Core GEMM");
    if (pass_cublas && pass_cutlass) {
#else
    if (pass_cublas) {
#endif
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";
    return 0;
}
