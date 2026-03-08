// CUTLASS GEMM 示例 - 使用 CUTLASS 3.x API 实现高性能 GEMM
// 展示 CUTLASS 的 Gemm 模板配置与性能对比
//
// 注意：编译本文件需要安装 CUTLASS 头文件库
//   git clone https://github.com/NVIDIA/cutlass.git
//   cmake .. -DCUTLASS_DIR=/path/to/cutlass
#include <code_abbreviation.h>
#include <cublas_v2.h>

// CUTLASS 头文件（需要 CUTLASS_DIR 设置正确）
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

// GEMM（CPU，手写）
void gemm_cpu(CPFloat A, CPFloat B, PFloat C, CInt M, CInt N, CInt K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ========================= 验证函数 =========================

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = 1e-2f) {
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配\n";
        return false;
    }

    int error_count = 0;
    float max_diff = 0.0f;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float diff = fabs(gpu_result[i] - cpu_result[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > epsilon) error_count++;
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

// CUTLASS GEMM 封装（GPU，手写）
// 使用 CUTLASS 2.x/3.x 的 device::Gemm 模板
GpuTimingResult cutlass_gemm_run(CRMatrix h_A, CRMatrix h_B, RMatrix h_C,
                                  CInt M, CInt N, CInt K, CInt iterations) {
    // 定义 CUTLASS GEMM 类型
    // RowMajor 布局，使用 float 精度
    using Gemm = cutlass::gemm::device::Gemm<
        float,                               // ElementA
        cutlass::layout::RowMajor,           // LayoutA
        float,                               // ElementB
        cutlass::layout::RowMajor,           // LayoutB
        float,                               // ElementC
        cutlass::layout::RowMajor,           // LayoutC
        float,                               // ElementAccumulator
        cutlass::arch::OpClassSimt,          // OperatorClass (使用 CUDA Core)
        cutlass::arch::Sm80                  // 架构 (Ampere 及以上)
    >;

    PFloat d_A = nullptr, d_B = nullptr, d_C = nullptr;
    CSize size_A = M * K * FSIZE;
    CSize size_B = K * N * FSIZE;
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

    // 配置 CUTLASS GEMM 参数
    Gemm gemm_op;
    Gemm::Arguments args(
        {M, N, K},                                              // 问题规模
        {d_A, K},                                               // TensorA (ptr, lda)
        {d_B, N},                                               // TensorB (ptr, ldb)
        {d_C, N},                                               // TensorC (ptr, ldc)
        {d_C, N},                                               // TensorD (输出, ptr, ldd)
        {1.0f, 0.0f}                                            // alpha, beta
    );

    // 预热
    gemm_op(args);
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        gemm_op(args);
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

// cuBLAS GEMM（GPU，手写）
GpuTimingResult cublas_gemm_run(CRMatrix h_A, CRMatrix h_B, RMatrix h_C,
                                 CInt M, CInt N, CInt K, CInt iterations) {
    PFloat d_A = nullptr, d_B = nullptr, d_C = nullptr;
    CSize size_A = M * K * FSIZE;
    CSize size_B = K * N * FSIZE;
    CSize size_C = M * N * FSIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    cublasHandle_t handle;
    cublasCreate(&handle);

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
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
    cout << "   CUTLASS GEMM 性能基准测试\n";
    cout << "========================================\n";
    cout << "矩阵维度：A(" << M << " x " << K << ") * B(" << K << " x " << N << ")\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    Matrix h_A(M * K), h_B(K * N);
    Matrix h_C_cpu(M * N, 0.0f);
    Matrix h_C_cublas(M * N, 0.0f);

    srand(42);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand() % 100) / 100.0f;

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    cpuTimer.stop();
    cout << "CPU 执行时间：" << setw(8) << cpuTimer.elapsed_ms() << " ms\n\n";

#if HAS_CUTLASS
    // CUTLASS GEMM
    cout << "--- CUTLASS GEMM ---\n";
    Matrix h_C_cutlass(M * N, 0.0f);
    GpuTimingResult res_cutlass = cutlass_gemm_run(h_A, h_B, h_C_cutlass, M, N, K, iterations);
    double cutlass_tflops = (flops / 1e12) / (res_cutlass.kernel_ms / 1000.0);
    cout << "Kernel 执行时间：" << setw(8) << res_cutlass.kernel_ms << " ms\n";
    cout << "计算性能：       " << setprecision(2) << cutlass_tflops << " TFLOPS\n\n";
#else
    cout << "[提示] CUTLASS 未安装，跳过 CUTLASS GEMM 测试\n";
    cout << "       请设置 CUTLASS_DIR 环境变量后重新编译\n\n";
#endif

    // cuBLAS 对比
    cout << "--- cuBLAS SGEMM (对比基准) ---\n";
    GpuTimingResult res_cublas = cublas_gemm_run(h_A, h_B, h_C_cublas, M, N, K, iterations);
    double cublas_tflops = (flops / 1e12) / (res_cublas.kernel_ms / 1000.0);
    cout << "Kernel 执行时间：" << setw(8) << res_cublas.kernel_ms << " ms\n";
    cout << "计算性能：       " << setprecision(2) << cublas_tflops << " TFLOPS\n\n";

    // 性能对比
    cout << "--- 性能对比 ---\n";
#if HAS_CUTLASS
    cout << "CUTLASS GEMM:  " << setprecision(2) << cutlass_tflops << " TFLOPS\n";
#endif
    cout << "cuBLAS SGEMM:  " << setprecision(2) << cublas_tflops << " TFLOPS\n";
#if HAS_CUTLASS
    cout << "CUTLASS/cuBLAS: " << setprecision(1) << (cutlass_tflops / cublas_tflops * 100.0) << "%\n";
#endif
    cout << "\n";

    // 验证
    cout << "--- 结果验证 ---\n";
    bool pass_cublas = verify_results(h_C_cublas, h_C_cpu, "cuBLAS SGEMM");
#if HAS_CUTLASS
    bool pass_cutlass = verify_results(h_C_cutlass, h_C_cpu, "CUTLASS GEMM");
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
