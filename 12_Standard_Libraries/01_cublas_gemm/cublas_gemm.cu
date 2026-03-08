// cuBLAS GEMM - 使用标准库进行矩阵乘法
#include <code_abbreviation.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <random>



// cuBLAS 初始化和销毁的 RAII 封装
class CuBLASContext {
public:
    cublasHandle_t handle;
    
    CuBLASContext() {
        cublasCreate(&handle);
    }
    
    ~CuBLASContext() {
        cublasDestroy(handle);
    }
};

class CuBLASLtContext {
public:
    cublasLtHandle_t handle;
    
    CuBLASLtContext() {
        cublasLtCreate(&handle);
    }
    
    ~CuBLASLtContext() {
        cublasLtDestroy(handle);
    }
};


// CPU 矩阵乘法参考基准 (CPU，手写)
void sgemm_cpu(CRMatrix A, CRMatrix B, RMatrix C, CInt M, CInt N, CInt K, CFloat alpha = 1.0f, CFloat beta = 0.0f) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

void batched_sgemm_cpu(const std::vector<Matrix>& A_batch, const std::vector<Matrix>& B_batch, std::vector<Matrix>& C_batch, CInt M, CInt N, CInt K, CInt batch_size) {
    for (int b = 0; b < batch_size; ++b) {
        sgemm_cpu(A_batch[b], B_batch[b], C_batch[b], M, N, K);
    }
}


// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, CInt n, const string& kernel_name) {
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(gpu_result[i] - cpu_result[i]) > 1e-3f) { // 矩阵乘法累加误差较大，适当放宽
            cout << "✗ " << kernel_name << " FAILED: 索引 " << i 
                 << " 结果 " << gpu_result[i] << " (期望 " << cpu_result[i] << ")\n";
            success = false;
            break;
        }
    }
    if (success) {
        cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result[0] << " (期望 " << cpu_result[0] << ")\n";
    }
    return success;
}


// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      // Host to Device 传输时间
    float kernel_ms;   // Kernel 执行时间（多次平均）
    float d2h_ms;      // Device to Host 传输时间
    float total_ms;    // 总时间
};


// 基础 SGEMM (单精度) 封装 (GPU，手写)
GpuTimingResult sgemm_basic_gpu(cublasHandle_t handle, CRMatrix h_A, CRMatrix h_B, RMatrix h_C, CInt M, CInt N, CInt K, CInt iterations) {
    PFloat d_A = nullptr, d_B = nullptr, d_C = nullptr;
    CSize size_A = M * K * FSIZE;
    CSize size_B = K * N * FSIZE;
    CSize size_C = M * N * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));
    
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // 预热：由于 C/C++ 是行主序 (Row-Major)，而 cuBLAS 是列主序 (Col-Major)
    // 根据矩阵转置公式 (A*B)^T = B^T * A^T
    // 当我们向 cuBLAS 传入一个行主序的矩阵，cuBLAS 会认为它是某个矩阵的转置。
    // 即传进去的行主序 A 相当于 cuBLAS 需要的 A^T。
    // 我们要计算 C = A * B，利用 C^T = B^T * A^T。
    // 把行主序的 B 作为第一个参数（被认为是某个矩阵 M_b 的转置），把 A 作为第二个参数。
    // cuBLAS 计算出 M_b^T * M_a^T = (M_a * M_b)^T。
    // 结果被按列主序写回 C，但因为我们算的是 C^T 的列主序，它恰好等同于 C 的行主序。
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,  // ldb = N (因为是连续存放，Row-Major B 的物理列宽为 N)
                d_A, K,  // lda = K
                &beta,
                d_C, N); // ldc = N
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
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

// Strided Batched GEMM 封装 (GPU，手写)
GpuTimingResult strided_batched_sgemm_gpu(cublasHandle_t handle, CRMatrix h_A, CRMatrix h_B, RMatrix h_C, CInt M, CInt N, CInt K, CInt batch_size, CInt iterations) {
    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;
    
    PFloat d_A = nullptr, d_B = nullptr, d_C = nullptr;
    CSize size_A = strideA * batch_size * FSIZE;
    CSize size_B = strideB * batch_size * FSIZE;
    CSize size_C = strideC * batch_size * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_B, N, strideB,
                              d_A, K, strideA,
                              &beta,
                              d_C, N, strideC,
                              batch_size);
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K, &alpha,
                                  d_B, N, strideB, d_A, K, strideA, &beta,
                                  d_C, N, strideC, batch_size);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
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

// cuBLASLt GEMM 封装 (GPU，手写)
GpuTimingResult cublaslt_gemm_gpu(cublasLtHandle_t handle, CRMatrix h_A, CRMatrix h_B, RMatrix h_C, CInt M, CInt N, CInt K, CInt iterations) {
    PFloat d_A = nullptr, d_B = nullptr, d_C = nullptr;
    CSize size_A = M * K * FSIZE;
    CSize size_B = K * N * FSIZE;
    CSize size_C = M * N * FSIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));
    
    // 配置 Descriptor (记得转置参数映射)
    // 根据转置规则，我们实际传入 cuBLASLt 的是 B(=A_lt), A(=B_lt) 得到 C(=C_lt)
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, N, K, N); // 对应原 B
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, K, M, K); // 对应原 A
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, N, M, N); // 对应原 C

    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    
    // Heuristics 推断
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t workspaceSize = 32 * 1024 * 1024; // 提供 32MB workspace 给算法做缓冲
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
    
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutB, layoutA, layoutC, layoutC, pref, 1, &heuristicResult, &returnedResults);
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // 预热 (注意传入顺序和指针)
    cublasLtMatmul(handle, matmulDesc,
                   &alpha, d_B, layoutB, d_A, layoutA,
                   &beta, d_C, layoutC, d_C, layoutC,
                   &heuristicResult.algo, d_workspace, workspaceSize, 0);
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        cublasLtMatmul(handle, matmulDesc,
                       &alpha, d_B, layoutB, d_A, layoutA,
                       &beta, d_C, layoutC, d_C, layoutC,
                       &heuristicResult.algo, d_workspace, workspaceSize, 0);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtMatmulPreferenceDestroy(pref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_workspace));
    
    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt M = 1024;
    CInt N = 1024;
    CInt K = 1024;
    CInt batch_size = 8;
    CInt iterations = 50;

    CSize size_A_single = M * K * FSIZE;
    CSize size_B_single = K * N * FSIZE;
    CSize size_C_single = M * N * FSIZE;
    const double total_mb_single = (size_A_single + size_B_single + size_C_single) / (1024.0 * 1024.0);
    const double total_mb_batch = total_mb_single * batch_size;

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      cuBLAS GEMM 官方标准库性能测试\n";
    cout << "========================================\n";
    cout << "矩阵形状：M=" << M << ", N=" << N << ", K=" << K << "\n";
    cout << "单算例数据量：" << fixed << setprecision(2) << total_mb_single << " MB\n";
    cout << "Batch 规模  ：" << batch_size << " 个矩阵组合 (" << total_mb_batch << " MB)\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 设置数据
    std::mt19937 gen(2026);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    Matrix h_A(M * K);
    Matrix h_B(K * N);
    Matrix h_C_cpu(M * N, 0.0f);
    Matrix h_C_sgemm(M * N, 0.0f);
    Matrix h_C_sgemm_lt(M * N, 0.0f);

    for (int i = 0; i < h_A.size(); ++i) h_A[i] = dist(gen);
    for (int i = 0; i < h_B.size(); ++i) h_B[i] = dist(gen);
    
    // Batch 数据设置
    Matrix h_A_batch(M * K * batch_size);
    Matrix h_B_batch(K * N * batch_size);
    Matrix h_C_batch_cpu(M * N * batch_size, 0.0f);
    Matrix h_C_batch_gpu(M * N * batch_size, 0.0f);
    for (int i = 0; i < h_A_batch.size(); ++i) h_A_batch[i] = dist(gen);
    for (int i = 0; i < h_B_batch.size(); ++i) h_B_batch[i] = dist(gen);

    // CPU 计算
    cout << "--- CPU 计时 (M="<<M<<", N="<<N<<", K="<<K<<") ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    // 只跑一次防卡死
    sgemm_cpu(h_A, h_B, h_C_cpu, M, N, K);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 单算例执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // Batched CPU 仅验证正确性，不单独剥离其测试计时避免过长
    for(int b=0; b<batch_size; ++b) {
        // 获取各个独立矩阵的首指针去执行
        Matrix A_slice(h_A_batch.begin() + b * M * K, h_A_batch.begin() + (b + 1) * M * K);
        Matrix B_slice(h_B_batch.begin() + b * K * N, h_B_batch.begin() + (b + 1) * K * N);
        Matrix C_slice(M * N, 0.0f);
        sgemm_cpu(A_slice, B_slice, C_slice, M, N, K);
        // 写回
        std::copy(C_slice.begin(), C_slice.end(), h_C_batch_cpu.begin() + b * M * N);
    }
    
    // GPU 环境配置
    CuBLASContext ctx;
    CuBLASLtContext ctxLt;

    // GPU 1: 基础 cublasSgemm
    cout << "--- GPU 版本 1: 基础 cublasSgemm ---\n";
    GpuTimingResult res_basic = sgemm_basic_gpu(ctx.handle, h_A, h_B, h_C_sgemm, M, N, K, iterations);
    cout << "H2D 传输时间：   " << setw(8) << res_basic.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_basic.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_basic.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_basic.total_ms << " ms\n";
    cout << "\n";

    // GPU 2: 高级 cublasLtMatmul
    cout << "--- GPU 版本 2: 启发式 cublasLtMatmul ---\n";
    GpuTimingResult res_lt = cublaslt_gemm_gpu(ctxLt.handle, h_A, h_B, h_C_sgemm_lt, M, N, K, iterations);
    cout << "H2D 传输时间：   " << setw(8) << res_lt.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_lt.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_lt.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_lt.total_ms << " ms\n";
    cout << "\n";

    // GPU 3: Strided Batched GEMM
    cout << "--- GPU 版本 3: Strided Batched SGEMM (Batch="<<batch_size<<") ---\n";
    GpuTimingResult res_batch = strided_batched_sgemm_gpu(ctx.handle, h_A_batch, h_B_batch, h_C_batch_gpu, M, N, K, batch_size, iterations);
    cout << "H2D 传输时间：   " << setw(8) << res_batch.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_batch.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_batch.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_batch.total_ms << " ms\n";
    cout << "\n";


    // 性能分析
    cout << "--- 性能分析 (TFLOPS) ---\n";
    // 浮点运算量：M*N*(2K-1) 约等于 2MNK
    double tflops_single = 2.0 * M * N * K;
    
    double tflops_cublas   = (tflops_single / 1e12) / (res_basic.kernel_ms / 1000.0);
    double tflops_cublaslt = (tflops_single / 1e12) / (res_lt.kernel_ms / 1000.0);
    double tflops_strided  = (tflops_single * batch_size / 1e12) / (res_batch.kernel_ms / 1000.0);

    cout << "CPU vs GPU (基础) 加速比：" << setprecision(2) << cpu_time_ms / res_basic.kernel_ms << "x\n\n";

    cout << "cublasSgemm       算力：" << setprecision(2) << setw(8) << tflops_cublas << " TFLOPS\n";
    cout << "cublasLtMatmul    算力：" << setprecision(2) << setw(8) << tflops_cublaslt << " TFLOPS\n";
    cout << "StridedBatched    算力：" << setprecision(2) << setw(8) << tflops_strided << " TFLOPS (相比单算例通常隐藏了 Kernel 启动开销)\n";
    cout << "(RTX 4090 FP32 理论峰值：~82.58 TFLOPS)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_C_sgemm, h_C_cpu, M * N, "cublasSgemm\t");
    bool pass2 = verify_results(h_C_sgemm_lt, h_C_cpu, M * N, "cublasLtMatmul\t");
    bool pass3 = verify_results(h_C_batch_gpu, h_C_batch_cpu, M * N * batch_size, "StridedBatched\t");

    if (pass1 && pass2 && pass3) {
        cout << "✓ GPU/CPU 一致性验证全部通过 (处理了 Row-Major 转置机制)\n";
    } else {
        cout << "✗ GPU/CPU 验证存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
