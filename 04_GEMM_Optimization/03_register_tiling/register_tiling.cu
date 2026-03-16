// Register Tiling GEMM - 寄存器分块高性能矩阵乘法 + cuBLAS 对比 Benchmark
// 核心优化：Shared Memory 分块 + 线程级寄存器分块（每线程计算 TM×TN 子块）
#include <code_abbreviation.h>
#include <cublas_v2.h>

// ========================= 常量定义 =========================

// Block Tile 大小 (BM × BN): 一个 Thread Block 负责 C 的 BM×BN 子块
constexpr int BM = 128;
constexpr int BN = 128;
// K 方向 Tile 大小
constexpr int BK = 8;
// Thread Tile 大小 (TM × TN): 每个线程负责 C 的 TM×TN 子块
constexpr int TM = 8;
constexpr int TN = 8;

// ========================= GPU Kernel =========================

// 寄存器分块 GEMM（GPU kernel，手写）
//
// 核心思想：
//   1. 每个 Block 负责 C 的一个 BM×BN 子块
//   2. 沿 K 维度迭代，每次将 A 的 BM×BK 和 B 的 BK×BN 加载到 Shared Memory
//   3. 每个线程从 Shared Memory 加载自己需要的数据到寄存器，计算 TM×TN 个累加结果
//   4. 这样每个 Shared Memory 中的数据被复用 TM（或 TN）次，减少访存
//
// 线程布局：
//   Block 内共 (BM/TM) × (BN/TN) 个线程 = 16 × 16 = 256 个线程
//   每个线程计算 C 的 TM×TN = 8×8 = 64 个元素（保存在寄存器中）
__global__ void register_tiling_gemm(CPFloat A, CPFloat B, PFloat C,
                                      CInt M, CInt N, CInt K) {
    // 当前 Block 负责 C 的起始行列
    const int bRow = blockIdx.y * BM;
    const int bCol = blockIdx.x * BN;

    // 线程在 Block 内的线性索引
    const int threadId = threadIdx.x;
    // 线程在 Thread Tile 网格中的行列位置
    const int threadRow = threadId / (BN / TN);  // 0..15
    const int threadCol = threadId % (BN / TN);  // 0..15

    // Shared Memory: 存储当前 K 维 Tile 的 A、B 子块
    // 💡 进阶面试考点与避坑指南 (Bank Conflict & Padding):
    // 在这段代码的阶段 3 (计算时)，同一个 Warp 内的 16 个线程在同一时刻需要读取 sB 的同一行的不同列。
    // 由于 TN=8，相邻线程访问 sB 的距离为 8 个 float，导致 Bank(0) 与 Bank(8)、Bank(16)...重叠，这会产生 4-way 甚至 8-way Bank Conflict。
    // 工业界的标准解法：定义 `__shared__ float sB[BK][BN + PAD]`（比如 PAD=1），
    // 通过错开步长即可完美打散 Bank Conflict。此处为了代码可读性和教学清晰度，暂且保留未经 PAD 污染的裸阵列。
    __shared__ float sA[BM][BK];  // BM×BK = 128×8
    __shared__ float sB[BK][BN];  // BK×BN = 8×128

    // 寄存器: 存储当前线程的 TM×TN 个结果累加值
    float regC[TM][TN] = {0.0f};
    // 寄存器: 临时存储从 Shared Memory 加载的 A、B 列/行
    float regA[TM];
    float regB[TN];

    // 计算每个线程在 H2D 搬运中负责的行列
    // 搬运 A: BM×BK 个元素，由 blockDim.x 个线程协作完成
    const int numThreads = blockDim.x;  // = (BM/TM)*(BN/TN) = 256
    const int strideA = numThreads / BK;  // 每次搬运 strideA 行
    const int innerRowA = threadId / BK;
    const int innerColA = threadId % BK;
    // 搬运 B: BK×BN 个元素
    const int strideB = numThreads / BN;
    const int innerRowB = threadId / BN;
    const int innerColB = threadId % BN;

    // 沿 K 维度迭代
    for (int bk = 0; bk < K; bk += BK) {

        // === 阶段 1: 将 A[bRow:bRow+BM, bk:bk+BK] 搬到 sA ===
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            int aRow = bRow + innerRowA + loadOffset;
            int aCol = bk + innerColA;
            if (aRow < M && aCol < K) {
                sA[innerRowA + loadOffset][innerColA] = A[aRow * K + aCol];
            } else {
                sA[innerRowA + loadOffset][innerColA] = 0.0f;
            }
        }

        // === 阶段 2: 将 B[bk:bk+BK, bCol:bCol+BN] 搬到 sB ===
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            int bRowIdx = bk + innerRowB + loadOffset;
            int bColIdx = bCol + innerColB;
            if (bRowIdx < K && bColIdx < N) {
                sB[innerRowB + loadOffset][innerColB] = B[bRowIdx * N + bColIdx];
            } else {
                sB[innerRowB + loadOffset][innerColB] = 0.0f;
            }
        }

        __syncthreads();

        // === 阶段 3: 从 Shared Memory 加载到寄存器并计算 ===
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // 加载 A 的一列到寄存器 (TM 个元素)
            for (int i = 0; i < TM; ++i) {
                regA[i] = sA[threadRow * TM + i][dotIdx];
            }
            // 加载 B 的一行到寄存器 (TN 个元素)
            for (int j = 0; j < TN; ++j) {
                regB[j] = sB[dotIdx][threadCol * TN + j];
            }
            // 外积累加: TM × TN 次 FMA
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    regC[i][j] = fmaf(regA[i], regB[j], regC[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // === 阶段 4: 将结果从寄存器写回 Global Memory ===
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int cRow = bRow + threadRow * TM + i;
            int cCol = bCol + threadCol * TN + j;
            if (cRow < M && cCol < N) {
                C[cRow * N + cCol] = regC[i][j];
            }
        }
    }
}

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
    int max_diff_idx = 0;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float diff = fabs(gpu_result[i] - cpu_result[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = static_cast<int>(i);
        }
        if (diff > epsilon) {
            error_count++;
        }
    }

    if (error_count > 0) {
        cout << "✗ " << kernel_name << " FAILED: " << error_count << " 个元素超出误差阈值\n";
        cout << "  最大差异位于索引 " << max_diff_idx
             << "：GPU=" << gpu_result[max_diff_idx]
             << ", CPU=" << cpu_result[max_diff_idx]
             << ", 差异=" << max_diff << "\n";
        return false;
    }

    cout << "✓ " << kernel_name << " PASSED (最大误差 " << max_diff << ")\n";
    return true;
}

// ========================= GPU 封装 =========================

// Register Tiling GEMM GPU 封装（GPU，手写）
GpuTimingResult register_tiling_gpu(CRMatrix h_A, CRMatrix h_B, RMatrix h_C,
                                     CInt M, CInt N, CInt K, CInt iterations) {
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

    // 线程配置：每个 Block 有 (BM/TM) × (BN/TN) = 16×16 = 256 个线程
    const dim3 block((BM / TM) * (BN / TN));  // 256
    const dim3 grid(cdiv(N, BN), cdiv(M, BM));

    // 预热
    register_tiling_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        register_tiling_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

// cuBLAS GEMM 对比 Benchmark（GPU，手写）
GpuTimingResult cublas_gemm_benchmark(CRMatrix h_A, CRMatrix h_B, RMatrix h_C,
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

    // cuBLAS uses column-major, so we compute C^T = B^T * A^T
    // which is equivalent to C = A * B in row-major
    float alpha = 1.0f, beta = 0.0f;

    // 预热
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
    // 矩阵维度: A(M x K) * B(K x N) = C(M x N)
    CInt M = 2048;
    CInt N = 2048;
    CInt K = 2048;
    CInt iterations = 20;

    CSize size_A = M * K * FSIZE;
    CSize size_B = K * N * FSIZE;
    CSize size_C = M * N * FSIZE;
    const double total_mb = (size_A + size_B + size_C) / (1024.0 * 1024.0);
    const double flops = 2.0 * M * N * K;  // 2*M*N*K 次浮点运算

    printDeviceInfo();

    cout << "========================================\n";
    cout << "  Register Tiling GEMM 性能基准测试\n";
    cout << "========================================\n";
    cout << "矩阵维度：A(" << M << " x " << K << ") * B(" << K << " x " << N << ") = C(" << M << " x " << N << ")\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB (A + B + C)\n";
    cout << "Block Tile: BM=" << BM << " BN=" << BN << " BK=" << BK << "\n";
    cout << "Thread Tile: TM=" << TM << " TN=" << TN << "\n";
    cout << "线程数/Block: " << (BM/TM)*(BN/TN) << "\n";
    cout << "每线程计算量: " << TM << "×" << TN << " = " << TM*TN << " 个 C 元素\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 数据初始化
    Matrix h_A(M * K), h_B(K * N);
    Matrix h_C_cpu(M * N, 0.0f);
    Matrix h_C_reg(M * N, 0.0f);
    Matrix h_C_cublas(M * N, 0.0f);

    srand(42);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand() % 100) / 100.0f;

    // CPU 计时 (过大则跳过以防阻塞)
    cout << "--- CPU 计时 (若尺寸较小) ---\n";
    CpuTimer cpuTimer;
    double cpu_time_ms = 0.0;
    double cpu_gflops = 0.0;
    const bool cpu_enabled = (M <= 512 && N <= 512 && K <= 512);
    if (cpu_enabled) {
        cpuTimer.start();
        gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
        cpuTimer.stop();
        cpu_time_ms = cpuTimer.elapsed_ms();
        cpu_gflops = (flops / 1e9) / (cpu_time_ms / 1000.0);
        cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
        cout << "CPU 计算性能：   " << setprecision(2) << cpu_gflops << " GFLOPS\n";
    } else {
        cout << "矩阵尺寸过大，跳过 CPU 参考计算。\n";
    }
    cout << "\n";

    // GPU 版本 1：Register Tiling GEMM
    cout << "--- GPU 版本 1: Register Tiling GEMM (手写) ---\n";
    GpuTimingResult res_reg = register_tiling_gpu(h_A, h_B, h_C_reg, M, N, K, iterations);
    double reg_tflops = (flops / 1e12) / (res_reg.kernel_ms / 1000.0);
    cout << "H2D 传输时间：   " << setw(8) << res_reg.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_reg.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_reg.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_reg.total_ms << " ms\n";
    cout << "计算性能：       " << setprecision(2) << reg_tflops << " TFLOPS\n";
    cout << "\n";

    // GPU 版本 2：cuBLAS SGEMM（对比基准）
    cout << "--- GPU 版本 2: cuBLAS SGEMM (对比基准) ---\n";
    GpuTimingResult res_cublas = cublas_gemm_benchmark(h_A, h_B, h_C_cublas, M, N, K, iterations);
    double cublas_tflops = (flops / 1e12) / (res_cublas.kernel_ms / 1000.0);
    cout << "H2D 传输时间：   " << setw(8) << res_cublas.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_cublas.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_cublas.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_cublas.total_ms << " ms\n";
    cout << "计算性能：       " << setprecision(2) << cublas_tflops << " TFLOPS\n";
    cout << "\n";

    // 性能对比分析
    cout << "--- 性能对比分析 ---\n";
    double ratio = reg_tflops / cublas_tflops * 100.0;
    cout << "Register Tiling: " << setprecision(2) << reg_tflops << " TFLOPS\n";
    cout << "cuBLAS SGEMM:    " << setprecision(2) << cublas_tflops << " TFLOPS\n";
    cout << "手写/cuBLAS 比率: " << setprecision(1) << ratio << "%\n";

    if (cpu_enabled) {
        double speedup_vs_cpu = cpu_time_ms / res_reg.kernel_ms;
        cout << "CPU vs 手写 GEMM 加速比：" << setprecision(1) << speedup_vs_cpu << "x\n";
    } else {
        cout << "CPU vs 手写 GEMM 加速比：N/A (未运行 CPU 参考)\n";
    }
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = false;
    bool pass2 = false;
    if (M <= 512 && N <= 512 && K <= 512) {
        pass1 = verify_results(h_C_reg, h_C_cpu, "Register Tiling GEMM");
        pass2 = verify_results(h_C_cublas, h_C_cpu, "cuBLAS SGEMM");

        if (pass1 && pass2) {
            cout << "✓ GPU/CPU 结果一致性验证通过\n";
        } else {
            cout << "✗ GPU/CPU 结果存在差异\n";
        }
    } else {
        cout << "✓ 结果验证跳过 (因矩阵尺寸过大，CPU 基准未计算)\n";
    }

    cout << "\n========================================\n";

    return 0;
}
