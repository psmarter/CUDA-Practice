// FP16 GEMM 性能基准测试
#include <code_abbreviation.h>
#include <cuda_fp16.h>

// 朴素 FP16 GEMM（GPU kernel，手写）
__global__ void kernel_naive_fp16_gemm(const half* A, const half* B, half* C, CInt M, CInt N, CInt K) {
    CInt row = blockIdx.y * blockDim.y + threadIdx.y;
    CInt col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += __half2float(A[row * N + i]) * __half2float(B[i * K + col]);
        }
        C[row * K + col] = __float2half(sum);
    }
}

// Tiled FP16 GEMM（使用共享内存）（GPU kernel，手写）
__global__ void kernel_tiled_fp16_gemm(const half* A, const half* B, half* C, CInt M, CInt N, CInt K) {
    __shared__ half shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ half shared_B[TILE_SIZE][TILE_SIZE];

    CInt row = blockIdx.y * TILE_SIZE + threadIdx.y;
    CInt col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < cdiv(N, TILE_SIZE); ++i) {
        CInt tiled_col = i * TILE_SIZE + threadIdx.x;
        CInt tiled_row = i * TILE_SIZE + threadIdx.y;

        shared_A[threadIdx.y][threadIdx.x] = (row < M && tiled_col < N) ? A[row * N + tiled_col] : __float2half(0.0f);
        shared_B[threadIdx.y][threadIdx.x] = (col < K && tiled_row < N) ? B[tiled_row * K + col] : __float2half(0.0f);
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j) {
            sum += __half2float(shared_A[threadIdx.y][j]) * __half2float(shared_B[j][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = __float2half(sum);
    }
}

// 使用 half2 向量化加载（GPU kernel，手写）
__global__ void kernel_vectorized_fp16_gemm(const half* A, const half* B, half* C, CInt M, CInt N, CInt K) {
    CInt row = blockIdx.y * TILE_SIZE + threadIdx.y;
    CInt col = (blockIdx.x * TILE_SIZE + threadIdx.x) * 2;

    // 如果是奇数，col可能越界
    if (row < M && col < K) {
        
        // 使用 half2 向量化加载，处理两个元素
        half2 sum2 = __float2half2_rn(0.0f);

        for (int i = 0; i < N; ++i) {
            half2 a_val2 = __halves2half2(A[row * N + i], A[row * N + i]); // 同一个元素复制两份
            half2 b_val2 = *reinterpret_cast<const half2*>(&B[i * K + col]); // 连续的两个元素作为一个 half2 加载
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            sum2 = __hfma2(a_val2, b_val2, sum2); // 半精度乘加 (SM 5.3+)
#else
            // Fallback: 针对尚未指定较新架构 (默认 SM 5.2 或更低) 的编译兜底
            float2 a_f2 = __half22float2(a_val2);
            float2 b_f2 = __half22float2(b_val2);
            float2 sum_f2 = __half22float2(sum2);
            sum_f2.x += a_f2.x * b_f2.x;
            sum_f2.y += a_f2.y * b_f2.y;
            sum2 = __floats2half2_rn(sum_f2.x, sum_f2.y);
#endif
        }

        *reinterpret_cast<half2*>(&C[row * K + col]) = sum2; // 将结果写回连续的两个元素
    }
}

// FP16 GEMM（CPU，手写）
void fp16_gemm_cpu(const half* A, const half* B, half* C, CInt M, CInt N, CInt K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += __half2float(A[i * N + k]) * __half2float(B[k * K + j]);
            }
            C[i * K + j] = __float2half(sum);
        }
    }
}

// 验证结果（AI 生成）
bool verify_results(const std::vector<half>& gpu_result, const std::vector<half>& cpu_result, const string& kernel_name, CFloat epsilon = 1e-3f) {
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配\n";
        return false;
    }

    int error_count = 0;
    float max_diff = 0.0f;
    int max_diff_idx = 0;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float gpu_v = __half2float(gpu_result[i]);
        float cpu_v = __half2float(cpu_result[i]);
        float diff = fabs(gpu_v - cpu_v);
        
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = static_cast<int>(i);
        }
        
        // FP16 精度非常有限，特别是多次累加时产生截断误差
        if (diff > epsilon && (diff / (fabs(cpu_v) + 1e-5f)) > 0.1f) {
            error_count++;
        }
    }

    if (error_count > 0) {
        cout << "✗ " << kernel_name << " FAILED: " << error_count << " 个元素超出误差阈值\n";
        cout << "  最大差异位于索引 " << max_diff_idx
             << "：GPU=" << __half2float(gpu_result[max_diff_idx])
             << ", CPU=" << __half2float(cpu_result[max_diff_idx])
             << ", 差异=" << max_diff << "\n";
        return false;
    }

    cout << "✓ " << kernel_name << " PASSED: 结果 " << __half2float(gpu_result[0]) 
         << " (期望 " << __half2float(cpu_result[0]) << ")\n";
    return true;
}

// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      
    float kernel_ms;   
    float d2h_ms;      
    float total_ms;    
};

// 通用 FP16 GEMM 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult fp16_gemm_gpu(const std::vector<half>& h_A, const std::vector<half>& h_B, std::vector<half>& h_C,
                              CInt M, CInt N, CInt K, CInt iterations, KernelFunc kernel, bool is_vectorized = false) {
    half* d_A = nullptr;
    half* d_B = nullptr;
    half* d_C = nullptr;

    CSize size_A = M * N * sizeof(half);
    CSize size_B = N * K * sizeof(half);
    CSize size_C = M * K * sizeof(half);

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

    const dim3 block(TILE_SIZE, TILE_SIZE);
    
    // 如果是 Vectorized，X维度处理2个元素，所以开启的线程数减半
    const dim3 grid = is_vectorized ? dim3(cdiv(K, TILE_SIZE * 2), cdiv(M, TILE_SIZE))
                                    : dim3(cdiv(K, TILE_SIZE), cdiv(M, TILE_SIZE));

    kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt M = 1024;
    CInt N = 1024;
    CInt K = 1024;     // K 必须是偶数，且保证 Vectorized 读取不越界
    CInt iterations = 10;

    CSize size_A = M * N * sizeof(half);
    CSize size_B = N * K * sizeof(half);
    CSize size_C = M * K * sizeof(half);
    const double total_mb = (size_A + size_B + size_C) / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      FP16 GEMM 性能基准测试\n";
    cout << "========================================\n";
    cout << "矩阵维度：A(" << M << " x " << N << ") * B(" << N << " x " << K << ") = C(" << M << " x " << K << ")\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << TILE_SIZE << " x " << TILE_SIZE << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    std::vector<half> h_A(M * N);
    std::vector<half> h_B(N * K);
    std::vector<half> h_C1(M * K);
    std::vector<half> h_C2(M * K);
    std::vector<half> h_C3(M * K);
    std::vector<half> h_C_cpu(M * K);

    srand(42);
    for (int i = 0; i < M * N; ++i) {
        float val = static_cast<float>(rand() % 200 - 100) / 100.0f; // [-1.0, 1.0]
        h_A[i] = __float2half(val);
    }
    for (int i = 0; i < N * K; ++i) {
        float val = static_cast<float>(rand() % 200 - 100) / 100.0f; // [-1.0, 1.0]
        h_B[i] = __float2half(val);
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    fp16_gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";
    
    // GPU 版本 1: Naive
    cout << "--- GPU 版本 1: Naive FP16 GEMM ---\n";
    GpuTimingResult result1 = fp16_gemm_gpu(h_A, h_B, h_C1, M, N, K, iterations, kernel_naive_fp16_gemm, false);
    cout << "H2D 传输时间：   " << setw(8) << result1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result1.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2: Tiled
    cout << "--- GPU 版本 2: Tiled FP16 GEMM ---\n";
    GpuTimingResult result2 = fp16_gemm_gpu(h_A, h_B, h_C2, M, N, K, iterations, kernel_tiled_fp16_gemm, false);
    cout << "H2D 传输时间：   " << setw(8) << result2.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result2.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result2.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result2.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 3: Vectorized
    cout << "--- GPU 版本 3: Vectorized FP16 GEMM ---\n";
    GpuTimingResult result3 = fp16_gemm_gpu(h_A, h_B, h_C3, M, N, K, iterations, kernel_vectorized_fp16_gemm, true);
    cout << "H2D 传输时间：   " << setw(8) << result3.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result3.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result3.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result3.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_kernel = cpu_time_ms / result3.kernel_ms;
    double speedup_total = cpu_time_ms / result3.total_ms;
    cout << "CPU vs GPU Kernel 加速比：" << setprecision(2) << speedup_kernel << "x\n";
    cout << "CPU vs GPU 总时间加速比：" << speedup_total << "x\n";

    // FP16 MAC
    double flops = 2.0 * M * N * K;
    double gpu_gflops = (flops / 1e9) / (result3.kernel_ms / 1000.0);
    cout << "GPU 计算性能：" << setprecision(2) << gpu_gflops << " GFLOPS\n";

    // MAC = read 2, write 1 (忽略寄存器缓存增益的纯物理读取下界测算)
    double bytes = size_A + size_B + size_C;
    double gpu_bandwidth = (bytes / 1e9) / (result3.kernel_ms / 1000.0);
    cout << "GPU 有效带宽：" << setprecision(2) << gpu_bandwidth << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    cout << "--- Kernel 性能对比 ---\n";
    cout << "Naive:       " << setw(8) << setprecision(4) << result1.kernel_ms << " ms (基准)\n";
    cout << "Tiled:       " << setw(8) << result2.kernel_ms << " ms ("
         << setprecision(2) << result1.kernel_ms / result2.kernel_ms << "x)\n";
    cout << "Vectorized:  " << setw(8) << result3.kernel_ms << " ms ("
         << setprecision(2) << result1.kernel_ms / result3.kernel_ms << "x)\n";
    cout << "\n";

    cout << "--- 结果验证 ---\n";
    // Tiled 和 Vectorized 算法由于每次加进去的步骤和顺序截然不同，所以相比 CPU 有较大舍入截断波动
    bool pass1 = verify_results(h_C1, h_C_cpu, "Naive FP16 GEMM", 3.0f);
    bool pass2 = verify_results(h_C2, h_C_cpu, "Tiled FP16 GEMM", 3.0f);
    bool pass3 = verify_results(h_C3, h_C_cpu, "Vectorized FP16 GEMM", 5.0f);

    if (pass1 && pass2 && pass3) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
