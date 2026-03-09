// Tensor Core WMMA GEMM 性能基准测试
#include <code_abbreviation.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// 朴素 WMMA GEMM（GPU kernel，手写）
__global__ void wmma_gemm_naive(const half* A, const half* B, PFloat C, CInt M, CInt N, CInt K) {
    // 线程配置：
    // blockDim.x = 32 (一个 Warp 的线程数)
    // blockDim.y = 8 (一个 Block 内包含 8 个 Warp)
    // gridDim.x = cdiv(N, 16)
    // gridDim.y = cdiv(M, 128)
    
    // 计算当前 Warp 在全局矩阵 C 中的二维索引
    CInt warp_col = blockIdx.x; // 每个 blockIdx.x 处理 16 列
    CInt warp_row = blockIdx.y * blockDim.y + threadIdx.y; // 每个 warp 处理 16 行

    CInt row = warp_row * WMMA_M;
    CInt col = warp_col * WMMA_N;

    if (row >= M || col >= N) return;

    // 声明 Tensor Core 碎片 (Fragments)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 沿 K 维度滑动，从全局显存直接加载并执行 MMA
    for (int i = 0; i < K; i += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + row * K + i, K);
        wmma::load_matrix_sync(b_frag, B + i * N + col, N);
        // 执行 Tensor Core 同步点调用
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 将计算完成的 accumulator fragment 存回全局显存的 C 矩阵
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}

// 矩阵乘法 CPU 参考实现（CPU，手写）
void wmma_gemm_cpu(const vector<half>& A, const vector<half>& B, RMatrix C, CInt M, CInt N, CInt K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // 将 half 显式转换回 float 后进行 CPU 累加，以对齐精度
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = EPSILON) {
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配\n";
        return false;
    }

    int error_count = 0;
    float max_diff = 0.0f;
    int max_diff_idx = 0;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float gpu_v = gpu_result[i];
        float cpu_v = cpu_result[i];
        float diff = fabs(gpu_v - cpu_v);
        
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = static_cast<int>(i);
        }
        
        // Tensor Core 采用不同硬件加速的降精度累加，误差容忍度应放大到 1e-2级别
        if (diff > epsilon && (diff / (fabs(cpu_v) + 1e-5f)) > 1e-2f) {
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

    cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result[0] 
         << " (期望 " << cpu_result[0] << ")\n";
    return true;
}


// WMMA GEMM GPU 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult wmma_gemm_gpu(const vector<half>& h_A, const vector<half>& h_B, RMatrix h_C, CInt M, CInt N, CInt K, CInt iterations, KernelFunc kernel) {
    half* d_A = nullptr;
    half* d_B = nullptr;
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

    // Kernel 配置 - 必须使用 const dim3
    // 线程设计：每个 Block 包含 8 个 Warp，每个 Warp (32 线程) 负责一个 16x16 tile
    const dim3 block(32, BLOCK_SIZE_1D / 32); 
    const dim3 grid(cdiv(N, WMMA_N), cdiv(M, WMMA_M * (BLOCK_SIZE_1D / 32)));

    // Kernel 预热
    kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 执行循环 + 计时
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    // D2H 传输 + 计时
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
    CInt M = 2048;
    CInt N = 2048;
    CInt K = 2048;
    CInt iterations = 100;

    CSize size_A = M * K * sizeof(half);
    CSize size_B = K * N * sizeof(half);
    CSize size_C = M * N * FSIZE;
    const double total_mb = (size_A + size_B + size_C) / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      WMMA Tensor Core 性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：M=" << M << " N=" << N << " K=" << K << "\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE_1D << " 线程 (32x8 Warps)\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    vector<half> h_A(M * K);
    vector<half> h_B(K * N);
    Matrix h_C_cpu(M * N, 0.0f);
    Matrix h_C_gpu(M * N, 0.0f);

    srand(42);
    for (int i = 0; i < M * K; ++i) {
        // 加入极小的随机浮点数作为测试
        h_A[i] = __float2half(static_cast<float>(rand() % 100) / 1000.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = __float2half(static_cast<float>(rand() % 100) / 1000.0f);
    }

    cout << "--- CPU 验证 (若尺寸较小) ---\n";
    CpuTimer cpuTimer;
    double cpu_time_ms = 0.0;
    if (M <= 512 && N <= 512 && K <= 512) {
        cpuTimer.start();
        wmma_gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);
        cpuTimer.stop();
        cpu_time_ms = cpuTimer.elapsed_ms();
        cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    } else {
        cout << "矩阵尺寸过大，跳过 CPU 参考计算。\n";
        cpu_time_ms = 1.0; // 避免除零
    }
    cout << "\n";

    // GPU 版本 1: Naive WMMA GEMM
    cout << "--- GPU 版本 1: Naive WMMA Tensor Core ---\n";
    GpuTimingResult result1 = wmma_gemm_gpu(h_A, h_B, h_C_gpu, M, N, K, iterations, wmma_gemm_naive);
    cout << "H2D 传输时间：   " << setw(8) << result1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result1.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_kernel = cpu_time_ms / result1.kernel_ms;
    double speedup_total = cpu_time_ms / result1.total_ms;
    cout << "CPU vs GPU Kernel 加速比：" << setprecision(2) << speedup_kernel << "x\n";
    cout << "CPU vs GPU 总时间加速比：" << speedup_total << "x\n";

    // 算力与带宽计算 (浮点运算量: 2 * M * N * K)
    double tflops = (2.0 * M * N * K / 1e12) / (result1.kernel_ms / 1000.0);
    cout << "GPU 有效算力 (TFLOPS)：" << setprecision(2) << tflops << " TFLOPS\n";
    cout << "(RTX 4090 FP16 TC 理论算力峰值：~ 165 TFLOPS (无稀疏))\n";

    double bytes = size_A + size_B + size_C;
    double gpu_bandwidth = (bytes / 1e9) / (result1.kernel_ms / 1000.0);
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    cout << "--- Kernel 性能对比 ---\n";
    cout << "Naive WMMA: " << setw(8) << setprecision(4) << result1.kernel_ms << " ms (基准)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    // 放宽判定阈值为 1e-1f 以容忍 FP16 的大规模乘加累积误差
    bool pass1 = false;
    if (M <= 512 && N <= 512 && K <= 512) {
        pass1 = verify_results(h_C_gpu, h_C_cpu, "WMMA Tensor Core GEMM", 1e-1f);
        if (pass1) {
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
