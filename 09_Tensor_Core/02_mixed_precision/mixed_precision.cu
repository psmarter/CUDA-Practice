// CUDA Tensor Core Mixed Precision (FP16 Input + FP32 Accumulate) 基准测试
#include <code_abbreviation.h>

#include <mma.h>

using namespace nvcuda;

// 传统 FP32 GEMM 计算（GPU kernel，手写）
__global__ void gemm_fp32_kernel(CPFloat A, CPFloat B, PFloat C, CInt M, CInt N, CInt K) {
    CInt row = blockIdx.y * blockDim.y + threadIdx.y;
    CInt col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 混合精度 GEMM (Tensor Core WMMA)（GPU kernel，手写）
__global__ void wmma_mixed_gemm_kernel(const half* A, const half* B, float* C, CInt M, CInt N, CInt K) {
    // 映射当前 Warp 到 C 矩阵的 16x16 瓦片(Tile)
    CInt warpM = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_M;
    CInt warpN = (blockIdx.x * blockDim.x / 32) * WMMA_N; // blockDim.x 固定维 32（1个Warp的宽）

    if (warpM >= M || warpN >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N, wmma::mem_row_major);
}

// FP32 参考实现（CPU，手写）
void gemm_cpu(CRMatrix A, CRMatrix B, RMatrix C, CInt M, CInt N, CInt K) {
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

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CInt n, CFloat epsilon = EPSILON) {
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配\n";
        return false;
    }

    int error_count = 0;
    float max_diff = 0.0f;
    int max_diff_idx = 0;

    for (int i = 0; i < n; ++i) {
        float gpu_v = gpu_result[i];
        float cpu_v = cpu_result[i];
        float diff = fabs(gpu_v - cpu_v);
        
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        
        // Tensor Core 处理可能会带来精度的下降，因此放宽差异容忍比率
        if (diff > epsilon && (diff / (fabs(cpu_v) + 1e-5f)) > 5e-2f) {
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


// 传统 FP32 GEMM 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult gemm_fp32_gpu(CRMatrix h_A, CRMatrix h_B, RMatrix h_C, 
                              CInt M, CInt N, CInt K, CInt iterations, KernelFunc kernel) {
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

    const dim3 block(16, 16);
    const dim3 grid(cdiv(N, 16), cdiv(M, 16));

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

// FP16 到 FP32 WMMA GPU 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult wmma_mixed_gemm_gpu(CRMatrix h_A, CRMatrix h_B, RMatrix h_C, 
                                    CInt M, CInt N, CInt K, CInt iterations, KernelFunc kernel) {
    PFloat d_A_fp32 = nullptr, d_B_fp32 = nullptr;
    half *d_A_fp16 = nullptr, *d_B_fp16 = nullptr;
    PFloat d_C = nullptr;
    
    CSize size_A_f32 = M * K * FSIZE;
    CSize size_B_f32 = K * N * FSIZE;
    CSize size_C_f32 = M * N * FSIZE;
    CSize size_A_f16 = M * K * sizeof(half);
    CSize size_B_f16 = K * N * sizeof(half);

    CUDA_CHECK(cudaMalloc((void**)&d_A_fp32, size_A_f32));
    CUDA_CHECK(cudaMalloc((void**)&d_B_fp32, size_B_f32));
    CUDA_CHECK(cudaMalloc((void**)&d_A_fp16, size_A_f16));
    CUDA_CHECK(cudaMalloc((void**)&d_B_fp16, size_B_f16));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C_f32));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A_fp32, h_A.data(), size_A_f32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp32, h_B.data(), size_B_f32, cudaMemcpyHostToDevice));
    
    const dim3 cvt_block(256);
    float2half_kernel<<<cdiv(M * K, 256), cvt_block>>>(d_A_fp32, d_A_fp16, M * K);
    float2half_kernel<<<cdiv(K * N, 256), cvt_block>>>(d_B_fp32, d_B_fp16, K * N);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms(); // 转换时间归属 H2D 开销范围

    const dim3 block(32, WARPS_PER_BLOCK_Y);
    const dim3 grid(cdiv(N, WMMA_N), cdiv(M, WMMA_M * WARPS_PER_BLOCK_Y));

    // 预热
    kernel<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C, M, N, K);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C_f32, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_A_fp32));
    CUDA_CHECK(cudaFree(d_B_fp32));
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    // 采用 1024 适中的数据量以防止 CPU 计算耗时过长，同时仍能看清性能差异
    CInt M = 1024;
    CInt N = 1024;
    CInt K = 1024;
    CInt iterations = 100;

    CSize size_A = M * K * FSIZE;
    CSize size_B = K * N * FSIZE;
    CSize size_C = M * N * FSIZE;
    const double total_mb = (size_A + size_B + size_C) / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      WMMA 混合精度性能基准测试\n";
    cout << "========================================\n";
    cout << "矩阵尺寸：" << M << " x " << N << " x " << K << "\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "WMMA 切块：" << WMMA_M << "x" << WMMA_N << "x" << WMMA_K << "\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_A(M * K), h_B(K * N);
    Matrix h_C_fp32(M * N), h_C_wmma(M * N), h_C_cpu(M * N);

    srand(42);
    // 限定在小随机实数以防累加时严重的向下溢出和精度丢失
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand() % 100) / 100.0f;

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1：传统 FP32
    cout << "--- GPU 版本 1: 传统 FP32 GEMM ---\n";
    GpuTimingResult result1 = gemm_fp32_gpu(h_A, h_B, h_C_fp32, M, N, K, iterations, gemm_fp32_kernel);
    cout << "H2D 传输时间：   " << setw(8) << result1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result1.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2：WMMA 混合精度 Tensor Core
    cout << "--- GPU 版本 2: WMMA 混合精度 (FP16乘加FP32) ---\n";
    GpuTimingResult result2 = wmma_mixed_gemm_gpu(h_A, h_B, h_C_wmma, M, N, K, iterations, wmma_mixed_gemm_kernel);
    cout << "H2D(含转换)时间：" << setw(8) << result2.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result2.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result2.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result2.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_kernel = cpu_time_ms / result2.kernel_ms;
    double speedup_total = cpu_time_ms / result2.total_ms;
    cout << "CPU vs WMMA Kernel 加速比：" << setprecision(2) << speedup_kernel << "x\n";
    cout << "CPU vs WMMA 总时间加速比： " << setprecision(2) << speedup_total << "x\n";

    double operations = 2.0 * static_cast<double>(M) * N * K;
    double tflops_fp32 = operations / (result1.kernel_ms / 1000.0) / 1e12;
    double tflops_wmma = operations / (result2.kernel_ms / 1000.0) / 1e12;
    
    double bytes_fp32 = (M * K * 4.0 + K * N * 4.0 + M * N * 4.0);
    double bw_fp32 = (bytes_fp32 / 1e9) / (result1.kernel_ms / 1000.0);
    double bytes_wmma_inner = (M * K * 2.0 + K * N * 2.0 + M * N * 4.0);
    double bw_wmma = (bytes_wmma_inner / 1e9) / (result2.kernel_ms / 1000.0);

    cout << "FP32 有效算力：" << setprecision(2) << tflops_fp32 << " TFLOPS\n";
    cout << "WMMA 有效算力：" << setprecision(2) << tflops_wmma << " TFLOPS\n";
    cout << "FP32 有效访存带宽：" << setprecision(2) << bw_fp32 << " GB/s\n";
    cout << "WMMA 有效访存带宽：" << setprecision(2) << bw_wmma << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "(附：RTX 4090 Tensor Core 算力理论峰值 ~330 TFLOPS)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Naive FP32 GEMM:       " << setw(8) << setprecision(4) << result1.kernel_ms << " ms (基准)\n";
    cout << "WMMA Mixed Precision:  " << setw(8) << result2.kernel_ms << " ms ("
         << setprecision(2) << result1.kernel_ms / result2.kernel_ms << "x)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_C_fp32, h_C_cpu, "传统 FP32", M * N);
    bool pass2 = verify_results(h_C_wmma, h_C_cpu, "Tensor Core WMMA (容忍更大精度误差)", M * N, 0.05f);

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
