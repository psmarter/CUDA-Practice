#include <code_abbreviation.h>

// 单独使用以下两种方式收益可能并不大，但结合使用之前的分块技术可显著提高性能
// GEMM-向量化（GPU kernel，手写）
// 这里使用一个线程连续读取四个数据，会产生发散
__global__ void vectorized_gemm(CPFloat A, CPFloat B, PFloat C, CInt M, CInt N, CInt K) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    CInt row = blockIdx.y * blockDim.y + threadIdx.y;
    CInt col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    for (int i = 0; i < cdiv(N, TILE_SIZE); ++i) {
        CInt tiled_col = i * TILE_SIZE + threadIdx.x;
        CInt tiled_row = i * TILE_SIZE + threadIdx.y;

        if (threadIdx.x % 4 == 0) {
            float4 vecA = make_float4(0, 0, 0, 0);
            if (tiled_col + 3 < N && row < M)  {
                vecA = *reinterpret_cast<const float4*>(&A[row * N + tiled_col]);
            } else if (row < M) {
                float temp[4] = {0, 0, 0, 0};
                for (int j = 0; j < 4 && tiled_col + j < N; ++j) {
                    temp[j] = A[row * N + tiled_col + j];
                }
                vecA = make_float4(temp[0], temp[1], temp[2], temp[3]);
                // 这句可能是错误的，因为并不能保证栈上对齐，不一定是连续的16B
                // vecA = *reinterpret_cast<const float4*>(&temp);
            }
            shared_A[threadIdx.y][threadIdx.x] = vecA.x;
            shared_A[threadIdx.y][threadIdx.x + 1] = vecA.y;
            shared_A[threadIdx.y][threadIdx.x + 2] = vecA.z;
            shared_A[threadIdx.y][threadIdx.x + 3] = vecA.w;
        }

        if (threadIdx.x % 4 == 0) {
            float4 vecB =  make_float4(0, 0, 0, 0);
            if (tiled_row < N && col + 3 < K) {
                vecB = *reinterpret_cast<const float4*>(&B[tiled_row * K + col]);
            } else if (tiled_row < N) {
                float temp[4] = {0, 0, 0, 0};
                for (int j = 0; j < 4 && col + j < K; ++j)  {
                    temp[j] = B[tiled_row * K + col + j];
                }
                vecB = make_float4(temp[0], temp[1], temp[2], temp[3]);
            }
            shared_B[threadIdx.y][threadIdx.x] = vecB.x;
            shared_B[threadIdx.y][threadIdx.x + 1] = vecB.y;
            shared_B[threadIdx.y][threadIdx.x + 2] = vecB.z;
            shared_B[threadIdx.y][threadIdx.x + 3] = vecB.w;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            value += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M &&  col < K) {
        C[row * K + col] = value;
    }
}

// GEMM-双缓冲（GPU kernel，手写）
// 非严格意义上的异步双缓冲，优化可考虑使用cp.async
__global__ void double_buffer_gemm(CPFloat A, CPFloat B, PFloat C, CInt M, CInt N, CInt K) {
    __shared__ float shared_A[2][TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[2][TILE_SIZE][TILE_SIZE];

    CInt row = blockIdx.y * blockDim.y + threadIdx.y;
    CInt col = blockIdx.x * blockDim.x + threadIdx.x;
    CInt num_tiles = cdiv(N, TILE_SIZE);

    int buffer_index = 0;
    float value = 0.0f;

    // 预加载第0块
    CInt tiled_col = 0 * TILE_SIZE + threadIdx.x;
    CInt tiled_row = 0 * TILE_SIZE + threadIdx.y;
    shared_A[buffer_index][threadIdx.y][threadIdx.x] = (row < M && tiled_col < N) ? A[row * N + tiled_col] : 0.0f;
    shared_B[buffer_index][threadIdx.y][threadIdx.x] = (tiled_row < N && col < K) ? B[tiled_row * K + col] : 0.0f;
    __syncthreads();

    for (int i = 0; i < num_tiles; ++i) {
        buffer_index = 1 - buffer_index;

        if (i + 1 < num_tiles) {
            CInt next_tiled_col = (i + 1) * TILE_SIZE + threadIdx.x;
            CInt next_tiled_row = (i + 1) * TILE_SIZE + threadIdx.y;
            shared_A[buffer_index][threadIdx.y][threadIdx.x] = (row < M && next_tiled_col < N) ? A[row * N + next_tiled_col] : 0.0f;
            shared_B[buffer_index][threadIdx.y][threadIdx.x] = (next_tiled_row < N && col < K) ? B[next_tiled_row * K + col] : 0.0f;
        }
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            value = fmaf(shared_A[1 - buffer_index][threadIdx.y][k], shared_B[1 - buffer_index][k][threadIdx.x], value);
        }
        __syncthreads();
    }

    if (row < M &&  col < K) {
        C[row * K + col] = value;
    }
}

// GEMM（CPU，手写）
void gemm_cpu(CPFloat A, CPFloat B, PFloat C, CInt M, CInt N, CInt K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = 1e-3f) {
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
        float rel_error = diff / (fabs(cpu_result[i]) + 1e-8f);
        if (rel_error > epsilon && diff > epsilon) {
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

    cout << "✓ " << kernel_name << " PASSED: 结果验证通过 (最大误差 " << max_diff << ")\n";
    return true;
}


// GEMM GPU 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult gemm_gpu(CRMatrix h_A, CRMatrix h_B, RMatrix h_C,
                         CInt M, CInt N, CInt K, CInt iterations,
                         KernelFunc kernel, dim3 grid, dim3 block) {
    PFloat d_A = nullptr;
    PFloat d_B = nullptr;
    PFloat d_C = nullptr;

    CSize size_A = M * N * FSIZE;
    CSize size_B = N * K * FSIZE;
    CSize size_C = M * K * FSIZE;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    // 计时器
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    // H2D
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    // 预热
    kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 计时
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    // D2H
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    // 总时间
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    // 释放
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    // 矩阵维度: A(M x N) * B(N x K) = C(M x K)
    CInt M = 1024;
    CInt N = 1024;
    CInt K = 1024;
    CInt iterations = 10;

    CSize size_A = M * N * FSIZE;
    CSize size_B = N * K * FSIZE;
    CSize size_C = M * K * FSIZE;
    const double total_mb = (size_A + size_B + size_C) / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "    Advanced GEMM 性能基准测试\n";
    cout << "========================================\n";
    cout << "矩阵维度：A(" << M << " x " << N << ") * B(" << N << " x " << K << ") = C(" << M << " x " << K << ")\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB (A + B + C)\n";
    cout << "Tile 大小：" << TILE_SIZE << " x " << TILE_SIZE << "\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_A(M * N);
    Matrix h_B(N * K);
    Matrix h_C1(M * K, 0.0f);
    Matrix h_C2(M * K, 0.0f);
    Matrix h_C_cpu(M * K, 0.0f);

    srand(42);
    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
    for (int i = 0; i < N * K; ++i) h_B[i] = static_cast<float>(rand() % 100) / 100.0f;

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1：Vectorized GEMM
    cout << "--- GPU 版本 1: Vectorized GEMM ---\n";
    const dim3 block1(TILE_SIZE, TILE_SIZE);
    const dim3 grid1(cdiv(K, TILE_SIZE), cdiv(M, TILE_SIZE));
    GpuTimingResult result1 = gemm_gpu(h_A, h_B, h_C1, M, N, K, iterations, vectorized_gemm, grid1, block1);
    cout << "H2D 传输时间：   " << setw(8) << result1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result1.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2：Double Buffer GEMM
    cout << "--- GPU 版本 2: Double Buffer GEMM ---\n";
    const dim3 block2(TILE_SIZE, TILE_SIZE);
    const dim3 grid2(cdiv(K, TILE_SIZE), cdiv(M, TILE_SIZE));
    GpuTimingResult result2 = gemm_gpu(h_A, h_B, h_C2, M, N, K, iterations, double_buffer_gemm, grid2, block2);
    cout << "H2D 传输时间：   " << setw(8) << result2.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result2.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result2.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result2.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_kernel = cpu_time_ms / result2.kernel_ms;
    double speedup_total = cpu_time_ms / result2.total_ms;
    cout << "CPU vs GPU Kernel 加速比：" << setprecision(2) << speedup_kernel << "x\n";
    cout << "CPU vs GPU 总时间加速比：" << speedup_total << "x\n";

    // GFLOPS 计算
    double flops = 2.0 * M * N * K;
    double gpu_gflops = (flops / 1e9) / (result2.kernel_ms / 1000.0);
    double cpu_gflops = (flops / 1e9) / (cpu_time_ms / 1000.0);
    cout << "GPU 计算性能：" << setprecision(2) << gpu_gflops << " GFLOPS\n";
    cout << "CPU 计算性能：" << setprecision(2) << cpu_gflops << " GFLOPS\n";
    cout << "(RTX 4090 理论峰值：~82.6 TFLOPS FP32)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Vectorized GEMM:    " << setw(8) << setprecision(4) << result1.kernel_ms << " ms (基准)\n";
    cout << "Double Buffer GEMM: " << setw(8) << result2.kernel_ms << " ms ("
         << setprecision(2) << result1.kernel_ms / result2.kernel_ms << "x)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_C1, h_C_cpu, "Vectorized GEMM");
    bool pass2 = verify_results(h_C2, h_C_cpu, "Double Buffer GEMM");

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}

