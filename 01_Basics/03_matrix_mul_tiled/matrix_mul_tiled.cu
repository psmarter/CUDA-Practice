#include <code_abbreviation.h>

constexpr int TILE_WIDTH = 32;

// 矩阵乘法-使用共享内存分块优化（GPU kernel，手写）
__global__ void matrix_mul_tiled(CPFloat a, CPFloat b, PFloat c, CInt m, CInt n, CInt k) {
    // 声明共享内存
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = TILE_WIDTH * by + ty;
    int col = TILE_WIDTH * bx + tx;

    float value = 0.0f;
    int num_tiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int tile = 0; tile < num_tiles; ++tile) {
        // 将数据加载到共享内存
        int mCol = tile * TILE_WIDTH + tx;
        if (row < m && mCol < n) {
            tile_a[ty][tx] = a[row * n + mCol];
        } else {
            tile_a[ty][tx] = 0.0f;
        }

        int nRow = tile * TILE_WIDTH + ty;
        if (nRow < n && col < k) {
            tile_b[ty][tx] = b[nRow * k + col];
        } else {
            tile_b[ty][tx] = 0.0f;
        }

        // 同步，确保数据加载完成
        __syncthreads();

        // 计算
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += tile_a[ty][i] * tile_b[i][tx];
        }

        // 同步，确保计算完成
        __syncthreads();
    }

    if (row < m && col < k) {
        c[row * k + col] = value;
    }
}

// 矩阵乘法（CPU，手写）
void matrix_mul_cpu(CPFloat a, CPFloat b, PFloat c, CInt m, CInt n, CInt k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            float value = 0.0f;
            for (int i = 0; i < n; ++i) {
                value += a[row * n + i] * b[i * k + col];
            }
            c[row * k + col] = value;
        }
    }
}


// 矩阵乘法（GPU，手写）
GpuTimingResult matrix_mul_tiled_device(CRMatrix a, CRMatrix b, RMatrix c, CInt m, CInt n, CInt k, CInt iterations = 100) {
    PFloat d_a = nullptr;
    PFloat d_b = nullptr;
    PFloat d_c = nullptr;
    CSize size_a = m * n * FSIZE;
    CSize size_b = n * k * FSIZE;
    CSize size_c = m * k * FSIZE;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_a, size_a));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size_b));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size_c));

    // 创建计时器
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    // H to D
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), size_b, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid((k + TILE_WIDTH -1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    // Kernel 预热
    matrix_mul_tiled<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 执行
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        matrix_mul_tiled<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    // D to H
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, size_c, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    // 计算总时间
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return result;
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix h_a, CRMatrix h_b, CRMatrix h_c, CInt m, CInt n, CInt k, int& error_count) {
    error_count = 0;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            float expected = 0.0f;
            for (int i = 0; i < n; ++i) {
                expected += h_a[row * n + i] * h_b[i * k + col];
            }
            float gpu_value = h_c[row * k + col];
            if (fabs(expected - gpu_value) > 1e-3) {        // 放宽阈值（乘法累计误差）
                error_count++;
                // 打印前 5 个错误
                if (error_count <= 5) { 
                    cout << "  错误 #" << error_count << ": 行 " << row
                         << ", 列 " << col << ", 期望值 " << expected
                         << ", 实际值 " << gpu_value << endl;
                }
            }
        }
    }
    return error_count == 0;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    // 矩阵维度: A(m x n) * B(n x k) = C(m x k)
    CInt m = 1024;  // A 的行数，C 的行数
    CInt n = 1024;  // A 的列数，B 的行数
    CInt k = 1024;  // B 的列数，C 的列数
    CInt iterations = 10;  // 矩阵乘法计算量大，减少迭代次数

    CSize size_a = m * n * FSIZE;
    CSize size_b = n * k * FSIZE;
    CSize size_c = m * k * FSIZE;
    const double total_mb = (size_a + size_b + size_c) / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "    Matrix Mul Tiled 性能基准测试\n";
    cout << "========================================\n";
    cout << "矩阵维度：A(" << m << " x " << n << ") * B(" << n << " x " << k << ") = C(" << m << " x " << k << ")\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB (A + B + C)\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存
    Matrix h_a(m * n);
    Matrix h_b(n * k);
    Matrix h_c_gpu(m * k, 0.0f);
    Matrix h_c_cpu(m * k, 0.0f);

    // 初始化矩阵
    for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) h_a[i] = static_cast<float>(i % 100) / 100.0f;
    for (size_t i = 0; i < static_cast<size_t>(n * k); ++i) h_b[i] = static_cast<float>(i % 100) / 100.0f;
    
    // GPU 执行
    cout << "--- GPU 详细计时 ---\n";
    GpuTimingResult gpu_result = matrix_mul_tiled_device(h_a, h_b, h_c_gpu, m, n, k, iterations);

    cout << "H2D 传输时间：   " << setw(8) << gpu_result.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << gpu_result.kernel_ms
         << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << gpu_result.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << gpu_result.total_ms << " ms\n";
    cout << "\n";

    // CPU 执行
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    matrix_mul_cpu(h_a.data(), h_b.data(), h_c_cpu.data(), m, n, k);
    cpuTimer.stop();

    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";

    // 加速比
    double speedup_kernel = cpu_time_ms / gpu_result.kernel_ms;
    cout << "CPU vs GPU Kernel 加速比：" << fixed << setprecision(2)
         << speedup_kernel << "x\n";

    double speedup_total = cpu_time_ms / gpu_result.total_ms;
    cout << "CPU vs GPU 总时间加速比：" << speedup_total << "x\n";

    // GFLOPS 计算：矩阵乘法 C = A * B，每个 C[i][j] 需要 n 次乘法 + (n-1) 次加法 ≈ 2n 次浮点运算
    // 总浮点运算数 = m * k * 2n = 2 * m * n * k
    double flops = 2.0 * m * n * k;
    double gpu_gflops = (flops / 1e9) / (gpu_result.kernel_ms / 1000.0);
    double cpu_gflops = (flops / 1e9) / (cpu_time_ms / 1000.0);

    cout << "GPU 计算性能：" << setprecision(2) << gpu_gflops << " GFLOPS\n";
    cout << "CPU 计算性能：" << setprecision(2) << cpu_gflops << " GFLOPS\n";
    cout << "(RTX 4090 理论峰值：~82.6 TFLOPS FP32)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";

    // 验证 GPU 结果
    int gpu_errors = 0;
    if (verify_results(h_a, h_b, h_c_gpu, m, n, k, gpu_errors)) {
        cout << "✓ GPU PASSED: 全部 " << m * k << " 个元素验证正确\n";
    } else {
        cout << "✗ GPU FAILED: 共有 " << gpu_errors << " 个错误\n";
    }

    // 验证 CPU 结果（与 GPU 比较）
    int diff_count = 0;
    for (int i = 0; i < m * k; ++i) {
        if (fabs(h_c_gpu[i] - h_c_cpu[i]) > 1e-3) {  // 矩阵乘法误差累积，放宽阈值
            diff_count++;
        }
    }
    if (diff_count == 0) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在 " << diff_count << " 处差异\n";
    }

    cout << "\n========================================\n";
    
    return 0;
}