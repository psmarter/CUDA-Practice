#include <code_abbreviation.h>

using matrix = Matrix;            // 兼容本文件中的 'matrix' 别名
using Int = CInt;                 // 兼容本文件中的 'Int' 别名
constexpr size_t sizeF = FSIZE;   // 兼容本文件中的 'sizeF' 别名

// 矩阵乘法（GPU kernel，手写）
__global__ void matrix_mul_naive(const float* A, const float* B, float* C, Int m, Int n, Int k){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < m && col < k) {
        float value = 0.0f;
        for (int i = 0; i < n; ++i) {
            value += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = value;
    }
}

// 矩阵乘法（CPU，手写）
void matrix_mul_cpu(const float* A, const float* B, float* C, Int m, Int n, Int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            float value = 0.0f;
            for (int i = 0; i < n; ++i) {
                value += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = value;
        }
    }
}


// 矩阵乘法（GPU，手写）
GpuTimingResult matrix_mul_naive_device(const matrix& h_a, const matrix& h_b, matrix& h_c, Int m, Int n, Int k, Int iterations = 100) {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    const size_t size_a = m * n * sizeF;
    const size_t size_b = n * k * sizeF;
    const size_t size_c = m * k * sizeF;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_a, size_a));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size_b));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size_c));

    // 创建计时器
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    // Host to Device 传输
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_b, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(16, 16);
    const dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    // Kernel 预热
    matrix_mul_naive<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 执行（多次取平均）
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        matrix_mul_naive<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    // Device to Host 传输
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_c, cudaMemcpyDeviceToHost));
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
bool verify_results(const matrix& h_a, const matrix& h_b, const matrix& h_c, Int m, Int n, Int k, int& error_count) {
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
    Int m = 1024;  // A 的行数，C 的行数
    Int n = 1024;  // A 的列数，B 的行数
    Int k = 1024;  // B 的列数，C 的列数
    Int iterations = 10;  // 矩阵乘法计算量大，减少迭代次数

    const size_t size_a = m * n * sizeF;
    const size_t size_b = n * k * sizeF;
    const size_t size_c = m * k * sizeF;
    const double total_mb = (size_a + size_b + size_c) / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "    Matrix Mul Naive 性能基准测试\n";
    cout << "========================================\n";
    cout << "矩阵维度：A(" << m << " x " << n << ") * B(" << n << " x " << k << ") = C(" << m << " x " << k << ")\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB (A + B + C)\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存
    matrix h_a(m * n);
    matrix h_b(n * k);
    matrix h_c_gpu(m * k, 0.0f);
    matrix h_c_cpu(m * k, 0.0f);

    // 初始化矩阵（使用随机值更真实，但为了验证用固定值）
    for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) h_a[i] = static_cast<float>(i % 100) / 100.0f;
    for (size_t i = 0; i < static_cast<size_t>(n * k); ++i) h_b[i] = static_cast<float>(i % 100) / 100.0f;

    // GPU 执行
    cout << "--- GPU 详细计时 ---\n";
    GpuTimingResult gpu_result = matrix_mul_naive_device(h_a, h_b, h_c_gpu, m, n, k, iterations);

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