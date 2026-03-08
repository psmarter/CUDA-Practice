#include <code_abbreviation.h>


// 向量加法（GPU kernel，手写）
__global__ void vector_add(const float* A, const float* B, float* C, const int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// 向量加法（CPU，手写）
void vector_add_cpu(const vector<float>& h_a, const vector<float>& h_b,
                    vector<float>& h_c, const int n) {
    for (int i = 0; i < n; ++i) {
        h_c[i] = h_a[i] + h_b[i];
    }
}


// 向量加法（GPU，手写）
GpuTimingResult vector_add_device(const vector<float>& h_a, const vector<float>& h_b,
                                   vector<float>& h_c, const int n,
                                   const int iterations = 100) {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    const size_t size = n * sizeof(float);

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // 创建计时器
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    // Host to Device 传输
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    // Kernel 预热
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 执行（多次取平均）
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    // Device to Host 传输
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost));
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
bool verify_results(const vector<float>& h_a, const vector<float>& h_b,
                    const vector<float>& h_c, const int n, int& error_count) {
    error_count = 0;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            error_count++;
            // 打印前 5 个错误
            if (error_count <= 5) {
                cout << "  错误 #" << error_count << ": 索引 " << i
                     << ", 期望值 " << expected << ", 实际值 " << h_c[i] << endl;
            }
        }
    }
    return error_count == 0;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    const int n = 1 << 26;  // 64M 元素
    const int iterations = 100;
    const size_t size_bytes = n * sizeof(float);
    const double size_mb = size_bytes / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      Vector Add 性能基准测试\n";
    cout << "========================================\n";
    cout << "问题规模：" << n << " (" << (n >> 20) << " M) 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << size_mb << " MB (每个数组)\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存
    vector<float> h_a(n, 1.0f);
    vector<float> h_b(n, 2.0f);
    vector<float> h_c_gpu(n, 0.0f);
    vector<float> h_c_cpu(n, 0.0f);

    // GPU 执行
    cout << "--- GPU 详细计时 ---\n";
    GpuTimingResult gpu_result = vector_add_device(h_a, h_b, h_c_gpu, n, iterations);

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
    vector_add_cpu(h_a, h_b, h_c_cpu, n);
    cpuTimer.stop();

    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";

    // 加速比（CPU vs GPU Kernel）
    double speedup_kernel = cpu_time_ms / gpu_result.kernel_ms;
    cout << "CPU vs GPU Kernel 加速比：" << fixed << setprecision(2)
         << speedup_kernel << "x\n";

    // 加速比（CPU vs GPU 总时间，含传输）
    double speedup_total = cpu_time_ms / gpu_result.total_ms;
    cout << "CPU vs GPU 总时间加速比：" << speedup_total << "x\n";

    // 有效带宽计算：读 2 个数组 + 写 1 个数组 = 3 * n * sizeof(float)
    double total_bytes = 3.0 * n * sizeof(float);
    double kernel_time_s = gpu_result.kernel_ms / 1000.0;
    double bandwidth_GBs = (total_bytes / 1e9) / kernel_time_s;
    cout << "有效带宽：" << setprecision(2) << bandwidth_GBs << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";

    // 验证 GPU 结果
    int gpu_errors = 0;
    if (verify_results(h_a, h_b, h_c_gpu, n, gpu_errors)) {
        cout << "✓ GPU PASSED: 全部 " << n << " 个元素验证正确\n";
    } else {
        cout << "✗ GPU FAILED: 共有 " << gpu_errors << " 个错误\n";
    }

    // 验证 CPU 结果
    int cpu_errors = 0;
    if (verify_results(h_a, h_b, h_c_cpu, n, cpu_errors)) {
        cout << "✓ CPU PASSED: 全部 " << n << " 个元素验证正确\n";
    } else {
        cout << "✗ CPU FAILED: 共有 " << cpu_errors << " 个错误\n";
    }

    cout << "\n========================================\n";

    return 0;
}