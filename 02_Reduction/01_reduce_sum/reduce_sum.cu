#include <code_abbreviation.h>


// 归约（GPU kernel，手写）
__global__ void simple_reduce_sum(PFloat input, PFloat output) {
    CInt i = 2 * threadIdx.x;
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

// 归约-通过收敛减少发散（GPU kernel，手写）
__global__ void convergent_reduce_sum(PFloat input, PFloat output) {
    CInt i = threadIdx.x;
    for (int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

// 归约-使用共享内存（GPU kernel，手写）
__global__ void shared_reduce_sum(PFloat input, PFloat output) {
    __shared__ float shared_data[BLOCK_SIZE];
    CInt i = threadIdx.x;
    shared_data[i] = input[i] + input[i + BLOCK_SIZE];
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            shared_data[i] += shared_data[i + stride];
        }
    }
    if (threadIdx.x == 0) {
        *output = shared_data[0];
    }
}

// 归约-求和（CPU，手写）
float reduce_sum_cpu(CPFloat data, CInt length) {
    double total = 0.0;
    for (int i = 0; i < length; i++) {
        total += data[i];
    }
    return static_cast<float>(total);
}


// 验证结果（AI 生成）
bool verify_results(float gpu_result, float cpu_result, const string& kernel_name) {
    if (fabs(gpu_result - cpu_result) < 1e-3) {
        cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result 
             << " (期望 " << cpu_result << ")\n";
        return true;
    } else {
        cout << "✗ " << kernel_name << " FAILED: 结果 " << gpu_result 
             << " (期望 " << cpu_result << ")\n";
        return false;
    }
}

// 通用归约 GPU 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult reduce_sum_gpu(CRMatrix h_input, float& h_output, CInt iterations, KernelFunc kernel) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    CSize size_input = h_input.size() * FSIZE;
    CSize size_output = FSIZE;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_output));

    // 创建计时器
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    // H to D
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(BLOCK_SIZE);
    const dim3 grid(1);

    // Kernel 预热
    kernel<<<grid, block>>>(d_input, d_output);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 执行
    // 注意：归约会破坏输入数据，所以需要额外处理
    float total_kernel_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        // 重新拷贝数据（不计入 kernel 时间）
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
        
        // 只计时 kernel 执行
        timerKernel.start();
        kernel<<<grid, block>>>(d_input, d_output);
        timerKernel.stop();
        total_kernel_ms += timerKernel.elapsed_ms();
    }
    CUDA_CHECK_LAST();
    result.kernel_ms = total_kernel_ms / static_cast<float>(iterations);

    // D to H
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, FSIZE, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    // 计算总时间
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt n = 2 * BLOCK_SIZE;  // 2048 元素（数据量较小，性能可能不如CPU）
    CInt iterations = 100;

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      Reduce Sum 性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(4) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_input(n);
    for (int i = 0; i < n; ++i) {
        h_input[i] = 1.0f;  // 全1，便于验证：结果应为 n
    }

    // CPU 执行
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    float cpu_result = reduce_sum_cpu(h_input.data(), n);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 计算 - 3 个版本对比
    float gpu_result = 0.0f;

    // 版本 1：Simple Reduce
    cout << "--- GPU 版本 1: Simple Reduce ---\n";
    GpuTimingResult result1 = reduce_sum_gpu(h_input, gpu_result, iterations, simple_reduce_sum);
    cout << "H2D 传输时间：   " << setw(8) << result1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result1.total_ms << " ms\n";
    cout << "\n";

    // 版本 2：Convergent Reduce
    cout << "--- GPU 版本 2: Convergent Reduce ---\n";
    GpuTimingResult result2 = reduce_sum_gpu(h_input, gpu_result, iterations, convergent_reduce_sum);
    cout << "H2D 传输时间：   " << setw(8) << result2.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result2.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result2.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result2.total_ms << " ms\n";
    cout << "\n";

    // 版本 3：Shared Memory Reduce
    cout << "--- GPU 版本 3: Shared Memory Reduce ---\n";
    GpuTimingResult result3 = reduce_sum_gpu(h_input, gpu_result, iterations, shared_reduce_sum);
    cout << "H2D 传输时间：   " << setw(8) << result3.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result3.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result3.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result3.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";

    // 加速比（使用最快的 Shared Memory 版本）
    double speedup_kernel = cpu_time_ms / result3.kernel_ms;
    double speedup_total = cpu_time_ms / result3.total_ms;
    cout << "CPU vs GPU Kernel 加速比：" << setprecision(2) << speedup_kernel << "x\n";
    cout << "CPU vs GPU 总时间加速比：" << speedup_total << "x\n";

    // 带宽计算：归约读取 n 个元素 + 写入 1 个元素 ≈ n * sizeof(float)
    double bytes = n * FSIZE;
    double gpu_bandwidth = (bytes / 1e9) / (result3.kernel_ms / 1000.0);
    cout << "GPU 有效带宽：" << setprecision(2) << gpu_bandwidth << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Simple:      " << setw(8) << setprecision(4) << result1.kernel_ms << " ms (基准)\n";
    cout << "Convergent:  " << setw(8) << result2.kernel_ms << " ms ("
         << setprecision(2) << result1.kernel_ms / result2.kernel_ms << "x)\n";
    cout << "Shared Mem:  " << setw(8) << result3.kernel_ms << " ms ("
         << setprecision(2) << result1.kernel_ms / result3.kernel_ms << "x)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    
    // 验证 Simple Reduce
    float gpu_result1 = 0.0f;
    reduce_sum_gpu(h_input, gpu_result1, 1, simple_reduce_sum);
    verify_results(gpu_result1, cpu_result, "Simple Reduce");

    // 验证 Convergent Reduce
    float gpu_result2 = 0.0f;
    reduce_sum_gpu(h_input, gpu_result2, 1, convergent_reduce_sum);
    verify_results(gpu_result2, cpu_result, "Convergent Reduce");

    // 验证 Shared Memory Reduce
    float gpu_result3 = 0.0f;
    reduce_sum_gpu(h_input, gpu_result3, 1, shared_reduce_sum);
    verify_results(gpu_result3, cpu_result, "Shared Mem Reduce");

    // GPU/CPU 结果一致性验证
    bool all_passed = (fabs(gpu_result1 - cpu_result) < 1e-3) &&
                      (fabs(gpu_result2 - cpu_result) < 1e-3) &&
                      (fabs(gpu_result3 - cpu_result) < 1e-3);
    if (all_passed) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}