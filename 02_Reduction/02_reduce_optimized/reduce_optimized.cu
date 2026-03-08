#include <code_abbreviation.h>


// 归约-任意长度（GPU kernel，手写）
__global__ void segmented_reduce_sum(PFloat input, PFloat output, CInt length) {
    __shared__ float shared_data[BLOCK_SIZE];
    CInt tid = threadIdx.x;
    CInt sid = 2 * blockDim.x * blockIdx.x + tid;

    // 边界检查
    CFloat v1 = (sid < length) ? input[sid] : 0.0f;
    CFloat v2 = (sid + BLOCK_SIZE < length) ? input[sid + BLOCK_SIZE] : 0.0f;
    shared_data[tid] = v1 + v2;

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
    }

    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

// 归约-线程粗化（GPU kernel，手写）
__global__ void coarsened_reduce_sum(PFloat input, PFloat output, CInt length) {
    __shared__ float shared_data[BLOCK_SIZE];
    CInt tid = threadIdx.x;
    CInt sid = 2 * COARSE_FACTOR * blockDim.x * blockIdx.x + tid;

    // 每个线程先累加多个元素
    float sum = 0.0f;
    if (sid < length) {
        sum = input[sid];
        for (int i = 1; i < COARSE_FACTOR * 2; ++i) {
            if (sid + i * BLOCK_SIZE < length) {
                sum += input[sid + i * BLOCK_SIZE];
            }
        }
    }
    shared_data[tid] = sum;

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
    }

    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

// 归约-最大值（GPU kernel，手写）
__global__ void coarsened_reduce_max(PFloat input, PFloat output, CInt length) {
    __shared__ float shared_data[BLOCK_SIZE];
    CInt tid = threadIdx.x;
    CInt sid = 2 * COARSE_FACTOR * blockDim.x * blockIdx.x + tid;

    // 每个线程先找多个元素的最大值
    float max_val = -INFINITY;
    if (sid < length) {
        max_val = input[sid];
        for (int i = 1; i < COARSE_FACTOR * 2; ++i) {
            if (sid + i * BLOCK_SIZE < length) {
                max_val = fmaxf(max_val, input[sid + i * BLOCK_SIZE]);
            }
        }
    }
    shared_data[tid] = max_val;

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
    }

    if (tid == 0) {
        // 仅适用于int类型的atomicMax，因为CUDA不支持float类型的atomicMax
        // atomicMax((unsigned int*)output, __float_as_uint(shared_data[0]));

        // 或者，负数可能出错
        float old;
        do {
            old = *output;
            if (shared_data[0] <= old) break;
        } while (atomicCAS((unsigned int*)output, __float_as_uint(old), __float_as_uint(shared_data[0])) != __float_as_uint(old));
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

// 归约-最大值（CPU，手写）
float reduction_max_cpu(CPFloat data, CInt length) {
    float max_val = data[0];
    for (int i = 1; i < length; i++) {
        max_val = max(max_val, data[i]);
    }
    return max_val;
}

// 验证结果（AI 生成）
bool verify_results(float gpu_result, float cpu_result, const string& kernel_name) {
    // 使用相对误差判断，适用于大数值
    float rel_error = fabs(gpu_result - cpu_result) / fabs(cpu_result);
    if (rel_error < 1e-5 || fabs(gpu_result - cpu_result) < 1e-3) {
        cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result 
             << " (期望 " << cpu_result << ")\n";
        return true;
    } else {
        cout << "✗ " << kernel_name << " FAILED: 结果 " << gpu_result 
             << " (期望 " << cpu_result << ", 误差 " << rel_error << ")\n";
        return false;
    }
}


// 通用归约求和 GPU 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult reduce_sum_gpu(CRMatrix h_input, float& h_output, CInt iterations, KernelFunc kernel, bool use_coarse = true) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    CInt length = static_cast<int>(h_input.size());
    CSize size_input = length * FSIZE;
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

    // 计算 grid 大小
    const dim3 block(BLOCK_SIZE);
    int grid_size = use_coarse ? cdiv(static_cast<unsigned int>(length), BLOCK_SIZE * 2 * COARSE_FACTOR) 
                               : cdiv(static_cast<unsigned int>(length), BLOCK_SIZE * 2);
    const dim3 grid(grid_size);

    // Kernel 预热
    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_output, &zero, FSIZE, cudaMemcpyHostToDevice));
    kernel<<<grid, block>>>(d_input, d_output, length);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 执行
    float total_kernel_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        // 重新拷贝数据
        // CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
        
        // atomicAdd 需要清零 output
        CUDA_CHECK(cudaMemcpy(d_output, &zero, FSIZE, cudaMemcpyHostToDevice));
        
        // 只计时 kernel 执行
        timerKernel.start();
        kernel<<<grid, block>>>(d_input, d_output, length);
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

// 归约最大值 GPU 封装（GPU，手写）
GpuTimingResult reduce_max_gpu(CRMatrix h_input, float& h_output, CInt iterations) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    CInt length = static_cast<int>(h_input.size());
    CSize size_input = length * FSIZE;
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
    const dim3 grid(cdiv(static_cast<unsigned int>(length), BLOCK_SIZE * 2 * COARSE_FACTOR));

    // Kernel 预热
    float neg_inf = -INFINITY;
    CUDA_CHECK(cudaMemcpy(d_output, &neg_inf, FSIZE, cudaMemcpyHostToDevice));
    coarsened_reduce_max<<<grid, block>>>(d_input, d_output, length);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 执行
    float total_kernel_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        // 重新拷贝数据
        // CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
        
        // atomicMax 需要初始化为 -INFINITY
        CUDA_CHECK(cudaMemcpy(d_output, &neg_inf, FSIZE, cudaMemcpyHostToDevice));
        
        // 只计时 kernel 执行
        timerKernel.start();
        coarsened_reduce_max<<<grid, block>>>(d_input, d_output, length);
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
    CInt n = 1 << 20;  // 1M 元素
    CInt iterations = 100;

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "   Reduce Optimized 性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " (" << (n >> 20) << " M) 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "粗化因子：" << COARSE_FACTOR << "\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_input(n);
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i % 100) / 100.0f;  // 0.00 ~ 0.99
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    float cpu_sum = reduce_sum_cpu(h_input.data(), n);
    float cpu_max = reduction_max_cpu(h_input.data(), n);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "CPU 求和结果：   " << cpu_sum << "\n";
    cout << "CPU 最大值结果： " << cpu_max << "\n";
    cout << "\n";

    float gpu_result = 0.0f;

    // GPU 版本 1：Segmented Reduce Sum
    cout << "--- GPU 版本 1: Segmented Reduce Sum ---\n";
    GpuTimingResult result1 = reduce_sum_gpu(h_input, gpu_result, iterations, segmented_reduce_sum, false);
    cout << "H2D 传输时间：   " << setw(8) << result1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result1.total_ms << " ms\n";
    float gpu_sum1 = gpu_result;
    cout << "\n";

    // GPU 版本 2：Coarsened Reduce Sum
    cout << "--- GPU 版本 2: Coarsened Reduce Sum ---\n";
    GpuTimingResult result2 = reduce_sum_gpu(h_input, gpu_result, iterations, coarsened_reduce_sum, true);
    cout << "H2D 传输时间：   " << setw(8) << result2.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result2.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result2.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result2.total_ms << " ms\n";
    float gpu_sum2 = gpu_result;
    cout << "\n";

    // GPU 版本 3：Coarsened Reduce Max
    cout << "--- GPU 版本 3: Coarsened Reduce Max ---\n";
    GpuTimingResult result3 = reduce_max_gpu(h_input, gpu_result, iterations);
    cout << "H2D 传输时间：   " << setw(8) << result3.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result3.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result3.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result3.total_ms << " ms\n";
    float gpu_max = gpu_result;
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_kernel = cpu_time_ms / result2.kernel_ms;
    double speedup_total = cpu_time_ms / result2.total_ms;
    cout << "CPU vs GPU Kernel 加速比：" << setprecision(2) << speedup_kernel << "x\n";
    cout << "CPU vs GPU 总时间加速比：" << speedup_total << "x\n";

    // 带宽计算
    double bytes = n * FSIZE;
    double gpu_bandwidth = (bytes / 1e9) / (result2.kernel_ms / 1000.0);
    cout << "GPU 有效带宽：" << setprecision(2) << gpu_bandwidth << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Segmented:   " << setw(8) << setprecision(4) << result1.kernel_ms << " ms (基准)\n";
    cout << "Coarsened:   " << setw(8) << result2.kernel_ms << " ms ("
         << setprecision(2) << result1.kernel_ms / result2.kernel_ms << "x)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(gpu_sum1, cpu_sum, "Segmented Sum");
    bool pass2 = verify_results(gpu_sum2, cpu_sum, "Coarsened Sum");
    bool pass3 = verify_results(gpu_max, cpu_max, "Coarsened Max");

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2 && pass3) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}