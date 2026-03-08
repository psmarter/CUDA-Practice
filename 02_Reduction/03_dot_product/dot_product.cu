#include <code_abbreviation.h>

// 点积-共享内存（GPU kernel，手写）
__global__ void shared_dot_product(CPFloat a, CPFloat b, PFloat output, CInt size) {
    __shared__ float shared_data[BLOCK_SIZE];
    CInt tid = threadIdx.x;
    CInt sid = 2 * blockDim.x * blockIdx.x + tid;

    float sum = 0.0f;
    if (sid < size) {
        sum += a[sid] * b[sid];
    }
    if (sid + BLOCK_SIZE < size) {
        sum += a[sid + BLOCK_SIZE] * b[sid + BLOCK_SIZE];
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

// 点积-线程粗化（GPU kernel，手写）
__global__ void coarsened_dot_product(CPFloat a, CPFloat b, PFloat output, CInt size) {
    __shared__ float shared_data[BLOCK_SIZE];
    CInt tid = threadIdx.x;
    CInt sid = 2 * COARSE_FACTOR * blockDim.x * blockIdx.x + tid;

    float sum = 0.0f;
    for (int i = 0; i < COARSE_FACTOR * 2; ++i) {
        if (sid + i * BLOCK_SIZE < size) {
            sum += a[sid + i * BLOCK_SIZE] * b[sid + i * BLOCK_SIZE];
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

// 点积-FMA优化（GPU kernel，手写）
__global__ void fma_dot_product(CPFloat a, CPFloat b, PFloat output, CInt size) {
    __shared__ float shared_data[BLOCK_SIZE];
    CInt tid = threadIdx.x;
    CInt sid = 2 * COARSE_FACTOR * blockDim.x * blockIdx.x + tid;

    float sum = 0.0f;
    for (int i = 0; i < COARSE_FACTOR * 2; ++i) {
        if (sid + i * BLOCK_SIZE < size) {
            // sum += a[sid + i * BLOCK_SIZE] * b[sid + i * BLOCK_SIZE]
            // 有些编译器会自动优化，使一个乘加操作变成一个FMA指令
            sum = fmaf(a[sid + i * BLOCK_SIZE], b[sid + i * BLOCK_SIZE], sum);
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

// 点积（CPU，手写）
float dot_product_cpu(CPFloat a, CPFloat b, CInt n) {
    double sum = 0.0;  // 使用 double 减少累积误差
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(sum);
}

// 验证结果（AI 生成）
bool verify_results(RFloat gpu_result, RFloat cpu_result, const string& kernel_name) {
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


// 通用点积 GPU 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult dot_product_gpu(CRMatrix h_a, CRMatrix h_b, RFloat h_output,  CInt iterations, KernelFunc kernel, bool use_coarse = true) {
    PFloat d_a = nullptr;
    PFloat d_b = nullptr;
    PFloat d_output = nullptr;
    CInt length = static_cast<int>(h_a.size());
    CSize size_input = length * FSIZE;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_a, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_output, FSIZE));

    // 创建计时器
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    // H2D
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_input, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_input, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    // 计算 grid 大小
    const dim3 block(BLOCK_SIZE);
    int grid_size = use_coarse ? cdiv(static_cast<unsigned int>(length), BLOCK_SIZE * 2 * COARSE_FACTOR)
                               : cdiv(static_cast<unsigned int>(length), BLOCK_SIZE * 2);
    const dim3 grid(grid_size);

    // 预热
    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_output, &zero, FSIZE, cudaMemcpyHostToDevice));
    kernel<<<grid, block>>>(d_a, d_b, d_output, length);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 执行
    float total_kernel_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        // 重新拷贝数据
        CUDA_CHECK(cudaMemcpy(d_output, &zero, FSIZE, cudaMemcpyHostToDevice));

        timerKernel.start();
        kernel<<<grid, block>>>(d_a, d_b, d_output, length);
        timerKernel.stop();
        total_kernel_ms += timerKernel.elapsed_ms();
    }
    CUDA_CHECK_LAST();
    result.kernel_ms = total_kernel_ms / static_cast<float>(iterations);

    // D2H
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, FSIZE, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    // 计算总时间
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt n = 1 << 20;  // 1M 元素
    CInt iterations = 100;

    CSize size_input = n * FSIZE;
    const double total_mb = 2.0 * size_input / (1024.0 * 1024.0);  // 两个输入向量

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      Dot Product 性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " (" << (n >> 20) << " M) 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB (两向量)\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "粗化因子：" << COARSE_FACTOR << "\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n\n";

    // 初始化数据
    Matrix h_a(n), h_b(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f;  // 全 1
        h_b[i] = 1.0f;  // 全 1，dot(a,b) = n
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    float cpu_result = dot_product_cpu(h_a.data(), h_b.data(), n);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "CPU 结果：       " << cpu_result << "\n\n";

    float gpu_result = 0.0f;

    // GPU 版本 1：Simple
    cout << "--- GPU 版本 1: Simple Dot Product ---\n";
    GpuTimingResult r1 = dot_product_gpu(h_a, h_b, gpu_result, iterations, shared_dot_product, false);
    cout << "H2D 传输时间：   " << setw(8) << r1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << setprecision(4) << r1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << r1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << r1.total_ms << " ms\n";
    float gpu_result1 = gpu_result;
    cout << "\n";

    // GPU 版本 2：Coarsened
    cout << "--- GPU 版本 2: Coarsened Dot Product ---\n";
    GpuTimingResult r2 = dot_product_gpu(h_a, h_b, gpu_result, iterations, coarsened_dot_product, true);
    cout << "H2D 传输时间：   " << setw(8) << r2.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << setprecision(4) << r2.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << r2.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << r2.total_ms << " ms\n";
    float gpu_result2 = gpu_result;
    cout << "\n";

    // GPU 版本 3：FMA
    cout << "--- GPU 版本 3: FMA Dot Product ---\n";
    GpuTimingResult r3 = dot_product_gpu(h_a, h_b, gpu_result, iterations, fma_dot_product, true);
    cout << "H2D 传输时间：   " << setw(8) << r3.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << setprecision(4) << r3.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << r3.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << r3.total_ms << " ms\n";
    float gpu_result3 = gpu_result;
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_kernel = cpu_time_ms / r3.kernel_ms;
    double speedup_total = cpu_time_ms / r3.total_ms;
    cout << "CPU vs GPU Kernel 加速比：" << setprecision(2) << speedup_kernel << "x\n";
    cout << "CPU vs GPU 总时间加速比：" << speedup_total << "x\n";

    // 带宽计算
    // 注：由于预热阶段将数据加载到 L2 缓存
    // 实际带宽可能超过 DRAM 理论峰值，这是 L2 缓存命中的正常现象
    double bytes = 2.0 * n * FSIZE;  // 读取两个向量
    double bandwidth = (bytes / 1e9) / (r3.kernel_ms / 1000.0);
    cout << "GPU 有效带宽：" << setprecision(2) << bandwidth << " GB/s\n";
    cout << "(RTX 4090 DRAM 理论峰值：~1008 GB/s，L2 峰值更高)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Simple:    " << setw(8) << setprecision(4) << r1.kernel_ms << " ms (基准)\n";
    cout << "Coarsened: " << setw(8) << setprecision(4) << r2.kernel_ms << " ms (" << setprecision(2) << r1.kernel_ms / r2.kernel_ms << "x)\n";
    cout << "FMA:       " << setw(8) << setprecision(4) << r3.kernel_ms << " ms (" << setprecision(2) << r1.kernel_ms / r3.kernel_ms << "x)\n";
    cout << "\n";
    
    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool p1 = verify_results(gpu_result1, cpu_result, "Simple");
    bool p2 = verify_results(gpu_result2, cpu_result, "Coarsened");
    bool p3 = verify_results(gpu_result3, cpu_result, "FMA");
    if (p1 && p2 && p3) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }
    cout << "\n========================================\n";
    return 0;
}