#include <code_abbreviation.h>

// 前缀和-kogge_stone，和Hillis_Steele类似（GPU kernel，手写）
__global__ void kogge_stone_scan(PFloat input, PFloat output, CInt n) {
    extern __shared__ float shared_data[];
    CInt tid = threadIdx.x;

    // 包含扫描
    if (tid < n) {
        shared_data[tid] = input[tid];
    } else {
        shared_data[tid] = 0.0f;
    }

    // 不包含扫描
    // if (tid < n && tid != 0) {
    //     shared_data[tid] = input[tid - 1];
    // } else {
    //     shared_data[tid] = 0.0f;
    // }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float val = 0.0f;
        if (tid >= stride) {
            val = shared_data[tid] + shared_data[tid - stride];
        }
        __syncthreads();
        
        if (tid >= stride) {
            shared_data[tid] = val;
        }
    }

    if (tid < n) {
        output[tid] = shared_data[tid];
    }
}

// 前缀和-Brent-Kung 算法（GPU kernel，手写）
__global__ void brent_kung_scan(PFloat input, PFloat output, CInt n) {
    extern __shared__ float shared_data[];
    CInt tid = threadIdx.x;

    // 包含扫描
    if (tid < n) {
        shared_data[tid] = input[tid];
    } else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();
    // 不包含扫描
    // if (tid < n && tid != 0) {
    //     shared_data[tid] = input[tid - 1];
    // } else {
    //     shared_data[tid] = 0.0f;
    // }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        CInt index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            shared_data[index] += shared_data[index - stride];
        } 
        __syncthreads();
    }

    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        CInt index = (tid + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            shared_data[index + stride] += shared_data[index];
        }
        __syncthreads();
    }

    if (tid < n) {
        output[tid] = shared_data[tid];
    }
}

// 前缀和（CPU，手写）
void prefix_sum_cpu(CPFloat input, PFloat output, CInt n) {
    output[0] = input[0];
    for (int i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i];
    }

    // 不包含扫描
    // output[0] = 0.0f;
    // for (int i = 1; i < n; ++i) {
    //     output[i] = output[i - 1] + input[i - 1];
    // }
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = 1e-3f) {
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配 (GPU: " << gpu_result.size() 
             << ", CPU: " << cpu_result.size() << ")\n";
        return false;
    }
    
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    int error_count = 0;
    
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float diff = fabs(gpu_result[i] - cpu_result[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = static_cast<int>(i);
        }
        // 使用相对误差进行判断（考虑累积误差）
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
    
    cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result.back() 
         << " (期望 " << cpu_result.back() << ")\n";
    return true;
}

// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      // Host to Device 传输时间
    float kernel_ms;   // Kernel 执行时间（多次平均）
    float d2h_ms;      // Device to Host 传输时间
    float total_ms;    // 总时间
};

// 通用前缀和 GPU 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult scan_gpu(CRMatrix h_input, RMatrix h_output, CInt iterations, KernelFunc kernel) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    CInt length = static_cast<int>(h_input.size());
    CSize size_input = length * FSIZE;
    CSize size_output = length * FSIZE;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_output));

    // 创建计时器
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    // H2D
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    // 计算 grid 大小
    const dim3 block(BLOCK_SIZE);
    const dim3 grid(cdiv(static_cast<unsigned int>(length), BLOCK_SIZE));
    size_t sharedMemSize = BLOCK_SIZE * FSIZE;

    // 预热
    kernel<<<grid, block, sharedMemSize>>>(d_input, d_output, length);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 执行
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block, sharedMemSize>>>(d_input, d_output, length);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    // D2H
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_output, cudaMemcpyDeviceToHost));
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
    CInt n = BLOCK_SIZE;  // 当前 kernel 仅支持单 block
    CInt iterations = 100;

    CSize size_input = n * FSIZE;
    const double total_kb = size_input / 1024.0;

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      前缀和（Scan）性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_kb << " KB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_input(n);
    Matrix h_output_ks(n);    // Kogge-Stone 结果
    Matrix h_output_bk(n);    // Brent-Kung 结果
    Matrix h_output_cpu(n);   // CPU 参考结果

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(rand() % 10) / 10.0f;  // 0.0 ~ 0.9
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    prefix_sum_cpu(h_input.data(), h_output_cpu.data(), n);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1：Kogge-Stone
    cout << "--- GPU 版本 1: Kogge-Stone ---\n";
    GpuTimingResult result1 = scan_gpu(h_input, h_output_ks, iterations, kogge_stone_scan);
    cout << "H2D 传输时间：   " << setw(8) << result1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result1.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2：Brent-Kung
    cout << "--- GPU 版本 2: Brent-Kung ---\n";
    GpuTimingResult result2 = scan_gpu(h_input, h_output_bk, iterations, brent_kung_scan);
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

    // 带宽计算
    double bytes = 2.0 * n * FSIZE;  // 读取 + 写入
    double gpu_bandwidth = (bytes / 1e9) / (result2.kernel_ms / 1000.0);
    cout << "GPU 有效带宽：" << setprecision(2) << gpu_bandwidth << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Kogge-Stone: " << setw(8) << setprecision(4) << result1.kernel_ms << " ms (基准)\n";
    cout << "Brent-Kung:  " << setw(8) << result2.kernel_ms << " ms ("
         << setprecision(2) << result1.kernel_ms / result2.kernel_ms << "x)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_output_ks, h_output_cpu, "Kogge-Stone");
    bool pass2 = verify_results(h_output_bk, h_output_cpu, "Brent-Kung");
    bool pass3 = verify_results(h_output_ks, h_output_bk, "算法一致性");

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2 && pass3) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
