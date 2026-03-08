#include <code_abbreviation.h>

// 前缀和-线程粗化（GPU kernel，手写）
__global__ void coarse_scan(CPFloat input, PFloat output, CInt n) {
    extern __shared__ float shared_data[];
    PFloat section_sums = &shared_data[n];

    CInt tid = threadIdx.x;

    // 加载局部数据
    for (int i = 0; i < COARSE_FACTOR; ++i) {
        CInt index = tid * COARSE_FACTOR + i;
        if (index < n) {
            shared_data[index] = input[index];
        }
    }
    __syncthreads();

     // 计算局部前缀和
     for (int i = 1; i < COARSE_FACTOR; ++i) {
        CInt index = tid * COARSE_FACTOR + i;
        if (index < n) {
            shared_data[index] += shared_data[index - 1];
        }
     }
     __syncthreads();

     // 收集末尾值
     CInt num_sections = cdiv(n, COARSE_FACTOR);
     if (tid < num_sections) {
        CInt end_idx = (tid + 1) * COARSE_FACTOR - 1;
        if (end_idx < n) {
            section_sums[tid] = shared_data[end_idx];
        } else {
            section_sums[tid] = shared_data[n - 1];
        }
     }
     __syncthreads();

     // 对末尾继续KS扫描
     for (int stride = 1; stride < num_sections; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride && tid < num_sections) {
            val = section_sums[tid] + section_sums[tid - stride];
        }
        __syncthreads();

        if (tid >= stride && tid < num_sections) {
            section_sums[tid] = val;
        }
        __syncthreads();
     }

     // 分发段和
     for (int i = 0; i < COARSE_FACTOR; ++i) {
        CInt index = tid * COARSE_FACTOR + i;
        if (index < n) {
            CInt section_id = index / COARSE_FACTOR;
            if (section_id > 0) {
                shared_data[index] += section_sums[section_id - 1];
            }
            output[index] = shared_data[index];
        }
     } 
}

// 前缀和-分段扫描（GPU kernel，手写）
__global__ void segmented_scan(CPFloat input, PFloat output, PFloat block_sums, CInt n) {
    extern __shared__ float shared_data[];
    CInt tid = threadIdx.x;
    CInt gid = blockIdx.x * blockDim.x + tid;

    // 加载对应段数据到共享内存
    if (gid < n) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();

    // KS 扫描
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride) {
            val = shared_data[tid] + shared_data[tid - stride];
        }
        __syncthreads();

        if (tid >= stride) {
            shared_data[tid] = val;
        }
        __syncthreads();
    }

    // 对应段写回
    if (gid < n) {
        output[gid] = shared_data[tid];
    }

    // 记录当前block的最后一个值（段和）
    if (tid == blockDim.x - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = shared_data[tid];
    }
}

// 前缀和-添加块和（GPU kernel，手写）
__global__ void add_block_sums(PFloat output, CPFloat scanned_block_sums, CInt n) {
    CInt gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && gid < n) {
        output[gid] += scanned_block_sums[blockIdx.x - 1];
    }
}

// 前缀和（CPU，手写）
void prefix_sum_cpu(CPFloat input, PFloat output, CInt n) {
    output[0] = input[0];
    for (int i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i];
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

    cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result.back()
         << " (期望 " << cpu_result.back() << ")\n";
    return true;
}

// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;
    float kernel_ms;
    float d2h_ms;
    float total_ms;
};

// 粗化扫描 GPU 封装（GPU，手写）
GpuTimingResult coarse_scan_gpu(CRMatrix h_input, RMatrix h_output, CInt iterations) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;

    CInt n = static_cast<int>(h_input.size());
    CSize size = n * FSIZE;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_input, size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));

    // 计时器
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    // H2D
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    // 共享内存大小：数据 + section_sums
    CInt num_sections = cdiv(static_cast<unsigned int>(n), COARSE_FACTOR);
    size_t sharedMemSize = (n + num_sections) * FSIZE;

    const dim3 block(BLOCK_SIZE);
    const dim3 grid(1);  // 单 Block

    // 预热
    coarse_scan<<<grid, block, sharedMemSize>>>(d_input, d_output, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 计时
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        coarse_scan<<<grid, block, sharedMemSize>>>(d_input, d_output, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    // D2H
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    // 总时间
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    // 释放
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

// 分段扫描 GPU 封装（GPU，手写）
GpuTimingResult segmented_scan_gpu(CRMatrix h_input, RMatrix h_output, CInt iterations) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    PFloat d_block_sums = nullptr;
    PFloat d_scanned_block_sums = nullptr;

    CInt n = static_cast<int>(h_input.size());
    CSize size = n * FSIZE;

    // 计算 grid 配置
    const dim3 block(BLOCK_SIZE);
    const dim3 grid(cdiv(static_cast<unsigned int>(n), BLOCK_SIZE));
    CInt num_blocks = grid.x;
    CSize block_sums_size = num_blocks * FSIZE;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_input, size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));
    CUDA_CHECK(cudaMalloc((void**)&d_block_sums, block_sums_size));
    CUDA_CHECK(cudaMalloc((void**)&d_scanned_block_sums, block_sums_size));

    // 计时器
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    // H2D
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    size_t sharedMemSize = BLOCK_SIZE * FSIZE;

    // 预热（完整三遍扫描）
    segmented_scan<<<grid, block, sharedMemSize>>>(d_input, d_output, d_block_sums, n);
    if (num_blocks > 1) {
        const dim3 block2(min(num_blocks, BLOCK_SIZE));
        const dim3 grid2(1);
        segmented_scan<<<grid2, block2, sharedMemSize>>>(d_block_sums, d_scanned_block_sums, nullptr, num_blocks);
        add_block_sums<<<grid, block>>>(d_output, d_scanned_block_sums, n);
    }
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 计时
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        segmented_scan<<<grid, block, sharedMemSize>>>(d_input, d_output, d_block_sums, n);
        if (num_blocks > 1) {
            const dim3 block2(min(num_blocks, BLOCK_SIZE));
            const dim3 grid2(1);
            segmented_scan<<<grid2, block2, sharedMemSize>>>(d_block_sums, d_scanned_block_sums, nullptr, num_blocks);
            add_block_sums<<<grid, block>>>(d_output, d_scanned_block_sums, n);
        }
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    // D2H
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    // 总时间
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    // 释放
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_scanned_block_sums));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt iterations = 100;

    // 打印设备信息
    printDeviceInfo();

    // ==================== 测试场景 1：小数据量（算法对比）====================
    CInt n_small = BLOCK_SIZE * COARSE_FACTOR;  // 4096 元素
    CSize size_small = n_small * FSIZE;
    const double total_kb = size_small / 1024.0;

    cout << "========================================\n";
    cout << "   测试场景 1：小数据量（算法对比）\n";
    cout << "========================================\n";
    cout << "数组大小：" << n_small << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_kb << " KB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "粗化因子：" << COARSE_FACTOR << "\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化（小数据量）
    Matrix h_input_small(n_small);
    Matrix h_output_coarse(n_small);
    Matrix h_output_seg_small(n_small);
    Matrix h_output_cpu_small(n_small);

    srand(42);
    for (int i = 0; i < n_small; ++i) {
        h_input_small[i] = static_cast<float>(rand() % 10) / 10.0f;
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimerSmall;
    cpuTimerSmall.start();
    prefix_sum_cpu(h_input_small.data(), h_output_cpu_small.data(), n_small);
    cpuTimerSmall.stop();
    double cpu_time_small = cpuTimerSmall.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_small << " ms\n";
    cout << "CPU 结果（末尾）：" << h_output_cpu_small.back() << "\n";
    cout << "\n";

    // GPU 版本 1：Coarse Scan（单 Block 粗化扫描）
    cout << "--- GPU 版本 1: Coarse Scan ---\n";
    GpuTimingResult result_coarse = coarse_scan_gpu(h_input_small, h_output_coarse, iterations);
    cout << "H2D 传输时间：   " << setw(8) << result_coarse.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result_coarse.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result_coarse.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result_coarse.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2：Segmented Scan（三遍扫描法）
    cout << "--- GPU 版本 2: Segmented Scan ---\n";
    GpuTimingResult result_seg_small = segmented_scan_gpu(h_input_small, h_output_seg_small, iterations);
    cout << "H2D 传输时间：   " << setw(8) << result_seg_small.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result_seg_small.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result_seg_small.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result_seg_small.total_ms << " ms\n";
    cout << "\n";

    // 性能分析（小数据量）
    cout << "--- 性能分析 ---\n";
    double speedup_kernel_small = cpu_time_small / result_coarse.kernel_ms;
    double speedup_total_small = cpu_time_small / result_coarse.total_ms;
    cout << "CPU vs GPU Kernel 加速比：" << setprecision(2) << speedup_kernel_small << "x\n";
    cout << "CPU vs GPU 总时间加速比：" << speedup_total_small << "x\n";

    // 带宽计算（小数据量）
    double bytes_small = 2.0 * n_small * FSIZE;  // 读取 + 写入
    double gpu_bandwidth_small = (bytes_small / 1e9) / (result_coarse.kernel_ms / 1000.0);
    cout << "GPU 有效带宽：" << setprecision(2) << gpu_bandwidth_small << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比（小数据量）
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Coarse Scan:    " << setw(8) << setprecision(4) << result_coarse.kernel_ms << " ms (基准)\n";
    cout << "Segmented Scan: " << setw(8) << result_seg_small.kernel_ms << " ms ("
         << setprecision(2) << result_coarse.kernel_ms / result_seg_small.kernel_ms << "x)\n";
    cout << "\n";

    // 结果验证（小数据量）
    cout << "--- 结果验证 ---\n";
    bool pass_coarse = verify_results(h_output_coarse, h_output_cpu_small, "Coarse Scan");
    bool pass_seg_small = verify_results(h_output_seg_small, h_output_cpu_small, "Segmented Scan");
    bool pass_consistency = verify_results(h_output_coarse, h_output_seg_small, "算法一致性");

    // GPU/CPU 结果一致性验证（小数据量）
    if (pass_coarse && pass_seg_small && pass_consistency) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    // ==================== 测试场景 2：大数据量（性能测试）====================
    CInt n_large = 1 << 20;  // 1M 元素
    CSize size_large = n_large * FSIZE;
    const double total_mb = size_large / (1024.0 * 1024.0);

    cout << "\n========================================\n";
    cout << "   测试场景 2：大数据量（性能测试）\n";
    cout << "========================================\n";
    cout << "数组大小：" << n_large << " (" << (n_large >> 20) << " M) 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "Block 数量：" << cdiv(static_cast<unsigned int>(n_large), BLOCK_SIZE) << "\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化（大数据量）
    Matrix h_input_large(n_large);
    Matrix h_output_seg_large(n_large);
    Matrix h_output_cpu_large(n_large);

    srand(42);
    for (int i = 0; i < n_large; ++i) {
        h_input_large[i] = static_cast<float>(rand() % 10) / 10.0f;
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimerLarge;
    cpuTimerLarge.start();
    prefix_sum_cpu(h_input_large.data(), h_output_cpu_large.data(), n_large);
    cpuTimerLarge.stop();
    double cpu_time_large = cpuTimerLarge.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_large << " ms\n";
    cout << "CPU 结果（末尾）：" << h_output_cpu_large.back() << "\n";
    cout << "\n";

    // GPU 版本：Segmented Scan（三遍扫描法，大数据量）
    // 注：Coarse Scan 仅支持 n <= BLOCK_SIZE * COARSE_FACTOR，不适用于大数据量
    cout << "--- GPU 版本: Segmented Scan ---\n";
    GpuTimingResult result_seg_large = segmented_scan_gpu(h_input_large, h_output_seg_large, iterations);
    cout << "H2D 传输时间：   " << setw(8) << result_seg_large.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result_seg_large.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result_seg_large.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result_seg_large.total_ms << " ms\n";
    cout << "\n";

    // 性能分析（大数据量）
    cout << "--- 性能分析 ---\n";
    double speedup_kernel_large = cpu_time_large / result_seg_large.kernel_ms;
    double speedup_total_large = cpu_time_large / result_seg_large.total_ms;
    cout << "CPU vs GPU Kernel 加速比：" << setprecision(2) << speedup_kernel_large << "x\n";
    cout << "CPU vs GPU 总时间加速比：" << speedup_total_large << "x\n";

    // 带宽计算（大数据量）
    double bytes_large = 2.0 * n_large * FSIZE;  // 读取 + 写入
    double gpu_bandwidth_large = (bytes_large / 1e9) / (result_seg_large.kernel_ms / 1000.0);
    cout << "GPU 有效带宽：" << setprecision(2) << gpu_bandwidth_large << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比（小数据量 vs 大数据量）
    cout << "--- Kernel 性能对比（数据规模扩展）---\n";
    cout << "小数据量 Segmented: " << setw(8) << setprecision(4) << result_seg_small.kernel_ms << " ms (" << n_small << " 元素)\n";
    cout << "大数据量 Segmented: " << setw(8) << result_seg_large.kernel_ms << " ms (" << (n_large >> 20) << "M 元素)\n";
    cout << "数据量增长：" << setprecision(0) << (double)n_large / n_small << "x\n";
    cout << "Kernel 时间增长：" << setprecision(2) << result_seg_large.kernel_ms / result_seg_small.kernel_ms << "x\n";
    cout << "\n";

    // 结果验证（大数据量）
    cout << "--- 结果验证 ---\n";
    bool pass_seg_large = verify_results(h_output_seg_large, h_output_cpu_large, "Segmented Scan");

    // GPU/CPU 结果一致性验证（大数据量）
    if (pass_seg_large) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
