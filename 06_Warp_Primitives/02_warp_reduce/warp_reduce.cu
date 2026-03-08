// Warp Reduce - 无 shared memory 的归约
#include <code_abbreviation.h>

// Block-level 归约汇总求和（GPU kernel，手写）
__global__ void block_reduce_sum(CPFloat input, PFloat output, CInt n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = (tid < n) ? input[tid] : 0.0f; 
    sum = kernel_warp_reduce_sum(sum);

    __shared__ float shared_warp_sums[32]; // 最多 32 个 Warp

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        shared_warp_sums[warp_id] = sum; // 每个 Warp 的第一个线程写入共享内存
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;

    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? shared_warp_sums[lane_id] : 0.0f; 
        sum = kernel_warp_reduce_sum(sum); 

        if (lane_id == 0) {
            output[blockIdx.x] = sum; 
        }
    }
}

// Block-level 归约汇总求最大值（GPU kernel，手写）
__global__ void block_reduce_max(CPFloat input, PFloat output, CInt n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float max_val = (tid < n) ? input[tid] : -INFINITY;
    
    max_val = kernel_warp_reduce_max(max_val);
    
    __shared__ float shared_warp_maxs[32];
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        shared_warp_maxs[warp_id] = max_val;
    } 
    __syncthreads();
    
    int num_warps = blockDim.x / 32;
    
    if (warp_id == 0) {
        max_val = (lane_id < num_warps) ? shared_warp_maxs[lane_id] : -INFINITY;
        max_val = kernel_warp_reduce_max(max_val);
        
        if (lane_id == 0) {
            output[blockIdx.x] = max_val;
        }
    }
}

// Block 归约求和（CPU，手写）
void block_reduce_sum_cpu(CPFloat input, PFloat output, CInt n, CInt block_size) {
    int num_blocks = cdiv(n, block_size);
    for (int b = 0; b < num_blocks; ++b) {
        float sum = 0.0f;
        int start_idx = b * block_size;
        int end_idx = std::min(start_idx + block_size, n);
        
        for (int i = start_idx; i < end_idx; ++i) {
            sum += input[i];
        }
        output[b] = sum;
    }
}

// Block 归约求最大值（CPU，手写）
void block_reduce_max_cpu(CPFloat input, PFloat output, CInt n, CInt block_size) {
    int num_blocks = cdiv(n, block_size);
    for (int b = 0; b < num_blocks; ++b) {
        float max_val = -1e30f;
        int start_idx = b * block_size;
        int end_idx = std::min(start_idx + block_size, n);
        
        for (int i = start_idx; i < end_idx; ++i) {
            max_val = std::max(max_val, input[i]);
        }
        output[b] = max_val;
    }
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = 1e-3f) {
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配 (" 
             << gpu_result.size() << " vs " << cpu_result.size() << ")\n";
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
        if (diff > epsilon) {
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

    cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result[0] << " (期望 " << cpu_result[0] << ")\n";
    return true;
}

// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      
    float kernel_ms;   
    float d2h_ms;      
    float total_ms;    
};

// 通用 Reduce 封装（GPU，部分手写，部分AI 生成）
template<typename KernelFunc>
GpuTimingResult warp_reduce_gpu(CRMatrix h_input, RMatrix h_output, CInt n, CInt block_size, CInt iterations, KernelFunc kernel) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    
    CSize size_input = n * FSIZE;
    
    CInt grid_size = cdiv(n, block_size);
    CSize size_output = grid_size * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_output));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const dim3 block(block_size);
    const dim3 grid(grid_size);
    
    kernel<<<grid, block>>>(d_input, d_output, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_input, d_output, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_output, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    // 设置 32M 元素，刚好填满计算量
    CInt n = 32 * 1024 * 1024;
    CInt iterations = 100;
    const int BLOCK_SIZE = 256;      // 使用经典 256 线程，也就是 8个 Warp
    CInt num_blocks = cdiv(n, BLOCK_SIZE);

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      Warp & Block Reduce 性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "Grid 大小：" << num_blocks << " Blocks\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_input(n);
    Matrix h_output_cpu(num_blocks, 0.0f);
    Matrix h_output_gpu(num_blocks, 0.0f);

    srand(666);
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(rand() % 100) / 100.0f; // [0, 1) 的随机数
    }

    // ---------------------------------------------------------
    // CPU 计算
    // ---------------------------------------------------------
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    
    cpuTimer.start();
    block_reduce_sum_cpu(h_input.data(), h_output_cpu.data(), n, BLOCK_SIZE);
    cpuTimer.stop();
    double cpu_sum_ms = cpuTimer.elapsed_ms();
    
    cpuTimer.start();
    block_reduce_max_cpu(h_input.data(), h_output_cpu.data(), n, BLOCK_SIZE);
    cpuTimer.stop();
    double cpu_max_ms = cpuTimer.elapsed_ms();

    cout << "CPU Reduce Sum 执行时间：" << setw(8) << cpu_sum_ms << " ms\n";
    cout << "CPU Reduce Max 执行时间：" << setw(8) << cpu_max_ms << " ms\n";
    cout << "\n";

    // ---------------------------------------------------------
    // GPU 版本测试
    // ---------------------------------------------------------
    
    cout << "--- GPU 版本 1: Block Reduce Sum ---\n";
    GpuTimingResult res_sum = warp_reduce_gpu(h_input, h_output_gpu, n, BLOCK_SIZE, iterations, block_reduce_sum);
    cout << "H2D 传输时间：   " << setw(8) << res_sum.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_sum.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_sum.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_sum.total_ms << " ms\n";
    
    block_reduce_sum_cpu(h_input.data(), h_output_cpu.data(), n, BLOCK_SIZE);
    bool pass1 = verify_results(h_output_gpu, h_output_cpu, "Block Reduce Sum", 1e-1f); // 浮点大数归约，放宽容差
    cout << "\n";

    cout << "--- GPU 版本 2: Block Reduce Max ---\n";
    GpuTimingResult res_max = warp_reduce_gpu(h_input, h_output_gpu, n, BLOCK_SIZE, iterations, block_reduce_max);
    cout << "H2D 传输时间：   " << setw(8) << res_max.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_max.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_max.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_max.total_ms << " ms\n";
    
    block_reduce_max_cpu(h_input.data(), h_output_cpu.data(), n, BLOCK_SIZE);
    bool pass2 = verify_results(h_output_gpu, h_output_cpu, "Block Reduce Max"); // Max不容易积累误差，默认容差即可
    cout << "\n";

    // ---------------------------------------------------------
    // 性能分析
    // ---------------------------------------------------------
    cout << "--- 性能分析 ---\n";
    double speedup_sum = cpu_sum_ms / res_sum.kernel_ms;
    double speedup_max = cpu_max_ms / res_max.kernel_ms;
    cout << "CPU vs Reduce Sum 加速比：" << setprecision(2) << speedup_sum << "x\n";
    cout << "CPU vs Reduce Max 加速比：" << setprecision(2) << speedup_max << "x\n";

    // 计算实际读写数据量: 读取所有的 Input, 写入对应的 Output 块 (等于 num_blocks)
    double bytes_processed = n * FSIZE + num_blocks * FSIZE; 
    
    double bw_sum = (bytes_processed / 1e9) / (res_sum.kernel_ms / 1000.0);
    double bw_max = (bytes_processed / 1e9) / (res_max.kernel_ms / 1000.0);
    
    cout << "Reduce Sum 有效带宽：" << setprecision(2) << bw_sum << " GB/s\n";
    cout << "Reduce Max 有效带宽：" << setprecision(2) << bw_max << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
