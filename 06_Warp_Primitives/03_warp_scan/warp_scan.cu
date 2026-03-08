// Warp Scan - 快速前缀和
#include <code_abbreviation.h>

// Block-level 包含型前缀和（GPU kernel，手写）
__global__ void block_scan_inclusive(CPFloat input, PFloat output, CInt n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? input[tid] : 0.0f;

    float inclusive_val = kernel_warp_scan_inclusive(val);

    __shared__ float warp_sums[32];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 31) {
        warp_sums[warp_id] = inclusive_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        float warp_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        
        // 这里需要对 Base 进行 Exclusive Scan，因为 Warp N 不需要加上自己的总和
        float dump_total;
        float base_offset = kernel_warp_scan_exclusive(warp_sum, dump_total);
        
        if (lane_id < num_warps) {
            warp_sums[lane_id] = base_offset;
        }
    }
    __syncthreads();

    float my_warp_base = warp_sums[warp_id];
    
    if (tid < n) {
        output[tid] = inclusive_val + my_warp_base;
    }
}

// Block-level 排他型前缀和（GPU kernel，手写）
__global__ void block_scan_exclusive(CPFloat input, PFloat output, CInt n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? input[tid] : 0.0f;

    float warp_total;
    float exclusive_val = kernel_warp_scan_exclusive(val, warp_total);

    __shared__ float warp_sums[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = warp_total; 
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        float warp_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        
        float dump_total;
        float base_offset = kernel_warp_scan_exclusive(warp_sum, dump_total);
        
        if (lane_id < num_warps) {
            warp_sums[lane_id] = base_offset;
        }
    }
    __syncthreads();

    float my_warp_base = warp_sums[warp_id];
    
    if (tid < n) {
        output[tid] = exclusive_val + my_warp_base;
    }
}

// Block Inclusive Scan（CPU，手写）
void block_scan_inclusive_cpu(CPFloat input, PFloat output, CInt n, CInt block_size) {
    int num_blocks = cdiv(n, block_size);
    for (int b = 0; b < num_blocks; ++b) {
        float sum = 0.0f;
        int start_idx = b * block_size;
        int end_idx = std::min(start_idx + block_size, n);
        
        for (int i = start_idx; i < end_idx; ++i) {
            sum += input[i];
            output[i] = sum;
        }
    }
}

// Block Exclusive Scan（CPU，手写）
void block_scan_exclusive_cpu(CPFloat input, PFloat output, CInt n, CInt block_size) {
    int num_blocks = cdiv(n, block_size);
    for (int b = 0; b < num_blocks; ++b) {
        float sum = 0.0f;
        int start_idx = b * block_size;
        int end_idx = std::min(start_idx + block_size, n);
        
        for (int i = start_idx; i < end_idx; ++i) {
            output[i] = sum;
            sum += input[i];
        }
    }
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = 1e-2f) {
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

    cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result[1023] << " (期望 " << cpu_result[1023] << ")\n";
    return true;
}

// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      
    float kernel_ms;   
    float d2h_ms;      
    float total_ms;    
};

// 通用 Scan 封装（GPU，部分手写，部分AI 生成）
template<typename KernelFunc>
GpuTimingResult warp_scan_gpu(CRMatrix h_input, RMatrix h_output, CInt n, CInt block_size, CInt iterations, KernelFunc kernel) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    
    CSize size_input = n * FSIZE;
    CSize size_output = size_input;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_output));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const dim3 block(block_size);
    const dim3 grid(cdiv(n, block_size));
    
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
    // 设置 32M 元素，用于施压带宽
    CInt n = 32 * 1024 * 1024;
    CInt iterations = 100;
    const int BLOCK_SIZE = 1024;     // 使用 1024 最大线程测试 Shared Memory 的协作极限

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "        Warp & Block Scan 性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程 (支持最大 32 Warps 协作)\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    Matrix h_input(n);
    Matrix h_output_cpu(n, 0.0f);
    Matrix h_output_gpu(n, 0.0f);

    srand(101);
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(rand() % 100) / 1000.0f; 
    }

    // ---------------------------------------------------------
    // CPU 计算
    // ---------------------------------------------------------
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    
    cpuTimer.start();
    block_scan_inclusive_cpu(h_input.data(), h_output_cpu.data(), n, BLOCK_SIZE);
    cpuTimer.stop();
    double cpu_inc_ms = cpuTimer.elapsed_ms();
    
    cpuTimer.start();
    block_scan_exclusive_cpu(h_input.data(), h_output_cpu.data(), n, BLOCK_SIZE);
    cpuTimer.stop();
    double cpu_exc_ms = cpuTimer.elapsed_ms();

    cout << "CPU Inclusive Scan 执行时间：" << setw(8) << cpu_inc_ms << " ms\n";
    cout << "CPU Exclusive Scan 执行时间：" << setw(8) << cpu_exc_ms << " ms\n";
    cout << "\n";

    // ---------------------------------------------------------
    // GPU 版本测试
    // ---------------------------------------------------------
    
    cout << "--- GPU 版本 1: Block Inclusive Scan ---\n";
    GpuTimingResult res_inc = warp_scan_gpu(h_input, h_output_gpu, n, BLOCK_SIZE, iterations, block_scan_inclusive);
    cout << "H2D 传输时间：   " << setw(8) << res_inc.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_inc.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_inc.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_inc.total_ms << " ms\n";
    
    block_scan_inclusive_cpu(h_input.data(), h_output_cpu.data(), n, BLOCK_SIZE);
    
    // Prefix Scan 非常容易产生超过 float 尾数的舍入浮定累计误差，因此容差设置略放宽
    bool pass1 = verify_results(h_output_gpu, h_output_cpu, "Block Inclusive Scan", 1e-1f); 
    cout << "\n";

    cout << "--- GPU 版本 2: Block Exclusive Scan ---\n";
    GpuTimingResult res_exc = warp_scan_gpu(h_input, h_output_gpu, n, BLOCK_SIZE, iterations, block_scan_exclusive);
    cout << "H2D 传输时间：   " << setw(8) << res_exc.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_exc.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_exc.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_exc.total_ms << " ms\n";
    
    block_scan_exclusive_cpu(h_input.data(), h_output_cpu.data(), n, BLOCK_SIZE);
    bool pass2 = verify_results(h_output_gpu, h_output_cpu, "Block Exclusive Scan", 1e-1f);
    cout << "\n";

    // ---------------------------------------------------------
    // 性能分析
    // ---------------------------------------------------------
    cout << "--- 性能分析 ---\n";
    double speedup_inc = cpu_inc_ms / res_inc.kernel_ms;
    double speedup_exc = cpu_exc_ms / res_exc.kernel_ms;
    cout << "CPU vs Inclusive Scan 加速比：" << setprecision(2) << speedup_inc << "x\n";
    cout << "CPU vs Exclusive Scan 加速比：" << setprecision(2) << speedup_exc << "x\n";

    // Scan是严格的 1 进 1 出计算机制，读 128MB，写 128MB
    double bytes_processed = n * FSIZE + n * FSIZE; 
    
    double bw_inc = (bytes_processed / 1e9) / (res_inc.kernel_ms / 1000.0);
    double bw_exc = (bytes_processed / 1e9) / (res_exc.kernel_ms / 1000.0);
    
    cout << "Inclusive Scan 有效带宽：" << setprecision(2) << bw_inc << " GB/s\n";
    cout << "Exclusive Scan 有效带宽：" << setprecision(2) << bw_exc << " GB/s\n";
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
