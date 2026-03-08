// Warp Shuffle - 线程间数据交换
#include <code_abbreviation.h>

// 广播 - 将某线程的值广播到 warp 内所有线程（GPU kernel，手写）
__global__ void kernel_warp_broadcast(CPFloat input, PFloat output, CInt n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = input[tid];

/* 
__shfl_sync(mask, var, srcLane) 的语义：在一个 warp 内，让所有参与的 lane都拿到 srcLane 那个线程的 var 值
这里 srcLane = 0：广播当前 warp 的 lane0 的 val 到所有 lane
mask = 0xffffffff：表示“我认为 warp 32 个 lane 都活跃并参与” 
*/
        val = __shfl_sync(0xffffffff, val, 0); // 从 lane 0 广播到整个 warp
        output[tid] = val;
    }
}

// XOR Shuffle - 蝴蝶式交换（GPU kernel，手写）
__global__ void kernel_warp_xor_shuffle(CPFloat input, PFloat output, CInt n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = input[tid];
/*
__shfl_xor_sync(mask, var, laneMask)：每个 lane 从 lane ^ laneMask 那个 lane 取 var。
这里 laneMask = 16：
lane 0 ↔ lane 16
lane 1 ↔ lane 17
...
lane 15 ↔ lane 31
本质是 warp 内“跨半个 warp”的配对交换。
*/
        val = __shfl_xor_sync(0xffffffff, val, 16); // 与 lane ^ 16 的线程交换
        output[tid] = val;
    }
}

// Up/Down Shuffle - 相邻线程交换（GPU kernel，手写）
__global__ void kernel_warp_up_down_shuffle(CPFloat input, PFloat output, CInt n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = input[tid];
/*
__shfl_down_sync(mask, var, delta)：从 lane + delta 取值（超出范围会返回自身值，细节与 mask/活跃性相关）
__shfl_up_sync(mask, var, delta)：从 lane - delta 取值。
*/
        int lane = threadIdx.x % 32;

        float down_val = __shfl_down_sync(0xffffffff, val, 1);
        float up_val   = __shfl_up_sync(0xffffffff, val, 1);

        if (lane % 2 == 0) {
            val = down_val; // 偶数 lane 向下交换
        } else {
            val = up_val;   // 奇数 lane 向上交换
        }
        output[tid] = val;
    }
}

// Warp Reduce Sum - Warp内归约求和（GPU kernel，手写）
__global__ void kernel_warp_reduce_sum(CPFloat input, PFloat output, CInt n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = input[tid];
        
        // 快速内归约 (蝴蝶法 / Down法)
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        // 将单个 warp 内的最终归约结果（只在 lane 0 是完整的总和）写出到 output
        int lane = threadIdx.x % 32;
        if (lane == 0) {
            output[tid / 32] = val;
        }
    }
}

// 广播（CPU，手写）
void warp_broadcast_cpu(CPFloat input, PFloat output, CInt n) {
    for (int i = 0; i < n; ++i) {
        int warp_start = (i / 32) * 32;
        output[i] = input[warp_start]; // 统一获取当前 warp 开始的第 0 个值
    }
}

// XOR Shuffle（CPU，手写）
void warp_xor_shuffle_cpu(CPFloat input, PFloat output, CInt n) {
    for (int i = 0; i < n; ++i) {
        int lane = i % 32;
        int warp_start = i - lane;
        int target_lane = lane ^ 16;
        output[i] = input[warp_start + target_lane];
    }
}

// Up/Down Shuffle（CPU，手写）
void warp_up_down_shuffle_cpu(CPFloat input, PFloat output, CInt n) {
    for (int i = 0; i < n; ++i) {
        int lane = i % 32;
        int warp_start = i - lane;
        int target_lane = lane;
        if (lane % 2 == 0) {
            target_lane = (lane + 1 < 32) ? lane + 1 : lane; // down 1
        } else {
            target_lane = (lane - 1 >= 0) ? lane - 1 : lane; // up 1
        }
        output[i] = input[warp_start + target_lane];
    }
}

// Warp Reduce Sum（CPU，手写）
void warp_reduce_sum_cpu(CPFloat input, PFloat output, CInt n) {
    int num_warps = n / 32;
    for (int i = 0; i < num_warps; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 32; ++j) {
            sum += input[i * 32 + j];
        }
        output[i] = sum;
    }
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = 1e-4f) {
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


// 通用 Warp Shuffle 算子测试框架 GPU 封装（GPU，部分手写，部分 AI 生成）
template<typename KernelFunc>
GpuTimingResult warp_shuffle_gpu(CRMatrix h_input, RMatrix h_output, CInt n, CInt iterations, KernelFunc kernel, bool is_reduce = false) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    
    CSize size_input = n * FSIZE;
    // 如果是降维归约操作，输出大小为 n / 32
    CSize size_output = is_reduce ? (n / 32) * FSIZE : size_input;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_output));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    // 强制 Block 大小为 256
    const int BLOCK_SIZE = 256;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid(cdiv(n, BLOCK_SIZE));
    
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
    // 设置 32M 元素，刚好填满中等规模内存带宽请求
    CInt n = 32 * 1024 * 1024;
    CInt iterations = 100;
    const int BLOCK_SIZE = 256;

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      Warp Primitives 性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_input(n);
    Matrix h_output_cpu(n);    // 大多数 Shuffle 的等长数组
    Matrix h_output_gpu(n);
    
    Matrix h_reduce_cpu(n / 32, 0.0f); // 专门应对归约数组
    Matrix h_reduce_gpu(n / 32, 0.0f);

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(rand() % 100) / 100.0f;
    }

    // ---------------------------------------------------------
    // CPU 计算（先执行 CPU，后执行 GPU）
    // ---------------------------------------------------------
    cout << "--- CPU 计时 (计算所有版本) ---\n";
    CpuTimer cpuTimer;
    
    cpuTimer.start();
    warp_broadcast_cpu(h_input.data(), h_output_cpu.data(), n);
    cpuTimer.stop();
    double cpu_bcast_ms = cpuTimer.elapsed_ms();
    
    cpuTimer.start();
    warp_xor_shuffle_cpu(h_input.data(), h_output_cpu.data(), n); // 共用缓存以节省内存
    cpuTimer.stop();
    double cpu_xor_ms = cpuTimer.elapsed_ms();
    
    cpuTimer.start();
    warp_up_down_shuffle_cpu(h_input.data(), h_output_cpu.data(), n);
    cpuTimer.stop();
    double cpu_ud_ms = cpuTimer.elapsed_ms();
    
    cpuTimer.start();
    warp_reduce_sum_cpu(h_input.data(), h_reduce_cpu.data(), n);
    cpuTimer.stop();
    double cpu_red_ms = cpuTimer.elapsed_ms();

    cout << "CPU Broadcast 执行时间：" << setw(8) << cpu_bcast_ms << " ms\n";
    cout << "CPU XOR       执行时间：" << setw(8) << cpu_xor_ms << " ms\n";
    cout << "CPU Up/Down   执行时间：" << setw(8) << cpu_ud_ms << " ms\n";
    cout << "CPU Reduce    执行时间：" << setw(8) << cpu_red_ms << " ms\n";
    cout << "\n";

    // ---------------------------------------------------------
    // GPU 版本测试
    // ---------------------------------------------------------
    
    cout << "--- GPU 版本 1: Warp Broadcast ---\n";
    GpuTimingResult res_bcast = warp_shuffle_gpu(h_input, h_output_gpu, n, iterations, kernel_warp_broadcast, false);
    cout << "H2D 传输时间：   " << setw(8) << res_bcast.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_bcast.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_bcast.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_bcast.total_ms << " ms\n";
    // 每次测完必须更新对应的 CPU 真值进行一次验证（复用的话会被覆盖）
    warp_broadcast_cpu(h_input.data(), h_output_cpu.data(), n);
    bool pass1 = verify_results(h_output_gpu, h_output_cpu, "Warp Broadcast");
    cout << "\n";

    cout << "--- GPU 版本 2: XOR Shuffle ---\n";
    GpuTimingResult res_xor = warp_shuffle_gpu(h_input, h_output_gpu, n, iterations, kernel_warp_xor_shuffle, false);
    cout << "Kernel 执行时间：" << setw(8) << res_xor.kernel_ms << " ms (" << iterations << " 次平均)\n";
    warp_xor_shuffle_cpu(h_input.data(), h_output_cpu.data(), n);
    bool pass2 = verify_results(h_output_gpu, h_output_cpu, "XOR Shuffle");
    cout << "\n";

    cout << "--- GPU 版本 3: Up/Down Shuffle ---\n";
    GpuTimingResult res_ud = warp_shuffle_gpu(h_input, h_output_gpu, n, iterations, kernel_warp_up_down_shuffle, false);
    cout << "Kernel 执行时间：" << setw(8) << res_ud.kernel_ms << " ms (" << iterations << " 次平均)\n";
    warp_up_down_shuffle_cpu(h_input.data(), h_output_cpu.data(), n);
    bool pass3 = verify_results(h_output_gpu, h_output_cpu, "Up/Down Shuffle");
    cout << "\n";

    cout << "--- GPU 版本 4: Warp Reduce Sum ---\n";
    // 归约操作极容易因为 float 的累加顺序导致末尾产生细微误差，放宽 epsilon
    GpuTimingResult res_red = warp_shuffle_gpu(h_input, h_reduce_gpu, n, iterations, kernel_warp_reduce_sum, true);
    cout << "Kernel 执行时间：" << setw(8) << res_red.kernel_ms << " ms (" << iterations << " 次平均)\n";
    bool pass4 = verify_results(h_reduce_gpu, h_reduce_cpu, "Warp Reduce Sum", 1e-2f);
    cout << "\n";

    // ---------------------------------------------------------
    // 性能分析
    // ---------------------------------------------------------
    cout << "--- 性能分析 ---\n";
    double speedup_bcast = cpu_bcast_ms / res_bcast.kernel_ms;
    double speedup_red = cpu_red_ms / res_red.kernel_ms;
    cout << "CPU vs Broadcast Kernel 加速比：" << setprecision(2) << speedup_bcast << "x\n";
    cout << "CPU vs Reduce Sum Kernel 加速比：" << setprecision(2) << speedup_red << "x\n";

    // 带宽计算 (基础读写，对于Reduce，因为输出数组小了32倍，写的数据量大幅下降)
    double bytes_base = n * FSIZE + n * FSIZE; 
    double bytes_red = n * FSIZE + (n / 32) * FSIZE; 
    
    double bw_bcast = (bytes_base / 1e9) / (res_bcast.kernel_ms / 1000.0);
    double bw_red = (bytes_red / 1e9) / (res_red.kernel_ms / 1000.0);
    
    cout << "Warp Broadcast 有效带宽：" << setprecision(2) << bw_bcast << " GB/s\n";
    cout << "Warp Reduce Sum 有效带宽：" << setprecision(2) << bw_red << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Warp Broadcast:  " << setw(8) << setprecision(4) << res_bcast.kernel_ms << " ms (基准)\n";
    cout << "XOR Shuffle:     " << setw(8) << res_xor.kernel_ms << " ms ("
         << setprecision(2) << res_bcast.kernel_ms / res_xor.kernel_ms << "x)\n";
    cout << "Up/Down Shuffle: " << setw(8) << res_ud.kernel_ms << " ms ("
         << setprecision(2) << res_bcast.kernel_ms / res_ud.kernel_ms << "x)\n";
    cout << "Warp Reduce Sum: " << setw(8) << res_red.kernel_ms << " ms (高度内存密集与运算)\n";
    cout << "\n";

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2 && pass3 && pass4) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
