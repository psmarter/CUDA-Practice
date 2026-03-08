// Occupancy 优化 - 占用率分析与优化
#include <code_abbreviation.h>
#include <string>

// 1. GPU Kernel 函数（手写）

// 示例 kernel（可调整资源使用）(GPU kernel，手写)
template<int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void configurable_kernel(CPFloat input, PFloat output, CInt n) {
    // 采用块内合并访存 (Block-wise Coalesced Access)
    // 每个线程处理的连续元素实际上分布在块内的步长上
    float items[ITEMS_PER_THREAD];
    
    int base_idx = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD;
    
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = base_idx + i * BLOCK_SIZE + threadIdx.x;
        if (idx < n) {
            items[i] = input[idx];
        } else {
            items[i] = 0.0f;
        }
    }
    
    // 处理 (ILP 掩盖延迟)
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        items[i] *= 2.0f;
    }
    
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = base_idx + i * BLOCK_SIZE + threadIdx.x;
        if (idx < n) {
            output[idx] = items[i];
        }
    }
}

// 使用共享内存的 kernel（影响 occupancy）(GPU kernel，手写)
template<int BLOCK_SIZE, int SHARED_SIZE>
__global__ void shared_memory_kernel(CPFloat input, PFloat output, CInt n) {
    // 强制使用大量的 shared memory 来限制每个 SM 活跃的 Block 数量
    __shared__ float shared[SHARED_SIZE / sizeof(float)];
    
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_id < n) {
        // Dummy 写入共享内存防优化
        shared[threadIdx.x % (SHARED_SIZE / sizeof(float))] = input[global_id];
        __syncthreads();
        
        output[global_id] = shared[threadIdx.x % (SHARED_SIZE / sizeof(float))] * 2.0f;
    }
}

// 限制寄存器使用 (GPU kernel，手写)
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
template<int BLOCK_SIZE>
__global__ __launch_bounds__(256, 4)
void register_limited_kernel(CPFloat input, PFloat output, CInt n) {
    // 编译器会尝试将寄存器使用限制在 (最大每线程 64 个) 以保证至少能塞下 4 个 block
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < n) {
        output[global_id] = input[global_id] * 2.0f;
    }
}

// 2. CPU 参考实现函数（手写）

// CPU 算子基础参考 (CPU，手写)
void base_cpu(CRMatrix input, RMatrix output, CInt n) {
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] * 2.0f;
    }
}

// 3. verify_results 验证函数（AI 生成）

bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, CInt n, const string& kernel_name) {
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(gpu_result[i] - cpu_result[i]) > 1e-5f) {
            cout << "✗ " << kernel_name << " FAILED: 索引 " << i 
                 << " 结果 " << gpu_result[i] << " (期望 " << cpu_result[i] << ")\n";
            success = false;
            break;
        }
    }
    if (success) {
        cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result[0] << " (期望 " << cpu_result[0] << ")\n";
    }
    return success;
}

// 4. GpuTimingResult 结构体（AI 生成）

struct GpuTimingResult {
    float h2d_ms;      // Host to Device 传输时间
    float kernel_ms;   // Kernel 执行时间（多次平均）
    float d2h_ms;      // Device to Host 传输时间
    float total_ms;    // 总时间
};

// 5. GPU 封装函数（部分手写）

// 查询 Occupancy 并执行 Kernel (GPU，手写)
template<typename KernelFunc, int BLOCK_SIZE, int ITEMS_PER_THREAD>
GpuTimingResult query_and_execute_gpu(CRMatrix h_input, RMatrix h_output, CInt n, CInt iterations, KernelFunc kernel, const string& desc, bool is_shared) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    CSize size_bytes = n * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_bytes));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    // === Occupancy 查询核心逻辑 ===
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    
    int num_blocks_per_sm;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm,
        kernel,
        BLOCK_SIZE, 
        0 // dynamic shared memory
    ));
    
    int active_threads_per_sm = num_blocks_per_sm * BLOCK_SIZE;
    float occupancy = (float)active_threads_per_sm / props.maxThreadsPerMultiProcessor;
    
    cout << "  >>> [" << desc << "] 理论信息 <<<\n";
    cout << "  Block配置        : < " << BLOCK_SIZE << " 线程, " << ITEMS_PER_THREAD << " 数据/线程 >\n";
    cout << "  活跃 Block 数量  : " << num_blocks_per_sm << " / SM\n";
    cout << "  活跃 Thread 数量 : " << active_threads_per_sm << " / SM (最大 " << props.maxThreadsPerMultiProcessor << ")\n";
    cout << "  理论 Occupancy   : " << fixed << setprecision(2) << occupancy * 100.0f << " %\n";
    
    const dim3 block(BLOCK_SIZE);
    // 可变负载，此时每个 Thread 处理 ITEMS_PER_THREAD 个元素
    int threads_needed = cdiv(n, ITEMS_PER_THREAD);
    const dim3 grid(cdiv(threads_needed, BLOCK_SIZE)); 
    
    // Kernel 预热
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
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

// 6. main() 函数（部分手写，部分AI 生成）

int main() {
    CInt n = 10000000; // 1000万元素
    CInt iterations = 100;

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      Occupancy 分析基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    Matrix h_input(n, 1.5f);
    Matrix h_cpu_output(n, 0.0f);
    Matrix h_gpu_out_high(n, 0.0f);
    Matrix h_gpu_out_mid(n, 0.0f);
    Matrix h_gpu_out_low(n, 0.0f);
    Matrix h_gpu_out_shared(n, 0.0f);
    Matrix h_gpu_out_bounds(n, 0.0f);

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    base_cpu(h_input, h_cpu_output, n);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1: 高 Occupancy，低 ILP (每个线程 1 个元素)
    cout << "--- GPU 结构 1: 追求满 Occupancy (<256, 1>) ---\n";
    auto kernel_high = configurable_kernel<256, 1>;
    GpuTimingResult res_high = query_and_execute_gpu<decltype(kernel_high), 256, 1>(h_input, h_gpu_out_high, n, iterations, kernel_high, "High-Occ, Low-ILP", false);
    cout << "Kernel 执行时间：" << setw(8) << res_high.kernel_ms << " ms\n\n";

    // GPU 版本 2: 中 Occupancy，高 ILP (每个线程 4 个元素)
    cout << "--- GPU 结构 2: 中等 Occupancy + ILP (<256, 4>) ---\n";
    auto kernel_mid = configurable_kernel<256, 4>;
    GpuTimingResult res_mid = query_and_execute_gpu<decltype(kernel_mid), 256, 4>(h_input, h_gpu_out_mid, n, iterations, kernel_mid, "Mid-Occ, High-ILP", false);
    cout << "Kernel 执行时间：" << setw(8) << res_mid.kernel_ms << " ms\n\n";

    // GPU 版本 3: 低 Occupancy，极高 ILP (每个线程 16 个元素)
    cout << "--- GPU 结构 3: 低 Occupancy + 终极 ILP (<64, 16>) ---\n";
    auto kernel_low = configurable_kernel<64, 16>;
    GpuTimingResult res_low = query_and_execute_gpu<decltype(kernel_low), 64, 16>(h_input, h_gpu_out_low, n, iterations, kernel_low, "Low-Occ, Max-ILP", false);
    cout << "Kernel 执行时间：" << setw(8) << res_low.kernel_ms << " ms\n\n";
    
    // GPU 版本 4: Shared Memory 挤占 Occupancy
    cout << "--- GPU 结构 4: Shared Memory 挤占测试 (<256, 1> + 32KB Shared) ---\n";
    const int SHARED_SIZE = 32768; // 32 KB per block
    auto kernel_shared = shared_memory_kernel<256, SHARED_SIZE>;
    GpuTimingResult res_shared = query_and_execute_gpu<decltype(kernel_shared), 256, 1>(h_input, h_gpu_out_shared, n, iterations, kernel_shared, "32KB Shared Occ", true);
    cout << "Kernel 执行时间：" << setw(8) << res_shared.kernel_ms << " ms\n\n";

    // GPU 版本 5: Launch Bounds 限制寄存器使用
    cout << "--- GPU 结构 5: __launch_bounds__ (<256, 1>) ---\n";
    auto kernel_bounds = register_limited_kernel<256>;
    GpuTimingResult res_bounds = query_and_execute_gpu<decltype(kernel_bounds), 256, 1>(h_input, h_gpu_out_bounds, n, iterations, kernel_bounds, "Launch Bounds", false);
    cout << "Kernel 执行时间：" << setw(8) << res_bounds.kernel_ms << " ms\n\n";

    // 性能分析
    cout << "--- 性能总结 (读 + 写带宽) ---\n";
    double bytes = n * FSIZE * 2; // Read + Write
    cout << "配置 1 (满 Occupancy)   有效带宽：" << setprecision(2) << setw(8) << (bytes / 1e9) / (res_high.kernel_ms / 1000.0) << " GB/s\n";
    cout << "配置 2 (ILP 均衡)       有效带宽：" << setprecision(2) << setw(8) << (bytes / 1e9) / (res_mid.kernel_ms / 1000.0) << " GB/s\n";
    cout << "配置 3 (低 Occupancy)   有效带宽：" << setprecision(2) << setw(8) << (bytes / 1e9) / (res_low.kernel_ms / 1000.0) << " GB/s\n";
    cout << "配置 4 (被挤占的 Occ)   有效带宽：" << setprecision(2) << setw(8) << (bytes / 1e9) / (res_shared.kernel_ms / 1000.0) << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";
    
    cout << "核心结论：Occupancy (占用率) 本质上是为了隐藏延迟。\n";
    cout << "当寄存器足够、通过使用 ILP (指令级并行) 也能极好地隐藏延迟时，即使 Occupancy 仅仅只有 25% 甚至 12%，其实际物理带宽吞吐依然能够逼近硬件极限，甚至超越高内存开销强制满 Occupancy 的情况。\n\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool p1 = verify_results(h_gpu_out_high, h_cpu_output, n, "高 Occupancy\t");
    bool p2 = verify_results(h_gpu_out_mid, h_cpu_output, n, "中等 Occupancy\t");
    bool p3 = verify_results(h_gpu_out_low, h_cpu_output, n, "低 Occupancy\t");
    bool p4 = verify_results(h_gpu_out_shared, h_cpu_output, n, "Shared限制\t");
    bool p5 = verify_results(h_gpu_out_bounds, h_cpu_output, n, "Bounds限制\t");

    if (p1 && p2 && p3 && p4 && p5) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
