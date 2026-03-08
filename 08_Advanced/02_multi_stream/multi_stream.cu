// 可计算隐藏访存延迟的多流并发 (Multi-Stream) 基准测试
#include <code_abbreviation.h>

// 计算密集型任务：A * sin(B) + B * cos(A)（GPU kernel，手写）
__global__ void compute_kernel(CPFloat A, CPFloat B, PFloat C, CInt n) {
    CInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float a = A[tid];
        float b = B[tid];

        // 刻意使用复杂的数学运算来增加 Kernel 的执行时间，使得计算时间和传输时间能够在一个数量级，从而更容易展示重叠（Overlap）效果
        C[tid] = a * sinf(b) + b * cosf(a);
    }
}

// CPU 参考实现（CPU，手写）
void compute_cpu(CPFloat A, CPFloat B, PFloat C, CInt n) {
    for (int i = 0; i < n; ++i) {
        float a = A[i];
        float b = B[i];
        C[i] = a * sin(b) + b * cos(a);
    }
}

// 验证结果（AI 生成）
bool verify_results(CPFloat gpu_result, CPFloat cpu_result, const string& kernel_name, CInt n, CFloat epsilon = EPSILON) {
    int error_count = 0;
    float max_diff = 0.0f;
    int max_diff_idx = 0;

    for (int i = 0; i < n; ++i) {
        float gpu_v = gpu_result[i];
        float cpu_v = cpu_result[i];
        float diff = abs(gpu_v - cpu_v);
        
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        
        if (diff > epsilon && (diff / (abs(cpu_v) + 1e-5f)) > 1e-3f) {
            error_count++;
        }
    }

    if (error_count > 0) {
        cout << "✗ " << kernel_name << " FAILED: " << error_count << " 个元素超出误差阈值\n";
        cout << "  首个错误异常差异位于索引 " << max_diff_idx
             << "：GPU=" << gpu_result[max_diff_idx]
             << ", CPU=" << cpu_result[max_diff_idx]
             << ", 差异=" << max_diff << "\n";
        return false;
    }

    cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result[0] 
         << " (期望 " << cpu_result[0] << ")\n";
    return true;
}

// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      
    float kernel_ms;   
    float d2h_ms;      
    float total_ms;    
};

// 单流 (串行执行全量任务) 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult single_stream_gpu(CPFloat h_A, CPFloat h_B, PFloat h_C, CInt n, CInt iterations, KernelFunc kernel) {
    PFloat d_A = nullptr;
    PFloat d_B = nullptr;
    PFloat d_C = nullptr;
    
    CSize size_io = n * FSIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_io));

    CudaTimer timerTotal;
    GpuTimingResult result{};
    
    // 为了公平对比 Pipeline 的吞吐率，我们将 H2D -> Kernel -> D2H 视作一个完整的周期
    timerTotal.start();
    for (int i = 0; i < iterations; ++i) {
        // 同步拷贝和执行，使用默认流 (Stream 0)
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_io, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_io, cudaMemcpyHostToDevice));
        
        const dim3 block(BLOCK_SIZE_1D);
        const dim3 grid(cdiv(n, BLOCK_SIZE_1D));
        
        kernel<<<grid, block>>>(d_A, d_B, d_C, n);
        
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_io, cudaMemcpyDeviceToHost));
    }
    timerTotal.stop();
    CUDA_CHECK_LAST();
    
    // 我们将耗时平摊到 total_ms，其余部分填 0（因测试粒度为 Pipeline）
    result.total_ms = timerTotal.elapsed_ms() / static_cast<float>(iterations);
    result.h2d_ms = 0.0f;
    result.kernel_ms = result.total_ms; // 挂载总耗时以便在 main 中输出
    result.d2h_ms = 0.0f;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

// 多流 (流水线并发) 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult multi_stream_gpu(CPFloat h_A, CPFloat h_B, PFloat h_C, CInt n, CInt iterations, CInt num_streams, KernelFunc kernel) {
    PFloat d_A = nullptr;
    PFloat d_B = nullptr;
    PFloat d_C = nullptr;
    
    CSize size_io = n * FSIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_io));

    // 创建流
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    CInt chunk_size = cdiv(n, num_streams);
    CudaTimer timerTotal;
    GpuTimingResult result{};

    timerTotal.start();
    for (int iter = 0; iter < iterations; ++iter) {

        // 多流并发：切块数据，交错发射 Async 拷贝和 Kernel 以打满 PCIe 和 Compute Pipeline
        for (int i = 0; i < num_streams; ++i) {
            CInt offset = i * chunk_size;
            CInt current_chunk = min(chunk_size, n - offset);
            CSize current_bytes = current_chunk * FSIZE;

            if (current_chunk > 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_A + offset, h_A + offset, current_bytes, cudaMemcpyHostToDevice, streams[i]));
                CUDA_CHECK(cudaMemcpyAsync(d_B + offset, h_B + offset, current_bytes, cudaMemcpyHostToDevice, streams[i]));
                
                const dim3 block(BLOCK_SIZE_1D);
                const dim3 grid(cdiv(current_chunk, BLOCK_SIZE_1D));
                kernel<<<grid, block, 0, streams[i]>>>(d_A + offset, d_B + offset, d_C + offset, current_chunk);
                
                CUDA_CHECK(cudaMemcpyAsync(h_C + offset, d_C + offset, current_bytes, cudaMemcpyDeviceToHost, streams[i]));
            }
        }
        
        // 每次大迭代后需要等待所有流完成，以保证计时准确
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    timerTotal.stop();
    CUDA_CHECK_LAST();

    result.total_ms = timerTotal.elapsed_ms() / static_cast<float>(iterations);
    result.h2d_ms = 0.0f;
    result.kernel_ms = result.total_ms;
    result.d2h_ms = 0.0f;

    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt n = 16 * 1024 * 1024; // 16M 元素
    CInt iterations = 10;
    CInt num_streams = 4;      // 分成 4 个流

    CSize size_io = n * FSIZE;
    const double total_mb = (size_io * 3) / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      多流并发隐藏延迟性能基准测试\n";
    cout << "========================================\n";
    cout << "测试算子：C = A * sin(B) + B * cos(A)\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "并发流数：" << num_streams << " 排队长度\n";
    cout << "Block 大小：" << BLOCK_SIZE_1D << " 线程\n";
    cout << "测试指标：全链路 (H2D -> Compute -> D2H) 周期执行时间\n";
    cout << "执行迭代：" << iterations << " 次\n";
    cout << "\n";

    // 为了使 cudaMemcpyAsync 能真正异步工作，Host 内存必须分配为 Pinned 锁页内存
    // 默认的 std::vector 等 Pageable 内存会强制系统在拷贝时加锁阻塞，从而破坏 Stream Overlap 效应
    PFloat h_A = nullptr, h_B = nullptr, h_C_single = nullptr, h_C_multi = nullptr, h_C_cpu = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_A, size_io));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, size_io));
    CUDA_CHECK(cudaMallocHost((void**)&h_C_single, size_io));
    CUDA_CHECK(cudaMallocHost((void**)&h_C_multi, size_io));
    h_C_cpu = new float[n]; // CPU 参照结果不需要锁页内存

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_B[i] = static_cast<float>(rand() % 100) / 100.0f;
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    compute_cpu(h_A, h_B, h_C_cpu, n);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1：传统单流
    cout << "--- GPU 版本 1: 传统单流 (完全串行) ---\n";
    GpuTimingResult result1 = single_stream_gpu(h_A, h_B, h_C_single, n, iterations, compute_kernel);
    cout << "Pipeline 周期时间：" << setw(8) << result1.total_ms << " ms (" << iterations << " 次平均)\n";
    cout << "\n";

    // GPU 版本 2：多流并发
    cout << "--- GPU 版本 2: 多流 (流水线并发) ---\n";
    GpuTimingResult result2 = multi_stream_gpu(h_A, h_B, h_C_multi, n, iterations, num_streams, compute_kernel);
    cout << "Pipeline 周期时间：" << setw(8) << result2.total_ms << " ms (" << iterations << " 次平均)\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_total = result1.total_ms / result2.total_ms;
    cout << "CPU vs GPU (多流) 总加速比：" << setprecision(2) << (cpu_time_ms / result2.total_ms) << "x\n";
    cout << "单流 vs 多流 并发加速比：   " << speedup_total << "x\n";

    double bytes = size_io * 3;
    double gpu_bandwidth = (bytes / 1e9) / (result2.total_ms / 1000.0);
    cout << "GPU 有效流水线吞吐带宽：" << setprecision(2) << gpu_bandwidth << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // --- Kernel 性能对比 ---
    cout << "--- Kernel 性能对比 ---\n";
    cout << "传统单流: " << setw(8) << setprecision(4) << result1.total_ms << " ms (基准)\n";
    cout << "多流并发:   " << setw(8) << result2.total_ms << " ms ("
         << setprecision(2) << speedup_total << "x 并发开销减免)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_C_single, h_C_cpu, "传统单流执行", n);
    bool pass2 = verify_results(h_C_multi, h_C_cpu, "多流并发执行", n);

    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    // 释放资源
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C_single));
    CUDA_CHECK(cudaFreeHost(h_C_multi));
    delete[] h_C_cpu;

    cout << "\n========================================\n";

    return 0;
}
