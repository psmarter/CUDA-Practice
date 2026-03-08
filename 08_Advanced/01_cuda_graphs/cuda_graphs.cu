// CUDA Graphs - 降低 Kernel 发射开销
#include <code_abbreviation.h>

// 向量加法（GPU kernel，手写）
__global__ void add_kernel(CPFloat A, CPFloat B, PFloat C, CInt n) {
    CInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        C[tid] = A[tid] + B[tid];
    }
}

// 向量乘法（GPU kernel，手写）
__global__ void mul_kernel(CPFloat A, CPFloat B, PFloat C, CInt n) {
    CInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        C[tid] = A[tid] * B[tid];
    }
}

// 复合流水算子 CPU 参考实现（CPU，手写）
void graph_pipeline_cpu(CPFloat A, CPFloat B, CPFloat D, CPFloat F, PFloat G, CInt n) {
    for (int i = 0; i < n; ++i) {
        // Step 1: C = A + B
        float c_val = A[i] + B[i];
        // Step 2: E = C * D
        float e_val = c_val * D[i];
        // Step 3: G = E + F
        G[i] = e_val + F[i];
    }
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = EPSILON) {
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配\n";
        return false;
    }

    int error_count = 0;
    float max_diff = 0.0f;
    int max_diff_idx = 0;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float gpu_v = gpu_result[i];
        float cpu_v = cpu_result[i];
        float diff = abs(gpu_v - cpu_v);
        
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = static_cast<int>(i);
        }
        
        if (diff > epsilon && (diff / (abs(cpu_v) + 1e-5f)) > 1e-3f) {
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

// 传统 Stream 发射模式 GPU 封装（GPU，手写）
template<typename KernelFuncAdd, typename KernelFuncMul>
GpuTimingResult standard_stream_gpu(CRMatrix h_A, CRMatrix h_B, CRMatrix h_D, CRMatrix h_F, RMatrix h_G,
                                    CInt n, CInt iterations, KernelFuncAdd add_func, KernelFuncMul mul_func) {
    PFloat d_A = nullptr, d_B = nullptr, d_C = nullptr, d_D = nullptr, d_E = nullptr, d_F = nullptr, d_G = nullptr;
    CSize size_io = n * FSIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_D, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_E, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_F, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_G, size_io));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D, h_D.data(), size_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F, h_F.data(), size_io, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(BLOCK_SIZE_1D);
    const dim3 grid(cdiv(n, BLOCK_SIZE_1D));

    // 预热
    add_func<<<grid, block>>>(d_A, d_B, d_C, n);
    mul_func<<<grid, block>>>(d_C, d_D, d_E, n);
    add_func<<<grid, block>>>(d_E, d_F, d_G, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        add_func<<<grid, block>>>(d_A, d_B, d_C, n);
        mul_func<<<grid, block>>>(d_C, d_D, d_E, n);
        add_func<<<grid, block>>>(d_E, d_F, d_G, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_G.data(), d_G, size_io, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D)); CUDA_CHECK(cudaFree(d_E)); CUDA_CHECK(cudaFree(d_F)); CUDA_CHECK(cudaFree(d_G));

    return result;
}

// CUDA Graphs 捕获发射模式 GPU 封装（GPU，手写）
template<typename KernelFuncAdd, typename KernelFuncMul>
GpuTimingResult cuda_graph_gpu(CRMatrix h_A, CRMatrix h_B, CRMatrix h_D, CRMatrix h_F, RMatrix h_G,
                               CInt n, CInt iterations, KernelFuncAdd add_func, KernelFuncMul mul_func) {
    PFloat d_A = nullptr, d_B = nullptr, d_C = nullptr, d_D = nullptr, d_E = nullptr, d_F = nullptr, d_G = nullptr;
    CSize size_io = n * FSIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_D, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_E, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_F, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_G, size_io));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D, h_D.data(), size_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F, h_F.data(), size_io, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(BLOCK_SIZE_1D);
    const dim3 grid(cdiv(n, BLOCK_SIZE_1D));

    // 使用 Graph Capture 录制流水线
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // 开启流捕获
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    add_func<<<grid, block, 0, stream>>>(d_A, d_B, d_C, n);
    mul_func<<<grid, block, 0, stream>>>(d_C, d_D, d_E, n);
    add_func<<<grid, block, 0, stream>>>(d_E, d_F, d_G, n);

    // 结束图捕获
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // 实例化这个执行图
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    // 预热一把可执行图，代替传统的多次启动预热
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {

        // 单个 API 调用发射整个复杂流水图，彻底省去 CPU 调用瓶颈
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
    }
    
    // GraphLaunch 默认也是异步的，所以需要 Sync 确保时间准确
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_G.data(), d_G, size_io, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    // 清理图及相关资源
    CUDA_CHECK(cudaGraphExecDestroy(instance));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D)); CUDA_CHECK(cudaFree(d_E)); CUDA_CHECK(cudaFree(d_F)); CUDA_CHECK(cudaFree(d_G));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    // 故意使用较小的数据量来放大 Launch Overhead (发射开销) 在时间中的比重
    CInt n = 100000;
    CInt iterations = 1000; // 较多的循环次数以测试发射流的拥堵状况

    CSize size_io = n * FSIZE;
    // 总共4个输入，3个中间态/结果，共 7 个数组
    const double total_mb = (size_io * 7) / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      CUDA Graphs 编排性能基准测试\n";
    cout << "========================================\n";
    cout << "流水线步骤：(A + B) * D + F = G\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "涉及显存：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE_1D << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_A(n), h_B(n), h_D(n), h_F(n);
    Matrix h_G_stream(n), h_G_graph(n), h_G_cpu(n);

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_B[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_D[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_F[i] = static_cast<float>(rand() % 100) / 100.0f;
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    graph_pipeline_cpu(h_A.data(), h_B.data(), h_D.data(), h_F.data(), h_G_cpu.data(), n);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1：传统 Stream 发射
    cout << "--- GPU 版本 1: 传统多 Kernel 发射 ---\n";
    GpuTimingResult result_stream = standard_stream_gpu(h_A, h_B, h_D, h_F, h_G_stream, n, iterations, add_kernel, mul_kernel);
    cout << "H2D 传输时间：   " << setw(8) << result_stream.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result_stream.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result_stream.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result_stream.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2：CUDA Graphs (Stream Capture) 发射
    cout << "--- GPU 版本 2: CUDA Graph Launch ---\n";
    GpuTimingResult result_graph = cuda_graph_gpu(h_A, h_B, h_D, h_F, h_G_graph, n, iterations, add_kernel, mul_kernel);
    cout << "H2D 传输时间：   " << setw(8) << result_graph.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result_graph.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result_graph.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result_graph.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_kernel = cpu_time_ms / result_graph.kernel_ms;
    double speedup_total = cpu_time_ms / result_graph.total_ms;
    cout << "CPU vs GPU (Graph) Kernel 加速比：" << setprecision(2) << speedup_kernel << "x\n";
    cout << "CPU vs GPU (Graph) 总时间加速比：" << speedup_total << "x\n";

    double bytes = size_io * 9;
    double gpu_bandwidth = (bytes / 1e9) / (result_graph.kernel_ms / 1000.0);
    cout << "GPU 有效带宽：" << setprecision(2) << gpu_bandwidth << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    cout << "--- Kernel 性能对比 ---\n";
    cout << "Multi-Stream Launch: " << setw(8) << setprecision(4) << result_stream.kernel_ms << " ms (基准)\n";
    cout << "CUDA Graph Launch:   " << setw(8) << result_graph.kernel_ms << " ms ("
         << setprecision(2) << result_stream.kernel_ms / result_graph.kernel_ms << "x CPU 发射开销减免)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_G_stream, h_G_cpu, "传统流发射 Kernel 流水");
    bool pass2 = verify_results(h_G_graph, h_G_cpu, "CUDA Graphs 发射");

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
