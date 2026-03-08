// Kernel Fusion - 多算子融合减少访存
#include <code_abbreviation.h>
#include <string>
#include <random>


// 非融合版本（多次访存）

// 分离的 Add kernel (GPU kernel，手写)
__global__ void add_kernel(CPFloat a, CPFloat b, PFloat output, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

// 分离的 ReLU kernel (GPU kernel，手写)
__global__ void relu_kernel(CPFloat input, PFloat output, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

// 分离的 Scale kernel (GPU kernel，手写)
__global__ void scale_kernel(CPFloat input, PFloat output, CFloat scale, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale;
    }
}

// 融合版本（单次访存）

// 融合 Add + ReLU (GPU kernel，手写)
__global__ void fused_add_relu(CPFloat a, CPFloat b, PFloat output, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 在寄存器中保留结果，不写回主存中间变量
        float sum = a[idx] + b[idx];
        output[idx] = fmaxf(sum, 0.0f);
    }
}

// 融合 Add + ReLU + Scale (GPU kernel，手写)
__global__ void fused_add_relu_scale(CPFloat a, CPFloat b, PFloat output, 
                                      CFloat scale, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 完成三次操作，仅消耗一次主存 Read (A+B) 和一次主存 Write (Output)
        float sum = a[idx] + b[idx];
        float activated = fmaxf(sum, 0.0f);
        output[idx] = activated * scale;
    }
}

// 高级推断融合（矩阵运算）

// GELU 激活函数
__device__ float gelu(float x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float c = 0.7978845608f;  // sqrt(2/π)
    const float k = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x * x * x)));
}

// 融合 Linear + GELU（常用于 Transformer FFN）(GPU kernel，手写)
__global__ void fused_linear_gelu(CPFloat input, CPFloat weight, CPFloat bias,
                                   PFloat output, CInt batch, CInt in_features, 
                                   CInt out_features) {
    // blockIdx.x 负责 batch，threadIdx.x 负责 out_features
    int b = blockIdx.x;
    int out_f = threadIdx.x;
    
    if (b < batch && out_f < out_features) {
        float sum = bias[out_f];
        for (int in_f = 0; in_f < in_features; ++in_f) {
            sum += input[b * in_features + in_f] * weight[out_f * in_features + in_f];
        }
        // 融合 GELU 激活
        output[b * out_features + out_f] = gelu(sum);
    }
}


// CPU 复合计算参考基准 (CPU，手写)
void add_relu_scale_cpu(CRMatrix a, CRMatrix b, RMatrix output, float scale, CInt n) {
    for (int i = 0; i < n; ++i) {
        float sum = a[i] + b[i];
        float activated = std::max(sum, 0.0f);
        output[i] = activated * scale;
    }
}


// 验证结果（AI 生成）
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


// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      // Host to Device 传输时间
    float kernel_ms;   // Kernel 执行时间（多次平均）
    float d2h_ms;      // Device to Host 传输时间
    float total_ms;    // 总时间
};


// 未融合版本的顺序执行封装 (GPU，手写)
GpuTimingResult unfused_ops_gpu(CRMatrix h_a, CRMatrix h_b, RMatrix h_output, CFloat scale, CInt n, CInt iterations) {
    PFloat d_a = nullptr, d_b = nullptr, d_intermediate1 = nullptr, d_intermediate2 = nullptr, d_output = nullptr;
    CSize size_bytes = n * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_a, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_intermediate1, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_intermediate2, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_bytes));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const dim3 block(256);
    const dim3 grid(cdiv(n, 256));
    
    // 预热
    add_kernel<<<grid, block>>>(d_a, d_b, d_intermediate1, n);
    relu_kernel<<<grid, block>>>(d_intermediate1, d_intermediate2, n);
    scale_kernel<<<grid, block>>>(d_intermediate2, d_output, scale, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        add_kernel<<<grid, block>>>(d_a, d_b, d_intermediate1, n);
        relu_kernel<<<grid, block>>>(d_intermediate1, d_intermediate2, n);
        scale_kernel<<<grid, block>>>(d_intermediate2, d_output, scale, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_intermediate1));
    CUDA_CHECK(cudaFree(d_intermediate2));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

// 融合版本的单次执行封装 (GPU，手写)
GpuTimingResult fused_ops_gpu(CRMatrix h_a, CRMatrix h_b, RMatrix h_output, CFloat scale, CInt n, CInt iterations) {
    PFloat d_a = nullptr, d_b = nullptr, d_output = nullptr;
    CSize size_bytes = n * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_a, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_bytes));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const dim3 block(256);
    const dim3 grid(cdiv(n, 256));
    
    // 预热
    fused_add_relu_scale<<<grid, block>>>(d_a, d_b, d_output, scale, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        fused_add_relu_scale<<<grid, block>>>(d_a, d_b, d_output, scale, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}


// 主函数（部分手写，部分AI 生成）
int main() {
    // 采用足够大的数据规模来彰显访存瓶颈（128M Elements = 512MB per tensor）
    CInt n = 1 << 27; 
    CInt iterations = 50;
    CFloat test_scale = 0.5f;

    CSize size_bytes = n * FSIZE;
    const double total_mb_single = size_bytes / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      算子融合 (Kernel Fusion) 访存优化基准测试\n";
    cout << "========================================\n";
    cout << "特征数组大小：" << n << " 元素\n";
    cout << "单张量大小  ：" << fixed << setprecision(2) << total_mb_single << " MB\n";
    cout << "测试算子链路：Add(A, B) -> ReLU -> Scale\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配并初始化主机内存
    Matrix h_a(n);
    Matrix h_b(n);
    Matrix h_cpu_output(n, 0.0f);
    Matrix h_gpu_unfused_out(n, 0.0f);
    Matrix h_gpu_fused_out(n, 0.0f);
    
    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int i = 0; i < n; ++i) {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }

    // CPU 计算作为验证真值
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    add_relu_scale_cpu(h_a, h_b, h_cpu_output, test_scale, n);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1：非融合（多次访存机制）
    cout << "--- GPU 版本 1: 非融合序列 (Unfused Series) ---\n";
    GpuTimingResult res_unfused = unfused_ops_gpu(h_a, h_b, h_gpu_unfused_out, test_scale, n, iterations);
    cout << "H2D 传输时间：   " << setw(8) << res_unfused.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_unfused.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_unfused.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_unfused.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2：融合计算（单次访存机制）
    cout << "--- GPU 版本 2: 算子融合 (Fused Kernel) ---\n";
    GpuTimingResult res_fused = fused_ops_gpu(h_a, h_b, h_gpu_fused_out, test_scale, n, iterations);
    cout << "H2D 传输时间：   " << setw(8) << res_fused.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_fused.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_fused.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_fused.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 理论访存计算与性能分析 ---\n";
    // 理论访存推算：
    // 非融合：Add(读A, 读B, 写T1) + ReLU(读T1, 写T2) + Scale(读T2, 写O) = 读进 4次，写出 3次
    // 融合：  Fused(读A, 读B, 写O) = 读进 2次，写出 1次
    
    double bytes_unfused = 7.0 * n * FSIZE; 
    double bytes_fused   = 3.0 * n * FSIZE; 
    
    double bw_unfused = (bytes_unfused / 1e9) / (res_unfused.kernel_ms / 1000.0);
    double bw_fused   = (bytes_fused   / 1e9) / (res_fused.kernel_ms   / 1000.0);

    // 有效带宽（指数学上必需的有效传输：读 A, 读 B, 写 O）
    double bw_effective_unfused = (bytes_fused / 1e9) / (res_unfused.kernel_ms / 1000.0);
    double bw_effective_fused   = (bytes_fused / 1e9) / (res_fused.kernel_ms   / 1000.0);

    cout << "非融合版本物理带宽：" << setw(8) << setprecision(2) << bw_unfused << " GB/s\n";
    cout << "已融合版本物理带宽：" << setw(8) << setprecision(2) << bw_fused << " GB/s\n";
    cout << "--------------------------------\n";
    cout << "非融合版本有效带宽：" << setw(8) << setprecision(2) << bw_effective_unfused << " GB/s (受制于无效的中间访存)\n";
    cout << "算子融合后有效带宽：" << setw(8) << setprecision(2) << bw_effective_fused << " GB/s (接近硬件物理极限)\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    cout << "--- Kernel 性能加速比 ---\n";
    cout << "非融合序列 耗时  : " << setw(8) << setprecision(4) << res_unfused.kernel_ms << " ms\n";
    cout << "算子融合版 耗时  : " << setw(8) << setprecision(4) << res_fused.kernel_ms << " ms\n";
    cout << ">> 融合加速比   : " << setprecision(2) << res_unfused.kernel_ms / res_fused.kernel_ms << "x\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_gpu_unfused_out, h_cpu_output, n, "Unfused Kernels");
    bool pass2 = verify_results(h_gpu_fused_out, h_cpu_output, n, "Fused Kernels");

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }
    cout << "\n========================================\n";

    return 0;
}
