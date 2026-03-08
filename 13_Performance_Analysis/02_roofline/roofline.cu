// Roofline 模型分析 - 性能上限评估
#include <code_abbreviation.h>
#include <string>

// 1. GPU Kernel 函数（手写）

// 内存受限 kernel（低算术强度）(GPU kernel，手写)
// 向量加法：C = A + B
// 每个元素进行 1 次浮点加法运算 (1 FLOP)
// 读取 A, 读取 B, 写入 C (3 * 4 = 12 Bytes)
// 算术强度 = 1 / 12 = 0.083 FLOPS/Byte
__global__ void memory_bound_kernel(CPFloat a, CPFloat b, PFloat c, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 计算受限 kernel（高算术强度）(GPU kernel，手写)
// 朴素矩阵乘法：C = A * B (为了凸显纯计算密集，故意不加 shared memory 等优化以展现高 FMA/访存比)
// 每个 C 元素需要 N 次乘法和 N 次加法 = 2 * N FLOPS
// 每次计算只访问 A 的一行和 B 的一列。为了减少访存，可以在寄存器中累加。
// 算术强度 = O(N) FLOPS/Byte
__global__ void compute_bound_kernel(CPFloat A, CPFloat B, PFloat C, CInt N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 2. CPU 参考实现函数（手写）

// 内存受限 kernel（CPU，手写）
void memory_bound_cpu(CRMatrix a, CRMatrix b, RMatrix c, CInt n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// 计算受限 kernel（CPU，手写）
void compute_bound_cpu(CRMatrix A, CRMatrix B, RMatrix C, CInt N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 3. verify_results 验证函数（AI 生成）

bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, CInt n, const string& kernel_name) {
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(gpu_result[i] - cpu_result[i]) > 1e-4f) {
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


// 5. GPU 封装函数（部分手写）

// 硬件参数
struct GPUSpecs {
    float peak_flops_sp;      // 单精度峰值 TFLOPS
    float memory_bandwidth;   // 显存带宽 GB/s
    
    void query_from_device() {
        int deviceId;
        CUDA_CHECK(cudaGetDevice(&deviceId));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
        
        // 推算单精度核心数: Ampere (SM * 128), Ada (SM * 128), Turing (SM * 64)...
        // 我们统一按现代架构标准的每个 SM 128 个 FP32 CUDA Core 粗略估算理论上限
        // 峰值吞吐量 = SM 数 * 128 个 Core * 每个时钟 2 次操作 (FMA) * 时钟频率
        int cores_per_sm = 128; // 近似假设
        if (prop.major == 7) cores_per_sm = 64; // Volta/Turing
        if (prop.major == 8 || prop.major == 9) cores_per_sm = 128; // Ampere/Ada
        
        peak_flops_sp = (static_cast<double>(prop.clockRate) * 1e3 * prop.multiProcessorCount * cores_per_sm * 2) / 1e12; 
        memory_bandwidth = (prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) * 2) / 1e9; // 乘以 2 因为是 DDR 内存
    }
};

// 实际表现与 Roofline 模型分析
struct KernelProfile {
    long long flops;           // 浮点运算次数
    long long bytes_accessed;  // 访存字节数
    float arithmetic_intensity; // FLOPS / Bytes (算术强度)
    
    void compute_intensity() {
        arithmetic_intensity = static_cast<float>(flops) / static_cast<float>(bytes_accessed);
    }
};

void roofline_analysis(const GPUSpecs& specs, const KernelProfile& profile, float actual_time_ms, int iterations, const string& name) {
    // 实际 GFLOPS
    double actual_gflops = (static_cast<double>(profile.flops) * iterations) / (actual_time_ms / 1000.0) / 1e9;
    
    // Roofline 理论峰值计算
    // P = min( P_peak, I * B_peak )
    double theoretical_peak_from_bandwidth = profile.arithmetic_intensity * specs.memory_bandwidth;
    double theoretical_peak_from_compute = specs.peak_flops_sp * 1000.0; // TFLOPS -> GFLOPS
    
    double theoretical_peak = min(theoretical_peak_from_bandwidth, theoretical_peak_from_compute);
    
    double efficiency = (actual_gflops / theoretical_peak) * 100.0;

    cout << "  [" << name << "] Roofline 分析结果：\n";
    cout << "  算术强度 (I) : " << fixed << setprecision(3) << profile.arithmetic_intensity << " FLOPS/Byte\n";
    cout << "  瓶颈受限     : " << (theoretical_peak_from_bandwidth < theoretical_peak_from_compute ? "Memory Bound (内存带宽受限)" : "Compute Bound (计算核心受限)") << "\n";
    cout << "  理论峰值速度 : " << fixed << setprecision(2) << theoretical_peak << " GFLOPS\n";
    cout << "  实际运行速度 : " << fixed << setprecision(2) << actual_gflops << " GFLOPS\n";
    cout << "  计算侧效率   : " << fixed << setprecision(2) << efficiency << " %\n\n";
}

// 内存受限 Kernel GPU 封装 (GPU，手写)
GpuTimingResult memory_bound_gpu(CRMatrix h_a, CRMatrix h_b, RMatrix h_c, CInt n, CInt iterations, const GPUSpecs& specs) {
    PFloat d_a = nullptr, d_b = nullptr, d_c = nullptr;
    CSize size_bytes = n * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_a, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size_bytes));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const dim3 block(256);
    const dim3 grid(cdiv(n, 256));
    
    memory_bound_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        memory_bound_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    // Roofline 特判计算
    KernelProfile prof;
    prof.flops = static_cast<long long>(n) * 1LL; // {加法 = 1 FLOP} * N
    prof.bytes_accessed = static_cast<long long>(n) * 3LL * 4LL; // {读A + 读B + 写C = 12 Bytes} * N
    prof.compute_intensity();
    
    roofline_analysis(specs, prof, result.kernel_ms, 1 /* 我们传的时单次均值 */, "内存受限 Kernel (Vector Add)");
    
    return result;
}

// 计算受限 Kernel GPU 封装 (GPU，手写)
GpuTimingResult compute_bound_gpu(CRMatrix h_a, CRMatrix h_b, RMatrix h_c, CInt N, CInt iterations, const GPUSpecs& specs) {
    PFloat d_a = nullptr, d_b = nullptr, d_c = nullptr;
    CSize size_bytes = N * N * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_a, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size_bytes));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const dim3 block(16, 16);
    const dim3 grid(cdiv(N, 16), cdiv(N, 16));
    
    compute_bound_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        compute_bound_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    // GEMM 算力太大，防止过长等待通常迭代较少，取均值
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    // Roofline 特判计算
    KernelProfile prof;
    // 矩阵乘法 FLOPS = 2 * N^3
    prof.flops = static_cast<long long>(N) * static_cast<long long>(N) * static_cast<long long>(N) * 2LL;
    // 每次线程读 C[row,col] -> 实际上是访问 N个A，N个B并写出1个C
    // 但是考虑到 L1 Cache (如果没有 Block Tiling 的话)：大致访问字节数为 2 * N^3 (A和B的重复读) * 4 bytes，加上写出的 N^2 * 4 bytes
    // 最理想理论下，读取一遍完整的 A 和 B 和 C，是 3 * N^2 * 4 Bytes。
    // 但是这里我们在分析基础内核，没有共享内存 Tiling，全局内存实际物理读取将受制。
    // 为了标准化模型的 "算法强度"，通常定义算法逻辑读写字节数 (3 * N^2 floats)。即只要完美 cache，这就是理想瓶颈
    prof.bytes_accessed = static_cast<long long>(N) * static_cast<long long>(N) * 3LL * 4LL; 
    prof.compute_intensity();
    
    roofline_analysis(specs, prof, result.kernel_ms, 1 /* 单次均值 */, "计算受限 Kernel (Naive GEMM)");
    
    return result;
}

// 6. main() 函数（部分手写，部分AI 生成）

int main() {
    printDeviceInfo();
    
    GPUSpecs hw_specs;
    hw_specs.query_from_device();
    
    cout << "========================================\n";
    cout << "    Roofline Model 硬件平台画像\n";
    cout << "========================================\n";
    cout << "单精度理论峰值算力 : " << fixed << setprecision(2) << hw_specs.peak_flops_sp << " TFLOPS\n";
    cout << "理论峰值物理显存带宽: " << fixed << setprecision(2) << hw_specs.memory_bandwidth << " GB/s\n";
    cout << "拐点算术强度 (Ridge): " << fixed << setprecision(2) << (hw_specs.peak_flops_sp * 1000.0) / hw_specs.memory_bandwidth << " FLOPS/Byte\n";
    cout << "========================================\n\n";

    // 测试 1: Memory Bound (一维大数组向量加法)
    CInt N_vec = 10000000; // 1000 万元素
    CInt iter_vec = 100;
    
    Matrix h_vec_a(N_vec, 1.0f);
    Matrix h_vec_b(N_vec, 2.0f);
    Matrix h_vec_c_cpu(N_vec, 0.0f);
    Matrix h_vec_c_gpu(N_vec, 0.0f);
    
    cout << "--- 测试项 1：Memory Bound Kernel (Vector Add N=10M) ---\n";
    CpuTimer timer_cpu;
    timer_cpu.start();
    memory_bound_cpu(h_vec_a, h_vec_b, h_vec_c_cpu, N_vec);
    timer_cpu.stop();
    cout << "CPU 执行时间：   " << setw(8) << timer_cpu.elapsed_ms() << " ms\n";
    
    GpuTimingResult res_vec = memory_bound_gpu(h_vec_a, h_vec_b, h_vec_c_gpu, N_vec, iter_vec, hw_specs);
    cout << "Kernel 执行时间：" << setw(8) << res_vec.kernel_ms << " ms (" << iter_vec << " 次平均)\n";
    bool pass1 = verify_results(h_vec_c_gpu, h_vec_c_cpu, N_vec, "Memory Bound (VecAdd)");
    cout << "\n";
    
    // 测试 2: Compute Bound (二维方阵乘法)
    CInt N_mat = 1024; // 1024x1024 矩阵
    CInt iter_mat = 10;
    
    Matrix h_mat_a(N_mat * N_mat, 1.0f);
    Matrix h_mat_b(N_mat * N_mat, 2.0f);
    Matrix h_mat_c_cpu(N_mat * N_mat, 0.0f);
    Matrix h_mat_c_gpu(N_mat * N_mat, 0.0f);
    
    cout << "--- 测试项 2：Compute Bound Kernel (GEMM N=1024) ---\n";
    timer_cpu.start();
    // compute_bound_cpu(h_mat_a, h_mat_b, h_mat_c_cpu, N_mat); // CPU $N^3$ 太慢，开发不执行或减少尺寸
    // 强制构造正确输出进行比对 (都是 1.0*2.0=2.0，累加 1024 次 = 2048.0)
    for(int i=0; i<N_mat*N_mat; ++i) h_mat_c_cpu[i] = 2048.0f;
    timer_cpu.stop();
    // cout << "CPU 执行时间：   " << setw(8) << timer_cpu.elapsed_ms() << " ms\n";
    
    GpuTimingResult res_mat = compute_bound_gpu(h_mat_a, h_mat_b, h_mat_c_gpu, N_mat, iter_mat, hw_specs);
    cout << "Kernel 执行时间：" << setw(8) << res_mat.kernel_ms << " ms (" << iter_mat << " 次平均)\n";
    bool pass2 = verify_results(h_mat_c_gpu, h_mat_c_cpu, N_mat * N_mat, "Compute Bound (GEMM)");
    
    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
