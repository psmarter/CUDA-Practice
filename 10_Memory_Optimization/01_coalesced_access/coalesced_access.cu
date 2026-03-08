// 全局内存合并访问 - 内存访问模式优化
#include <code_abbreviation.h>
#include <string>

// 结构体数组 vs 数组结构体（AoS vs SoA）
struct alignas(16) AoS {
    float x, y, z, w;
};

struct SoA {
    float* x;
    float* y;
    float* z;
    float* w;
};

// 合并访问（每个线程访问连续地址）（GPU kernel，手写）
__global__ void coalesced_access(CPFloat input, PFloat output, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // 连续访问，合并
    }
}

// 非合并访问（跨步访问）（GPU kernel，手写）
__global__ void strided_access(CPFloat input, PFloat output, CInt n, CInt stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_idx = idx * stride;  // 跨步访问
    if (actual_idx < n) {
        output[actual_idx] = input[actual_idx] * 2.0f;
    }
}

// AoS 访问（非合并）（GPU kernel，手写）
__global__ void aos_access(const AoS* input, AoS* output, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // AoS方式：相近线程访问结构体的相近成员（跨步访问）
        output[idx].x = input[idx].x * 2.0f;
        output[idx].y = input[idx].y * 2.0f;
        output[idx].z = input[idx].z * 2.0f;
        output[idx].w = input[idx].w * 2.0f;
    }
}

// SoA 访问（合并）（GPU kernel，手写）
__global__ void soa_access(const SoA input, SoA output, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // SoA方式：每个成员都形成了连续的合并访问
        output.x[idx] = input.x[idx] * 2.0f;
        output.y[idx] = input.y[idx] * 2.0f;
        output.z[idx] = input.z[idx] * 2.0f;
        output.w[idx] = input.w[idx] * 2.0f;
    }
}

// 合并访问（CPU，手写）
void coalesced_access_cpu(CRMatrix input, RMatrix output, CInt n) {
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] * 2.0f;
    }
}

// 跨步访问（CPU，手写）
void strided_access_cpu(CRMatrix input, RMatrix output, CInt n, CInt stride) {
    for (int idx = 0; idx * stride < n; ++idx) {
        int actual_idx = idx * stride;
        output[actual_idx] = input[actual_idx] * 2.0f;
    }
}

// AoS 访问（CPU，手写）
void aos_access_cpu(const vector<AoS>& input, vector<AoS>& output, CInt n) {
    for (int i = 0; i < n; ++i) {
        output[i].x = input[i].x * 2.0f;
        output[i].y = input[i].y * 2.0f;
        output[i].z = input[i].z * 2.0f;
        output[i].w = input[i].w * 2.0f;
    }
}

// SoA 访问（CPU，手写）
void soa_access_cpu(CRMatrix in_x, CRMatrix in_y, CRMatrix in_z, CRMatrix in_w,
                    RMatrix out_x, RMatrix out_y, RMatrix out_z, RMatrix out_w, CInt n) {
    for (int i = 0; i < n; ++i) {
        out_x[i] = in_x[i] * 2.0f;
        out_y[i] = in_y[i] * 2.0f;
        out_z[i] = in_z[i] * 2.0f;
        out_w[i] = in_w[i] * 2.0f;
    }
}

// 验证函数（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, CInt n, const string& kernel_name) {
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(gpu_result[i] - cpu_result[i]) > EPSILON) {
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

bool verify_results_aos(const AoS* gpu_result, const vector<AoS>& cpu_result, CInt n, const string& kernel_name) {
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(gpu_result[i].x - cpu_result[i].x) > EPSILON ||
            std::abs(gpu_result[i].y - cpu_result[i].y) > EPSILON ||
            std::abs(gpu_result[i].z - cpu_result[i].z) > EPSILON ||
            std::abs(gpu_result[i].w - cpu_result[i].w) > EPSILON) {
            cout << "✗ " << kernel_name << " FAILED: 索引 " << i << " 发现不匹配的数据\n";
            success = false;
            break;
        }
    }
    if (success) {
        cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result[0].x << " (期望 " << cpu_result[0].x << ")\n";
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

// GPU 封装函数（GPU，手写）
template<typename KernelFunc>
GpuTimingResult coalesced_gpu(CRMatrix h_input, RMatrix h_output, CInt n, CInt iterations, KernelFunc kernel) {
    PFloat d_input = nullptr, d_output = nullptr;
    CSize size_input = n * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_input));
    CUDA_CHECK(cudaMemset(d_output, 0, size_input));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
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
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_input, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

template<typename KernelFunc>
GpuTimingResult strided_gpu(CRMatrix h_input, RMatrix h_output, CInt n, CInt stride, CInt iterations, KernelFunc kernel) {
    PFloat d_input = nullptr, d_output = nullptr;
    CSize size_input = n * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_input));
    CUDA_CHECK(cudaMemset(d_output, 0, size_input));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    int active_threads = cdiv(n, stride);
    const dim3 block(BLOCK_SIZE);
    const dim3 grid(cdiv(active_threads, BLOCK_SIZE));
    
    kernel<<<grid, block>>>(d_input, d_output, n, stride);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_input, d_output, n, stride);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_input, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

template<typename KernelFunc>
GpuTimingResult aos_gpu(const vector<AoS>& h_input, vector<AoS>& h_output, CInt n, CInt iterations, KernelFunc kernel) {
    AoS* d_input = nullptr;
    AoS* d_output = nullptr;
    CSize size_input = n * sizeof(AoS);
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_input));
    CUDA_CHECK(cudaMemset(d_output, 0, size_input));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
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
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_input, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

template<typename KernelFunc>
GpuTimingResult soa_gpu(CRMatrix h_in_x, CRMatrix h_in_y, CRMatrix h_in_z, CRMatrix h_in_w,
                        RMatrix h_out_x, RMatrix h_out_y, RMatrix h_out_z, RMatrix h_out_w,
                        CInt n, CInt iterations, KernelFunc kernel) {
    CSize size_array = n * sizeof(float);
    
    SoA d_input, d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input.x, size_array));
    CUDA_CHECK(cudaMalloc((void**)&d_input.y, size_array));
    CUDA_CHECK(cudaMalloc((void**)&d_input.z, size_array));
    CUDA_CHECK(cudaMalloc((void**)&d_input.w, size_array));
    
    CUDA_CHECK(cudaMalloc((void**)&d_output.x, size_array));
    CUDA_CHECK(cudaMalloc((void**)&d_output.y, size_array));
    CUDA_CHECK(cudaMalloc((void**)&d_output.z, size_array));
    CUDA_CHECK(cudaMalloc((void**)&d_output.w, size_array));
    CUDA_CHECK(cudaMemset(d_output.x, 0, size_array));
    CUDA_CHECK(cudaMemset(d_output.y, 0, size_array));
    CUDA_CHECK(cudaMemset(d_output.z, 0, size_array));
    CUDA_CHECK(cudaMemset(d_output.w, 0, size_array));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input.x, h_in_x.data(), size_array, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input.y, h_in_y.data(), size_array, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input.z, h_in_z.data(), size_array, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input.w, h_in_w.data(), size_array, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
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
    CUDA_CHECK(cudaMemcpy(h_out_x.data(), d_output.x, size_array, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_y.data(), d_output.y, size_array, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_z.data(), d_output.z, size_array, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_w.data(), d_output.w, size_array, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_input.x));
    CUDA_CHECK(cudaFree(d_input.y));
    CUDA_CHECK(cudaFree(d_input.z));
    CUDA_CHECK(cudaFree(d_input.w));
    
    CUDA_CHECK(cudaFree(d_output.x));
    CUDA_CHECK(cudaFree(d_output.y));
    CUDA_CHECK(cudaFree(d_output.z));
    CUDA_CHECK(cudaFree(d_output.w));
    
    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt n = 1 << 24; // 16M elements
    CInt iterations = 100;
    CInt stride = 2;

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      内存访问模式优化性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "单数组数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "AoS/SoA 总数据大小：" << fixed << setprecision(2) << total_mb * 4 << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 一维数组准备（合并/跨步访问使用）
    Matrix h_input(n);
    Matrix h_cpu_output_coalesced(n, 0.0f);
    Matrix h_cpu_output_strided2(n, 0.0f);
    Matrix h_gpu_output_coalesced(n, 0.0f);
    Matrix h_gpu_output_strided2(n, 0.0f);
    
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i % 100);
    }
    
    // AoS 数组准备
    vector<AoS> aos_input(n);
    vector<AoS> aos_cpu_output(n);
    vector<AoS> aos_gpu_output(n);
    for (int i = 0; i < n; ++i) {
        aos_input[i].x = static_cast<float>(i % 100);
        aos_input[i].y = static_cast<float>((i + 1) % 100);
        aos_input[i].z = static_cast<float>((i + 2) % 100);
        aos_input[i].w = static_cast<float>((i + 3) % 100);
    }
    
    // SoA 数组准备
    Matrix soa_in_x(n), soa_in_y(n), soa_in_z(n), soa_in_w(n);
    Matrix soa_cpu_out_x(n, 0.0f), soa_cpu_out_y(n, 0.0f), soa_cpu_out_z(n, 0.0f), soa_cpu_out_w(n, 0.0f);
    Matrix soa_gpu_out_x(n, 0.0f), soa_gpu_out_y(n, 0.0f), soa_gpu_out_z(n, 0.0f), soa_gpu_out_w(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        soa_in_x[i] = aos_input[i].x;
        soa_in_y[i] = aos_input[i].y;
        soa_in_z[i] = aos_input[i].z;
        soa_in_w[i] = aos_input[i].w;
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    coalesced_access_cpu(h_input, h_cpu_output_coalesced, n);
    cpuTimer.stop();
    double cpu_coalesced_ms = cpuTimer.elapsed_ms();
    
    strided_access_cpu(h_input, h_cpu_output_strided2, n, stride);
    aos_access_cpu(aos_input, aos_cpu_output, n);
    soa_access_cpu(soa_in_x, soa_in_y, soa_in_z, soa_in_w,
                   soa_cpu_out_x, soa_cpu_out_y, soa_cpu_out_z, soa_cpu_out_w, n);
    
    cout << "CPU 执行时间：   " << setw(8) << cpu_coalesced_ms << " ms\n";
    cout << "\n";

    // GPU 计算
    cout << "--- GPU 版本 1: 合并访问 ---\n";
    GpuTimingResult res_coalesced = coalesced_gpu(h_input, h_gpu_output_coalesced, n, iterations, coalesced_access);
    cout << "H2D 传输时间：   " << setw(8) << res_coalesced.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_coalesced.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_coalesced.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_coalesced.total_ms << " ms\n";
    cout << "\n";

    cout << "--- GPU 版本 2: 跨步访问 (Stride " << stride << ") ---\n";
    GpuTimingResult res_strided2 = strided_gpu(h_input, h_gpu_output_strided2, n, stride, iterations, strided_access);
    cout << "H2D 传输时间：   " << setw(8) << res_strided2.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_strided2.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_strided2.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_strided2.total_ms << " ms\n";
    cout << "\n";

    cout << "--- GPU 版本 3: AoS 访问 ---\n";
    GpuTimingResult res_aos = aos_gpu(aos_input, aos_gpu_output, n, iterations, aos_access);
    cout << "H2D 传输时间：   " << setw(8) << res_aos.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_aos.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_aos.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_aos.total_ms << " ms\n";
    cout << "\n";

    cout << "--- GPU 版本 4: SoA 访问 ---\n";
    GpuTimingResult res_soa = soa_gpu(soa_in_x, soa_in_y, soa_in_z, soa_in_w,
                    soa_gpu_out_x, soa_gpu_out_y, soa_gpu_out_z, soa_gpu_out_w, n, iterations, soa_access);
    cout << "H2D 传输时间：   " << setw(8) << res_soa.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_soa.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_soa.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_soa.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析与带宽利用率 ---\n";
    double bytes_1d_useful = 2.0 * cdiv(n, stride) * FSIZE; // 有效处理的字节数
    double bytes_1d_all = 2.0 * n * FSIZE; // 每个元素 4 字节，读写各一次
    
    // 带宽计算：传输的总有效字节数 / 核执行时间
    double bw_coalesced = (bytes_1d_all / 1e9) / (res_coalesced.kernel_ms / 1000.0);
    double bw_strided2 = (bytes_1d_useful / 1e9) / (res_strided2.kernel_ms / 1000.0);

    double bytes_struct = 2.0 * n * sizeof(AoS); // AoS 对齐为16 bytes，每个读取再写入
    double bw_aos = (bytes_struct / 1e9) / (res_aos.kernel_ms / 1000.0);
    double bw_soa = (bytes_struct / 1e9) / (res_soa.kernel_ms / 1000.0);

    cout << "合并访问    有效带宽：" << setw(8) << setprecision(2) << bw_coalesced << " GB/s\n";
    cout << "跨步访问("<<stride<<") 有效带宽：" << setw(8) << setprecision(2) << bw_strided2 << " GB/s\n";
    cout << "AoS 访问    有效带宽：" << setw(8) << setprecision(2) << bw_aos << " GB/s\n";
    cout << "SoA 访问    有效带宽：" << setw(8) << setprecision(2) << bw_soa << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "单数组合并 vs 跨步(2): " << setprecision(2) << res_strided2.kernel_ms / res_coalesced.kernel_ms << "x 变慢\n";
    cout << "SoA结构访问 vs AoS访问: " << setprecision(2) << res_aos.kernel_ms / res_soa.kernel_ms << "x 加速\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_gpu_output_coalesced, h_cpu_output_coalesced, n, "Coalesced Access");
    bool pass2 = verify_results(h_gpu_output_strided2, h_cpu_output_strided2, n, "Strided Access (stride=2)");
    bool pass3 = verify_results_aos(aos_gpu_output.data(), aos_cpu_output, n, "AoS Access");
    bool pass4 = verify_results(soa_gpu_out_x, soa_cpu_out_x, n, "SoA Access X");
    
    // GPU/CPU 结果一致性验证
    if (pass1 && pass2 && pass3 && pass4) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
