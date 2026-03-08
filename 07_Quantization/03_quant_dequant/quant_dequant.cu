// 量化/反量化 Kernel
#include <code_abbreviation.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <algorithm>

// FP32 -> INT8 Per-tensor 量化（GPU kernel，手写）
__global__ void quantize_per_tensor(CPFloat input, int8_t* output, float scale, CInt n) {
    CInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float scaled = roundf(input[tid] / scale);
        output[tid] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, scaled)));
    }
}

// INT8 -> FP32 反量化（GPU kernel，手写）
__global__ void dequantize_per_tensor(const int8_t* input, PFloat output, float scale, CInt n) {
    CInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = static_cast<float>(input[tid]) * scale;
    }
}

// Per-channel 量化（GPU kernel，手写）
__global__ void quantize_per_channel(CPFloat input, int8_t* output, CPFloat scales, CInt batch, CInt channels) {
    CInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    CInt n = batch * channels;
    if (tid < n) {
        CInt c = tid % channels;
        float scale = scales[c];
        float scaled = roundf(input[tid] / scale);
        output[tid] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, scaled)));
    }
}

// FP32 -> FP16 转换（GPU kernel，手写）
__global__ void fp32_to_fp16(CPFloat input, half* output, CInt n) {
    CInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = __float2half(input[tid]);
    }
}

// FP16 -> FP32 转换（GPU kernel，手写）
__global__ void fp16_to_fp32(const half* input, PFloat output, CInt n) {
    CInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = __half2float(input[tid]);
    }
}

// CPU 参考实现（CPU，手写）
void quantize_per_tensor_cpu(CPFloat input, int8_t* output, float scale, CInt n) {
    for (int i = 0; i < n; ++i) {
        float scaled = std::round(input[i] / scale);
        output[i] = static_cast<int8_t>(std::clamp(scaled, -128.0f, 127.0f));
    }
}

void dequantize_per_tensor_cpu(const int8_t* input, PFloat output, float scale, CInt n) {
    for (int i = 0; i < n; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
}

void quantize_per_channel_cpu(CPFloat input, int8_t* output, CPFloat scales, CInt batch, CInt channels) {
    CInt n = batch * channels;
    for (int i = 0; i < n; ++i) {
        CInt c = i % channels;
        float scale = scales[c];
        float scaled = std::round(input[i] / scale);
        output[i] = static_cast<int8_t>(std::clamp(scaled, -128.0f, 127.0f));
    }
}

void fp32_to_fp16_cpu(CPFloat input, half* output, CInt n) {
    for (int i = 0; i < n; ++i) {
        output[i] = __float2half(input[i]);
    }
}

void fp16_to_fp32_cpu(const half* input, PFloat output, CInt n) {
    for (int i = 0; i < n; ++i) {
        output[i] = __half2float(input[i]);
    }
}

// 验证结果（AI 生成）
template <typename T>
bool verify_results(const std::vector<T>& gpu_result, const std::vector<T>& cpu_result, const string& kernel_name, CFloat epsilon = 1e-3f) {
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配\n";
        return false;
    }

    int error_count = 0;
    float max_diff = 0.0f;
    int max_diff_idx = 0;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float gpu_v, cpu_v;
        if constexpr (std::is_same_v<T, half>) {
            gpu_v = __half2float(gpu_result[i]);
            cpu_v = __half2float(cpu_result[i]);
        } else {
            gpu_v = static_cast<float>(gpu_result[i]);
            cpu_v = static_cast<float>(cpu_result[i]);
        }
        
        float diff = fabs(gpu_v - cpu_v);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = static_cast<int>(i);
        }
        
        if constexpr (std::is_integral_v<T>) {
            if (diff > 0) error_count++;
        } else {
            if (diff > epsilon) error_count++;
        }
    }

    if (error_count > 0) {
        cout << "✗ " << kernel_name << " FAILED: " << error_count << " 个元素超出误差阈值\n";
        float print_gpu, print_cpu;
        if constexpr (std::is_same_v<T, half>) {
            print_gpu = __half2float(gpu_result[max_diff_idx]);
            print_cpu = __half2float(cpu_result[max_diff_idx]);
        } else {
            print_gpu = static_cast<float>(gpu_result[max_diff_idx]);
            print_cpu = static_cast<float>(cpu_result[max_diff_idx]);
        }
        cout << "  首个错误异常差异位于索引 " << max_diff_idx
             << "：GPU=" << print_gpu
             << ", CPU=" << print_cpu
             << ", 差异=" << max_diff << "\n";
        return false;
    }

    float print_gpu, print_cpu;
    if constexpr (std::is_same_v<T, half>) {
        print_gpu = __half2float(gpu_result[0]);
        print_cpu = __half2float(cpu_result[0]);
    } else {
        print_gpu = static_cast<float>(gpu_result[0]);
        print_cpu = static_cast<float>(cpu_result[0]);
    }
    cout << "✓ " << kernel_name << " PASSED: 结果 " << print_gpu
         << " (期望 " << print_cpu << ")\n";
    return true;
}


// 通用 Quant/Dequant 封装（带缩放参数 Float Scale）（GPU，手写）
template<typename InputT, typename OutputT, typename KernelFunc>
GpuTimingResult quant_gpu_scale(const std::vector<InputT>& h_input, std::vector<OutputT>& h_output, 
                                float scale, CInt n, CInt iterations, KernelFunc kernel) {
    InputT* d_input = nullptr;
    OutputT* d_output = nullptr;

    CSize size_in = n * sizeof(InputT);
    CSize size_out = n * sizeof(OutputT);

    CUDA_CHECK(cudaMalloc((void**)&d_input, size_in));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_out));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_in, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(BLOCK_SIZE_1D);
    const dim3 grid(cdiv(n, BLOCK_SIZE_1D));

    kernel<<<grid, block>>>(d_input, d_output, scale, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_input, d_output, scale, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_out, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

// 通用 Quant/Dequant 封装（带数组 Float Scales）（GPU，手写）
template<typename InputT, typename OutputT, typename KernelFunc>
GpuTimingResult quant_gpu_channel(const std::vector<InputT>& h_input, std::vector<OutputT>& h_output, 
                                  const std::vector<float>& h_scales, CInt batch, CInt channels, CInt iterations, KernelFunc kernel) {
    InputT* d_input = nullptr;
    OutputT* d_output = nullptr;
    float* d_scales = nullptr;

    CInt n = batch * channels;
    CSize size_in = n * sizeof(InputT);
    CSize size_out = n * sizeof(OutputT);
    CSize size_scales = channels * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_input, size_in));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_out));
    CUDA_CHECK(cudaMalloc((void**)&d_scales, size_scales));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_in, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), size_scales, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(BLOCK_SIZE_1D);
    const dim3 grid(cdiv(n, BLOCK_SIZE_1D));

    kernel<<<grid, block>>>(d_input, d_output, d_scales, batch, channels);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_input, d_output, d_scales, batch, channels);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_out, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_scales));

    return result;
}

// 通用 Quant/Dequant 封装（无缩放参数）（GPU，手写）
template<typename InputT, typename OutputT, typename KernelFunc>
GpuTimingResult cast_gpu_basic(const std::vector<InputT>& h_input, std::vector<OutputT>& h_output, 
                               CInt n, CInt iterations, KernelFunc kernel) {
    InputT* d_input = nullptr;
    OutputT* d_output = nullptr;

    CSize size_in = n * sizeof(InputT);
    CSize size_out = n * sizeof(OutputT);

    CUDA_CHECK(cudaMalloc((void**)&d_input, size_in));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_out));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_in, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(BLOCK_SIZE_1D);
    const dim3 grid(cdiv(n, BLOCK_SIZE_1D));

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
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_out, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt N = 10 * 1024 * 1024; // 10M elements
    CInt batch = 1024;
    CInt channels = 10240;      // batch * channels = 10M
    CInt iterations = 100;

    CSize size_fp32 = N * sizeof(float);
    const double total_mb = size_fp32 / (1024.0 * 1024.0);
    const float scale_tensor = 0.05f;

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      量化/反量化性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << N << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE_1D << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_input_fp32(N);
    std::vector<int8_t> h_input_int8(N);
    std::vector<half> h_input_fp16(N);
    Matrix h_scales(channels);

    std::vector<int8_t> h_out_q_tensor_gpu(N);
    std::vector<int8_t> h_out_q_tensor_cpu(N);

    Matrix h_out_dq_tensor_gpu(N);
    Matrix h_out_dq_tensor_cpu(N);

    std::vector<int8_t> h_out_q_chan_gpu(N);
    std::vector<int8_t> h_out_q_chan_cpu(N);

    std::vector<half> h_out_fp16_gpu(N);
    std::vector<half> h_out_fp16_cpu(N);

    Matrix h_out_fp32_gpu(N);
    Matrix h_out_fp32_cpu(N);

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_input_fp32[i] = static_cast<float>(rand() % 400 - 200) / 100.0f; // [-2.0, 2.0]
        h_input_int8[i] = static_cast<int8_t>(rand() % 256 - 128);          // [-128, 127]
        h_input_fp16[i] = __float2half(h_input_fp32[i]);
    }
    for (int i = 0; i < channels; ++i) {
        h_scales[i] = static_cast<float>(rand() % 100 + 1) / 1000.0f; // [0.001, 0.1]
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;

    cpuTimer.start();
    quantize_per_tensor_cpu(h_input_fp32.data(), h_out_q_tensor_cpu.data(), scale_tensor, N);
    cpuTimer.stop();
    double cpu_q1_ms = cpuTimer.elapsed_ms();

    cpuTimer.start();
    dequantize_per_tensor_cpu(h_input_int8.data(), h_out_dq_tensor_cpu.data(), scale_tensor, N);
    cpuTimer.stop();
    double cpu_dq1_ms = cpuTimer.elapsed_ms();

    cpuTimer.start();
    quantize_per_channel_cpu(h_input_fp32.data(), h_out_q_chan_cpu.data(), h_scales.data(), batch, channels);
    cpuTimer.stop();
    double cpu_q2_ms = cpuTimer.elapsed_ms();

    cpuTimer.start();
    fp32_to_fp16_cpu(h_input_fp32.data(), h_out_fp16_cpu.data(), N);
    cpuTimer.stop();
    double cpu_c1_ms = cpuTimer.elapsed_ms();

    cpuTimer.start();
    fp16_to_fp32_cpu(h_input_fp16.data(), h_out_fp32_cpu.data(), N);
    cpuTimer.stop();
    double cpu_c2_ms = cpuTimer.elapsed_ms();

    cout << "CPU 执行时间：\n"
         << " - Quantize Per-Tensor:   " << setw(8) << cpu_q1_ms << " ms\n"
         << " - Dequantize Per-Tensor: " << setw(8) << cpu_dq1_ms << " ms\n"
         << " - Quantize Per-Channel:  " << setw(8) << cpu_q2_ms << " ms\n"
         << " - FP32 to FP16:          " << setw(8) << cpu_c1_ms << " ms\n"
         << " - FP16 to FP32:          " << setw(8) << cpu_c2_ms << " ms\n\n";

    // GPU 版本 1: FP32 to INT8 Per-Tensor 量化
    cout << "--- GPU 版本 1: FP32 to INT8 Per-Tensor 量化 ---\n";
    GpuTimingResult result_q1 = quant_gpu_scale(h_input_fp32, h_out_q_tensor_gpu, scale_tensor, N, iterations, quantize_per_tensor);
    cout << "H2D 传输时间：   " << setw(8) << result_q1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result_q1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result_q1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result_q1.total_ms << " ms\n\n";

    // GPU 版本 2: INT8 to FP32 Per-Tensor 反量化
    cout << "--- GPU 版本 2: INT8 to FP32 Per-Tensor 反量化 ---\n";
    GpuTimingResult result_dq1 = quant_gpu_scale(h_input_int8, h_out_dq_tensor_gpu, scale_tensor, N, iterations, dequantize_per_tensor);
    cout << "H2D 传输时间：   " << setw(8) << result_dq1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result_dq1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result_dq1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result_dq1.total_ms << " ms\n\n";

    // GPU 版本 3: FP32 to INT8 Per-Channel 量化
    cout << "--- GPU 版本 3: FP32 to INT8 Per-Channel 量化 ---\n";
    GpuTimingResult result_q2 = quant_gpu_channel(h_input_fp32, h_out_q_chan_gpu, h_scales, batch, channels, iterations, quantize_per_channel);
    cout << "H2D 传输时间：   " << setw(8) << result_q2.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result_q2.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result_q2.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result_q2.total_ms << " ms\n\n";

    // GPU 版本 4: FP32 to FP16 直接转换
    cout << "--- GPU 版本 4: FP32 to FP16 直接转换 ---\n";
    GpuTimingResult result_c1 = cast_gpu_basic(h_input_fp32, h_out_fp16_gpu, N, iterations, fp32_to_fp16);
    cout << "H2D 传输时间：   " << setw(8) << result_c1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result_c1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result_c1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result_c1.total_ms << " ms\n\n";

    // GPU 版本 5: FP16 to FP32 直接转换
    cout << "--- GPU 版本 5: FP16 to FP32 直接转换 ---\n";
    GpuTimingResult result_c2 = cast_gpu_basic(h_input_fp16, h_out_fp32_gpu, N, iterations, fp16_to_fp32);
    cout << "H2D 传输时间：   " << setw(8) << result_c2.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result_c2.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result_c2.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result_c2.total_ms << " ms\n\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    cout << "CPU vs GPU Kernel 加速比：\n"
         << " - Quantize Per-Tensor:   " << setprecision(2) << (cpu_q1_ms / result_q1.kernel_ms) << "x\n"
         << " - Dequantize Per-Tensor: " << setprecision(2) << (cpu_dq1_ms / result_dq1.kernel_ms) << "x\n"
         << " - Quantize Per-Channel:  " << setprecision(2) << (cpu_q2_ms / result_q2.kernel_ms) << "x\n"
         << " - FP32 to FP16:          " << setprecision(2) << (cpu_c1_ms / result_c1.kernel_ms) << "x\n"
         << " - FP16 to FP32:          " << setprecision(2) << (cpu_c2_ms / result_c2.kernel_ms) << "x\n\n";

    auto calc_bw = [](double reads, double writes, float ms) {
        double bytes = reads + writes;
        return (bytes / 1e9) / (ms / 1000.0);
    };

    double bytes_f_r = N * sizeof(float);
    double bytes_i_w = N * sizeof(int8_t);
    double bytes_h_w = N * sizeof(half);
    double scales_r = channels * sizeof(float);

    cout << "GPU 有效带宽：\n"
         << " - Quantize Per-Tensor:   " << setprecision(2) << calc_bw(bytes_f_r, bytes_i_w, result_q1.kernel_ms) << " GB/s\n"
         << " - Dequantize Per-Tensor: " << setprecision(2) << calc_bw(bytes_i_w, bytes_f_r, result_dq1.kernel_ms) << " GB/s\n"
         << " - Quantize Per-Channel:  " << setprecision(2) << calc_bw(bytes_f_r + scales_r, bytes_i_w, result_q2.kernel_ms) << " GB/s\n"
         << " - FP32 to FP16:          " << setprecision(2) << calc_bw(bytes_f_r, bytes_h_w, result_c1.kernel_ms) << " GB/s\n"
         << " - FP16 to FP32:          " << setprecision(2) << calc_bw(bytes_h_w, bytes_f_r, result_c2.kernel_ms) << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_out_q_tensor_gpu, h_out_q_tensor_cpu, "Quantize Per-Tensor");
    bool pass2 = verify_results(h_out_dq_tensor_gpu, h_out_dq_tensor_cpu, "Dequantize Per-Tensor", 1e-3f);
    bool pass3 = verify_results(h_out_q_chan_gpu, h_out_q_chan_cpu, "Quantize Per-Channel");
    bool pass4 = verify_results(h_out_fp16_gpu, h_out_fp16_cpu, "FP32 to FP16 Cast", 1e-3f);
    bool pass5 = verify_results(h_out_fp32_gpu, h_out_fp32_cpu, "FP16 to FP32 Cast", 1e-3f);

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2 && pass3 && pass4 && pass5) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
