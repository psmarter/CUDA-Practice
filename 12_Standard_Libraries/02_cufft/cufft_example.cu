// cuFFT - 快速傅里叶变换
#include <code_abbreviation.h>
#include <cufft.h>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Complex = std::complex<float>;


// 归一化 Kernel (GPU kernel，手写)
// cuFFT 的 IFFT 不自动除以 N，这是通用的信号处理物理映射需求
__global__ void normalize_complex_kernel(cufftComplex* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

// cuFFT 面向对象封装
class FFT1D {
public:
    cufftHandle plan;
    int n;
    
    FFT1D() : plan(0), n(0) {}
    
    void create_plan(int size) {
        n = size;
        cufftPlan1d(&plan, n, CUFFT_C2C, 1);
    }
    
    void forward(cufftComplex* input, cufftComplex* output) {
        cufftExecC2C(plan, input, output, CUFFT_FORWARD);
    }
    
    void inverse(cufftComplex* input, cufftComplex* output) {
        cufftExecC2C(plan, input, output, CUFFT_INVERSE);
        // 执行归一化 kernel 
        const dim3 block(256);
        const dim3 grid(cdiv(n, 256));
        normalize_complex_kernel<<<grid, block>>>(output, n);
    }
    
    ~FFT1D() {
        if(plan) cufftDestroy(plan);
    }
};

class BatchedFFT {
public:
    cufftHandle plan;
    
    BatchedFFT() : plan(0) {}
    
    void create_plan(int n, int batch) {
        int n_arr[] = {n};
        // 1D batched FFT
        cufftPlanMany(&plan, 1, n_arr, 
                      NULL, 1, n,      // input inembed, stride, dist
                      NULL, 1, n,      // output inembed, stride, dist
                      CUFFT_C2C, batch);
    }
    
    void forward(cufftComplex* input, cufftComplex* output) {
        cufftExecC2C(plan, input, output, CUFFT_FORWARD);
    }
    
    ~BatchedFFT() {
        if(plan) cufftDestroy(plan);
    }
};

class FFT_R2C {
public:
    cufftHandle plan;
    int n;
    
    FFT_R2C() : plan(0), n(0) {}
    
    void create_plan(int size) {
        n = size;
        cufftPlan1d(&plan, n, CUFFT_R2C, 1);
    }
    
    void forward(cufftReal* input, cufftComplex* output) {
        cufftExecR2C(plan, input, output);
    }
    
    ~FFT_R2C() {
        if(plan) cufftDestroy(plan);
    }
};


// 一维离散傅里叶变换 (DFT) (CPU，手写)
// 复杂度 O(N^2)，仅用于校验小规模数据
// 内部使用 double 精度累加，避免 float 的 N 项求和舍入噪声污染参考基准
void dft_1d_cpu(const Complex* input, Complex* output, int n, bool inverse = false) {
    double sign = inverse ? 1.0 : -1.0;
    for (int k = 0; k < n; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (int t = 0; t < n; ++t) {
            double angle = sign * 2.0 * M_PI * k * t / n;
            std::complex<double> w(cos(angle), sin(angle));
            sum += std::complex<double>(input[t].real(), input[t].imag()) * w;
        }
        if (inverse) {
            sum /= static_cast<double>(n);
        }
        output[k] = Complex(static_cast<float>(sum.real()), static_cast<float>(sum.imag()));
    }
}


// 用于常规代码框架不报错的假占位（此文件因为用 cufftComplex 的特殊性，自己重新写一份）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, CInt n, const string& kernel_name) { return true; }

bool verify_complex_results(cufftComplex* gpu_result, const std::vector<Complex>& cpu_result, CInt n, const string& kernel_name) {
    // 保存 cout 格式状态，避免被外部 setprecision/fixed 污染
    auto old_flags = cout.flags();
    auto old_prec  = cout.precision();
    cout << std::defaultfloat << setprecision(6);

    bool success = true;
    constexpr float atol = 1e-3f;  // 绝对容差 (CPU 参考已用 double，误差 << 1e-3)
    constexpr float rtol = 1e-5f;  // 相对容差 (大幅值分量按比例放宽)
    for (int i = 0; i < n; ++i) {
        float diff_real = fabs(gpu_result[i].x - cpu_result[i].real());
        float diff_imag = fabs(gpu_result[i].y - cpu_result[i].imag());
        // 对大幅值分量采用相对容差，避免 float 累积误差导致误判
        float mag = std::max(fabs(gpu_result[i].x), std::max(fabs(gpu_result[i].y),
                    std::max(fabs(cpu_result[i].real()), fabs(cpu_result[i].imag()))));
        float tol = std::max(atol, rtol * mag);
        if (diff_real > tol || diff_imag > tol) {
            cout << std::scientific << setprecision(6);
            cout << "✗ " << kernel_name << " FAILED: 索引 " << i 
                 << " 结果 (" << gpu_result[i].x << "," << gpu_result[i].y << ") "
                 << "期望 (" << cpu_result[i].real() << "," << cpu_result[i].imag() << ")"
                 << " (diff_r=" << diff_real << ", diff_i=" << diff_imag << ", tol=" << tol << ")\n";
            success = false;
            break;
        }
    }
    if (success) {
        cout << "✓ " << kernel_name << " PASSED: 结果 (" << gpu_result[0].x << "," << gpu_result[0].y << ") "
             << "(期望 (" << cpu_result[0].real() << "," << cpu_result[0].imag() << "))\n";
    }
    // 恢复 cout 格式状态
    cout.flags(old_flags);
    cout.precision(old_prec);
    return success;
}




// 1D C2C GPU 封装 (GPU，手写)
GpuTimingResult fft1d_gpu(const std::vector<Complex>& h_input, std::vector<cufftComplex>& h_output, CInt n, CInt iterations, bool inverse = false) {
    cufftComplex* d_data;
    CSize size_bytes = n * sizeof(cufftComplex);
    CUDA_CHECK(cudaMalloc((void**)&d_data, size_bytes));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    // Complex 和 cufftComplex 是结构对齐等价的，可以直接 memcpy
    CUDA_CHECK(cudaMemcpy(d_data, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    FFT1D fft;
    fft.create_plan(n);
    
    // 预热 (In-place)
    if (inverse) fft.inverse(d_data, d_data);
    else         fft.forward(d_data, d_data);
    CUDA_SYNC_CHECK();
    
    // 预热改变了 d_data（in-place），必须重新上传原始数据
    CUDA_CHECK(cudaMemcpy(d_data, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
    
    // 为了测速，如果不重置数据，多次 forward/inverse 会叠加，但是不影响测时执行本身
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        if (inverse) fft.inverse(d_data, d_data);
        else         fft.forward(d_data, d_data);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_data, size_bytes, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    CUDA_CHECK(cudaFree(d_data));
    
    return result;
}

// 批量 C2C GPU 封装 (GPU，手写)
GpuTimingResult batched_fft_gpu(const std::vector<Complex>& h_input, std::vector<cufftComplex>& h_output, CInt n, CInt batch, CInt iterations) {
    cufftComplex* d_data;
    CSize size_bytes = n * batch * sizeof(cufftComplex);
    CUDA_CHECK(cudaMalloc((void**)&d_data, size_bytes));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_data, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    BatchedFFT fft;
    fft.create_plan(n, batch);
    
    // 预热 (In-place)
    fft.forward(d_data, d_data);
    CUDA_SYNC_CHECK();
    
    // 预热改变了 d_data，重新上传原始数据
    CUDA_CHECK(cudaMemcpy(d_data, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        fft.forward(d_data, d_data);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_data, size_bytes, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    CUDA_CHECK(cudaFree(d_data));
    
    return result;
}



// 主函数（部分手写，部分AI 生成）
int main() {
    CInt N_verify = 4096; // 如果太大 CPU DFT O(N^2) 会卡死
    CInt iterations = 100;
    
    CSize size_bytes_verify = N_verify * sizeof(Complex);
    const double total_mb_verify = size_bytes_verify / (1024.0 * 1024.0);

    printDeviceInfo();
    cout << "========================================\n";
    cout << "      cuFFT 官方标准库 频谱计算测试\n";
    cout << "========================================\n";
    cout << "校验波长规模：" << N_verify << " 采样点 (" << fixed << setprecision(4) << total_mb_verify << " MB)\n";
    cout << "验证算法  ：GPU cuFFT vs CPU $O(N^2)$ 自研基准\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 构造信号：两个频率成分混合的实波
    std::vector<Complex> h_signal(N_verify);
    for (int i = 0; i < N_verify; ++i) {
        // frequency = 5 and 20
        float t = (float)i / N_verify;
        float val = sinf(2.0f * M_PI * 5.0f * t) + 0.5f * cosf(2.0f * M_PI * 20.0f * t);
        h_signal[i] = Complex(val, 0.0f);
    }

    std::vector<Complex> h_cpu_fft(N_verify);
    std::vector<Complex> h_cpu_ifft(N_verify);
    std::vector<cufftComplex> h_gpu_fft(N_verify);
    std::vector<cufftComplex> h_gpu_ifft(N_verify);

    // CPU 计算
    cout << "--- CPU 计时 (DFT $O(N^2)$) ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    dft_1d_cpu(h_signal.data(), h_cpu_fft.data(), N_verify, false);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU DFT 耗时：   " << setw(8) << cpu_time_ms << " ms\n";
    
    // CPU IFFT 用于确认我们数学逆转换是正确的
    dft_1d_cpu(h_cpu_fft.data(), h_cpu_ifft.data(), N_verify, true);
    
    cout << "\n";

    // GPU 版本 1：1D Forward C2C
    cout << "--- GPU 版本 1: cuFFT 1D (Forward) ---\n";
    // 提醒：通常一次测速会使得数据失效（因为我们用 in-place 操作，迭代会覆盖），
    // 故重新构造 h_signal 一次以保证结果只做了一次 Forward 被拿回来校验
    GpuTimingResult res_fft = fft1d_gpu(h_signal, h_gpu_fft, N_verify, 1, false); 
    // 下面跑高迭代仅为了拿稳定延时时间，这里用副本测速：
    std::vector<cufftComplex> dummy(N_verify);
    GpuTimingResult res_fft_iter = fft1d_gpu(h_signal, dummy, N_verify, iterations, false); 
    
    cout << "H2D 传输时间：   " << setw(8) << res_fft.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_fft_iter.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_fft.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_fft_iter.kernel_ms << " ms (纯算力段)\n";
    cout << "\n";

    // 将 GPU 的 Forward 结论当做输入进行 Inverse
    std::vector<Complex> gpu_ret_as_complex(N_verify);
    for(int i=0; i<N_verify; ++i) gpu_ret_as_complex[i] = Complex(h_gpu_fft[i].x, h_gpu_fft[i].y);
    
    cout << "--- GPU 版本 2: cuFFT 1D (Inverse) ---\n";
    GpuTimingResult res_ifft = fft1d_gpu(gpu_ret_as_complex, h_gpu_ifft, N_verify, 1, true);
    GpuTimingResult res_ifft_iter = fft1d_gpu(gpu_ret_as_complex, dummy, N_verify, iterations, true);
    cout << "Kernel 执行时间：" << setw(8) << res_ifft_iter.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析与拓展基准 ---\n";
    double speedup_kernel = cpu_time_ms / res_fft_iter.kernel_ms;
    cout << "CPU(Naive) vs GPU(cuFFT) 加速比（" << N_verify << " 维度下）：" << setprecision(2) << speedup_kernel << "x\n";
    cout << ">> 注：这并不公平，因为 CPU 为 O(N^2) 而 GPU 基于 O(N log N) 库。仅仅用于功能对比和感受差距。\n";
    
    // 我们跑一个巨大的 Batched 看看吞吐量
    CInt N_big = 1024;
    CInt Batch = 65536; // 65536 条 1024 FFT 信号 = 64M elements = 512MB
    std::vector<Complex> huge_signals(N_big * Batch, Complex(1.0f, 0.5f));
    std::vector<cufftComplex> huge_out(N_big * Batch);
    
    cout << "\n===> 测试大数据吞吐量 (Batch=" << Batch << ", N=" << N_big << ") <===\n";
    GpuTimingResult res_batch = batched_fft_gpu(huge_signals, huge_out, N_big, Batch, 10);
    double bytes = N_big * Batch * sizeof(cufftComplex);
    // FFT 中每次转换均包含多次全局访存，但用单一 I/O 计算等效评估带宽
    double gpu_bandwidth = (bytes / 1e9) / (res_batch.kernel_ms / 1000.0);
    cout << "巨量 Batch Kernel 执行时间：" << setw(8) << res_batch.kernel_ms << " ms (10 次平均)\n";
    cout << "GPU 最小理论访存有效带宽：" << setprecision(2) << gpu_bandwidth << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_complex_results(h_gpu_fft.data(), h_cpu_fft, N_verify, "FFT (Forward)");
    bool pass2 = verify_complex_results(h_gpu_ifft.data(), h_signal, N_verify, "IFFT(Inverse)");

    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 一致性验证通过！ (包含数学上的完美逆向重构功能验证)\n";
    } else {
        cout << "✗ GPU/CPU 验证存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
