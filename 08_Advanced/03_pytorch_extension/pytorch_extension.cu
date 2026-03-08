// PyTorch Extension (Swish 自定义激活算子实现与基准测试)
#include <code_abbreviation.h>

// Swish 前向传播计算（GPU kernel，手写）
__global__ void swish_forward_kernel(CPFloat x, PFloat y, CInt n) {
    CInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = x[tid];
        // Swish(x) = x / (1 + exp(-x))
        y[tid] = val / (1.0f + expf(-val));
    }
}

// Swish 反向传播梯度计算（GPU kernel，手写）
__global__ void swish_backward_kernel(CPFloat grad_y, CPFloat x, PFloat grad_x, CInt n) {
    CInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = x[tid];
        float sigmoid_x = 1.0f / (1.0f + expf(-val));
        float swish_x = val * sigmoid_x;
        // d(Swish)/dx = Swish(x) + sigmoid(x) * (1 - Swish(x))
        float grad_val = swish_x + sigmoid_x * (1.0f - swish_x);
        grad_x[tid] = grad_y[tid] * grad_val;
    }
}

// Swish 前向传播计算（CPU，手写）
void swish_forward_cpu(CPFloat x, PFloat y, CInt n) {
    for (int i = 0; i < n; ++i) {
        float val = x[i];
        y[i] = val / (1.0f + exp(-val));
    }
}

// Swish 反向传播梯度计算（CPU，手写）
void swish_backward_cpu(CPFloat grad_y, CPFloat x, PFloat grad_x, CInt n) {
    for (int i = 0; i < n; ++i) {
        float val = x[i];
        float sigmoid_x = 1.0f / (1.0f + exp(-val));
        float swish_x = val * sigmoid_x;
        float grad_val = swish_x + sigmoid_x * (1.0f - swish_x);
        grad_x[i] = grad_y[i] * grad_val;
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
        
        // 相对误差和绝对误差验证
        if (diff > epsilon && (diff / (abs(cpu_v) + 1e-5f)) > 1e-4f) {
            error_count++;
        }
    }

    if (error_count > 0) {
        cout << "✗ " << kernel_name << " FAILED: " << error_count << " 个元素超出误差阈值\n";
        cout << "  最大差异异常位于索引 " << max_diff_idx
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

// Swish 前向 GPU 封装（GPU，部分手写）
template<typename KernelFunc>
GpuTimingResult swish_forward_gpu(CRMatrix h_x, RMatrix h_y, CInt n, CInt iterations, KernelFunc kernel) {
    PFloat d_x = nullptr;
    PFloat d_y = nullptr;
    
    CSize size_io = n * FSIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_x, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_y, size_io));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_io, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(BLOCK_SIZE_1D);
    const dim3 grid(cdiv(n, BLOCK_SIZE_1D));

    // 预热
    kernel<<<grid, block>>>(d_x, d_y, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_x, d_y, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, size_io, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return result;
}

// Swish 反向 GPU 封装（GPU，部分手写）
template<typename KernelFunc>
GpuTimingResult swish_backward_gpu(CRMatrix h_grad_y, CRMatrix h_x, RMatrix h_grad_x, CInt n, CInt iterations, KernelFunc kernel) {
    PFloat d_grad_y = nullptr;
    PFloat d_x = nullptr;
    PFloat d_grad_x = nullptr;
    
    CSize size_io = n * FSIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_grad_y, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_x, size_io));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_x, size_io));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_grad_y, h_grad_y.data(), size_io, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_io, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(BLOCK_SIZE_1D);
    const dim3 grid(cdiv(n, BLOCK_SIZE_1D));

    // 预热
    kernel<<<grid, block>>>(d_grad_y, d_x, d_grad_x, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_grad_y, d_x, d_grad_x, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_grad_x.data(), d_grad_x, size_io, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_grad_y));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_grad_x));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt n = 10 * 1024 * 1024; // 10M
    CInt iterations = 100;

    CSize size_io = n * FSIZE;
    const double total_mb = size_io / (1024.0 * 1024.0); // 单个数组的大小

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      PyTorch Extension 基准测试计算层\n";
    cout << "========================================\n";
    cout << "自定义算子：Swish Activation (Forward & Backward)\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "单数组大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE_1D << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_x(n), h_y_cpu(n), h_y_gpu(n);
    Matrix h_grad_y(n), h_grad_x_cpu(n), h_grad_x_gpu(n);

    srand(1234);
    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(rand() % 200 - 100) / 10.0f;        // [-10, 10]
        h_grad_y[i] = static_cast<float>(rand() % 200 - 100) / 100.0f;  // [-1, 1]
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    
    cpuTimer.start();
    swish_forward_cpu(h_x.data(), h_y_cpu.data(), n);
    cpuTimer.stop();
    double cpu_fwd_ms = cpuTimer.elapsed_ms();
    
    cpuTimer.start();
    swish_backward_cpu(h_grad_y.data(), h_x.data(), h_grad_x_cpu.data(), n);
    cpuTimer.stop();
    double cpu_bwd_ms = cpuTimer.elapsed_ms();

    cout << "Forward CPU 执行时间：   " << setw(8) << cpu_fwd_ms << " ms\n";
    cout << "Backward CPU 执行时间：  " << setw(8) << cpu_bwd_ms << " ms\n";
    cout << "\n";

    // GPU 计算
    cout << "--- GPU 版本: Custom Swish ---\n";
    GpuTimingResult res_fwd = swish_forward_gpu(h_x, h_y_gpu, n, iterations, swish_forward_kernel);
    cout << "[Forward] H2D 传输时间： " << setw(8) << res_fwd.h2d_ms << " ms\n";
    cout << "[Forward] Kernel 时间：  " << setw(8) << res_fwd.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "[Forward] D2H 传输时间： " << setw(8) << res_fwd.d2h_ms << " ms\n";

    GpuTimingResult res_bwd = swish_backward_gpu(h_grad_y, h_x, h_grad_x_gpu, n, iterations, swish_backward_kernel);
    cout << "[Backward] H2D 传输时间：" << setw(8) << res_bwd.h2d_ms << " ms\n";
    cout << "[Backward] Kernel 时间： " << setw(8) << res_bwd.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "[Backward] D2H 传输时间：" << setw(8) << res_bwd.d2h_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    cout << "[Forward] CPU vs GPU Kernel 加速比：" << setprecision(2) << (cpu_fwd_ms / res_fwd.kernel_ms) << "x\n";
    cout << "[Backward] CPU vs GPU Kernel 加速比：" << setprecision(2) << (cpu_bwd_ms / res_bwd.kernel_ms) << "x\n";

    // 带宽计算：Forward 读1写1，Backward 读2写1
    double bw_fwd = ((size_io * 2) / 1e9) / (res_fwd.kernel_ms / 1000.0);
    double bw_bwd = ((size_io * 3) / 1e9) / (res_bwd.kernel_ms / 1000.0);
    cout << "GPU Forward 有效带宽：" << setprecision(2) << bw_fwd << " GB/s\n";
    cout << "GPU Backward 有效带宽：" << setprecision(2) << bw_bwd << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_y_gpu, h_y_cpu, "Swish Forward");
    bool pass2 = verify_results(h_grad_x_gpu, h_grad_x_cpu, "Swish Backward");

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}

// ==========================================================
//   PyTorch C++ ATen API 绑定层 (供 Extension 编译调用)
// ==========================================================
/* 
    说明：以下代码由预处理器宏 BUILD_PYTORCH_EXTENSION 保护包裹。
    当您使用原生的 CMake 时，这个宏并未定义，所以这段代码会被编译器彻底忽略，这防止了 <torch/extension.h> 缺失引发报错。
    当您在 Python 端执行 `python setup.py install` 使用 PyTorch JIT/Setuptools 编译扩展时，
    只需在 cpp_extension 中把宏拉起即可构建供 Python 引用的动态链接库 (.so / .pyd)
*/

#ifdef BUILD_PYTORCH_EXTENSION

#include <torch/extension.h>

// Forward 接口封装
torch::Tensor swish_forward_cuda(torch::Tensor x) {
    // 确保张量位于 CUDA，且是连续的一块显存
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    CInt n = x.numel();
    // 自动通过 PyTorch 内存池分配相同维度类型的显存作为返回结果
    auto y = torch::empty_like(x);

    const dim3 block(BLOCK_SIZE_1D);
    const dim3 grid(cdiv(n, BLOCK_SIZE_1D));

    // 获取张量数据指针(Pointer)并传递给核函数
    swish_forward_kernel<<<grid, block>>>(
        x.data_ptr<float>(), 
        y.data_ptr<float>(), 
        n
    );
    // 拓展环境下尽量避免粗暴同步，除非必要。此处直接返回，PyTorch 流引擎会处理异步依赖。
    return y;
}

// Backward 接口封装
torch::Tensor swish_backward_cuda(torch::Tensor grad_y, torch::Tensor x) {
    TORCH_CHECK(grad_y.device().is_cuda(), "grad_y must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");

    CInt n = x.numel();
    auto grad_x = torch::empty_like(x);

    const dim3 block(BLOCK_SIZE_1D);
    const dim3 grid(cdiv(n, BLOCK_SIZE_1D));

    swish_backward_kernel<<<grid, block>>>(
        grad_y.data_ptr<float>(),
        x.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        n
    );

    return grad_x;
}

// 绑定到 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward_cuda, "Swish forward (CUDA)");
    m.def("backward", &swish_backward_cuda, "Swish backward (CUDA)");
}

#endif // BUILD_PYTORCH_EXTENSION
