// Nsight Profiling 指南 - 性能分析工具使用
#include <code_abbreviation.h>
#include <string>

// 包含 NVIDIA NVTX 头文件，用于在 Nsight Systems 中画出彩色时间轴
#ifdef _WIN32
// Windows NVTX 需要链接 nvToolsExt64_1.lib
#include <nvtx3/nvToolsExt.h>
#else
#include <nvtx3/nvToolsExt.h>
#endif

/*
========== 完整的性能分析工作流程 ==========

前置条件：
  - nsys 已随 CUDA Toolkit 安装 (包含在 /usr/local/cuda/bin/ 或 /opt/nvidia/nsight-systems/)
  - ncu  需单独安装： sudo apt install nsight-compute
  - ⚠️ 注意：apt 安装后 `ncu` 可能不在 PATH 中！
    请执行：sudo ln -sf /opt/nvidia/nsight-compute/*/ncu /usr/local/bin/ncu
    以及：  sudo ln -sf /opt/nvidia/nsight-compute/*/ncu-ui /usr/local/bin/ncu-ui
    或者：
    找到 ncu 的实际位置
    find /opt -name "ncu" -type f 2>/dev/null
    dpkg -L nsight-compute-2025.1.1 | grep bin/ncu
    找到后直接用完整路径运行，比如：
    /opt/nvidia/nsight-compute/2025.1.1/ncu --set full -o nsight_metrics ./nsight_profiling
  - 编译时添加 -lineinfo 以支持源码级关联 (CMakeLists.txt 已配置)

Step 1: 普通运行查看基准数据
  ./nsight_profiling

Step 2: Nsight Systems - 系统级 Timeline 分析
  nsys profile --trace=cuda,nvtx -o nsight_timeline ./nsight_profiling
  nsys stats nsight_timeline.nsys-rep          # 命令行查看统计表
  nsys-ui nsight_timeline.nsys-rep             # GUI 查看时间线 (需显示器)

Step 3: Nsight Compute - Kernel 级深度指标分析
  sudo ncu --set full -o nsight_metrics ./nsight_profiling
  ncu-ui nsight_metrics.ncu-rep           # GUI 查看报告 (需显示器)
  (如果提示找不到 ncu，请先执行前置条件中的 ln -s 软链接挂载命令)

Step 3-alt: 不用 GUI 的 ncu 命令行分析
  sudo ncu --kernel-name profile_example --launch-skip 2 --launch-count 2 ./nsight_profiling
  这会跳过 2 次预热启动，抓取 Bad+Good 各 1 次，直接在终端打印指标。
*/

// 0. NVTX 辅助宏

// 为了代码简洁，提供 RAII 风格的 NVTX 范围捕捉器
class NVTXRange {
public:
    NVTXRange(const char* name) {
        nvtxRangePush(name);
    }
    ~NVTXRange() {
        nvtxRangePop();
    }
};

#define PROFILE_SCOPE(name) NVTXRange nvtx_scope_(name)

// 1. GPU Kernel 函数（手写）

// 用于 profiling 的 "Bad" Kernel (故意制造非合并访存) (GPU kernel，手写)
// 将线性 idx 映射为跳跃式地址，使得同一 Warp 内的相邻线程访问相距 stride 的地址
// 结果：每次 32B 事务只命中 1 个有效 float，Global Load Efficiency 极低
__global__ void profile_example_kernel_bad(CPFloat input, PFloat output, CInt n, CInt stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // 将连续的线程 ID 映射成跳跃式内存地址
    // 效果：thread 0,1,2,...31 分别访问 0, chunk, 2*chunk, ... 形成非合并访存
    int chunk = n / stride;  // 每个 stride 组的元素数量
    int mapped_idx = (idx % stride) * chunk + (idx / stride);
    if (mapped_idx >= n) return;
    
    float val = input[mapped_idx];
    val = val * val + val;  // 处理
    output[mapped_idx] = val;
}

// 修复过的 "Good" Kernel (标准的合并访存) (GPU kernel，手写)
__global__ void profile_example_kernel_good(CPFloat input, PFloat output, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        val = val * val + val;  // 处理
        output[idx] = val;
    }
}

// 2. CPU 参考实现函数（手写）

// CPU 算子基础参考 (CPU，手写)
void profile_cpu(CRMatrix input, RMatrix output, CInt n) {
    for (int i = 0; i < n; ++i) {
        float val = input[i];
        output[i] = val * val + val;
    }
}

// 3. verify_results 验证函数（AI 生成）

bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, CInt n, const string& kernel_name) {
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(gpu_result[i] - cpu_result[i]) > 1e-5f) {
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

// Bad Kernel GPU 封装 (GPU，手写)
GpuTimingResult profiling_bad_gpu(CRMatrix h_input, RMatrix h_output, CInt n, CInt iterations, CInt stride) {
    PROFILE_SCOPE("Profiling Bad Version"); // Nsys Marker
    
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    CSize size_bytes = n * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_bytes)); // 不初始化output内容以允许稀疏写
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    {
        PROFILE_SCOPE("Transfer H2D");
        timerH2D.start();
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
        timerH2D.stop();
        result.h2d_ms = timerH2D.elapsed_ms();
    }
    
    // 与 Good Kernel 相同的线程总量（公平对比），但访存模式不同
    const dim3 block(256);
    const dim3 grid(cdiv(n, 256));
    
    // Kernel 预热
    profile_example_kernel_bad<<<grid, block>>>(d_input, d_output, n, stride);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    {
        PROFILE_SCOPE("Kernel Computation (Bad)");
        timerKernel.start();
        for (int i = 0; i < iterations; ++i) {
            profile_example_kernel_bad<<<grid, block>>>(d_input, d_output, n, stride);
        }
        timerKernel.stop();
        CUDA_CHECK_LAST();
        result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    }
    
    {
        PROFILE_SCOPE("Transfer D2H");
        timerD2H.start();
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
        timerD2H.stop();
        result.d2h_ms = timerD2H.elapsed_ms();
    }
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

// Good Kernel GPU 封装 (GPU，手写)
GpuTimingResult profiling_good_gpu(CRMatrix h_input, RMatrix h_output, CInt n, CInt iterations) {
    PROFILE_SCOPE("Profiling Good Version"); // Nsys Marker
    
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    CSize size_bytes = n * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_bytes));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    {
        PROFILE_SCOPE("Transfer H2D");
        timerH2D.start();
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_bytes, cudaMemcpyHostToDevice));
        timerH2D.stop();
        result.h2d_ms = timerH2D.elapsed_ms();
    }
    
    const dim3 block(256);
    const dim3 grid(cdiv(n, 256));
    
    // Kernel 预热
    profile_example_kernel_good<<<grid, block>>>(d_input, d_output, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    {
        PROFILE_SCOPE("Kernel Computation (Good)");
        timerKernel.start();
        for (int i = 0; i < iterations; ++i) {
            profile_example_kernel_good<<<grid, block>>>(d_input, d_output, n);
        }
        timerKernel.stop();
        CUDA_CHECK_LAST();
        result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    }
    
    {
        PROFILE_SCOPE("Transfer D2H");
        timerD2H.start();
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_bytes, cudaMemcpyDeviceToHost));
        timerD2H.stop();
        result.d2h_ms = timerD2H.elapsed_ms();
    }
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

// 6. main() 函数（部分手写，部分AI 生成）

int main() {
    CInt n = 10000000; // 一千万元素
    CInt iterations = 100;
    
    // 为Bad核专门设定的巨大步长跨度
    CInt const stride = 32;

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      Nsight Profiling 基准测试与诱捕目标\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << 256 << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    Matrix h_input(n, 2.5f);
    Matrix h_cpu_output(n, 0.0f);
    Matrix h_gpu_bad(n, 0.0f);
    Matrix h_gpu_good(n, 0.0f);

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    profile_cpu(h_input, h_cpu_output, n);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1: Bad Kernel 故意破坏 Coalesced Access
    cout << "--- GPU 版本 1: Bad Kernel (非合并访存 Stride=32) ---\n";
    GpuTimingResult res_bad = profiling_bad_gpu(h_input, h_gpu_bad, n, iterations, stride);
    cout << "H2D 传输时间：   " << setw(8) << res_bad.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_bad.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_bad.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_bad.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2: Good Kernel 恢复 Coalesced Access
    cout << "--- GPU 版本 2: Good Kernel (规范合并访存) ---\n";
    GpuTimingResult res_good = profiling_good_gpu(h_input, h_gpu_good, n, iterations);
    cout << "H2D 传输时间：   " << setw(8) << res_good.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_good.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_good.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_good.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_good_over_bad = res_bad.kernel_ms / res_good.kernel_ms;
    cout << "合并访存 vs 非合并访存 加速比：" << setprecision(2) << speedup_good_over_bad << "x\n\n";

    // 带宽计算 (两个 Kernel 处理相同数量的元素，可以公平对比)
    double bytes = n * FSIZE * 2; // 只读一次写入一次
    double bw_bad  = (bytes / 1e9) / (res_bad.kernel_ms / 1000.0);
    double bw_good = (bytes / 1e9) / (res_good.kernel_ms / 1000.0);
    cout << "Bad  Kernel 有效带宽：" << setprecision(2) << bw_bad  << " GB/s\n";
    cout << "Good Kernel 有效带宽：" << setprecision(2) << bw_good << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // 结果验证 (两个 Kernel 都处理全部 N 个元素，可以全面校验)
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_gpu_bad,  h_cpu_output, n, "Bad Kernel\t");
    bool pass2 = verify_results(h_gpu_good, h_cpu_output, n, "Good Kernel\t");

    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";
    cout << "完整性能分析步骤：\n";
    cout << "\n";
    cout << "Step 1: 已完成 - 基准数据在上方\n";
    cout << "\n";
    cout << "Step 2: Nsight Systems (系统级 Timeline)\n";
    cout << "  >> nsys profile --trace=cuda,nvtx -o nsight_timeline ./nsight_profiling\n";
    cout << "  >> nsys stats nsight_timeline.nsys-rep\n";
    cout << "  >> nsys-ui nsight_timeline.nsys-rep    # 需要显示器\n";
    cout << "\n";
    cout << "Step 3: Nsight Compute (Kernel 深度分析)\n";
    cout << "  (1) 安装: sudo apt install nsight-compute\n";
    cout << "  (2) 链接: sudo ln -sf /opt/nvidia/nsight-compute/*/ncu /usr/local/bin/ncu\n";
    cout << "            sudo ln -sf /opt/nvidia/nsight-compute/*/ncu-ui /usr/local/bin/ncu-ui\n";
    cout << "  (3) 生成: sudo ncu --set full -o nsight_metrics ./nsight_profiling\n";
    cout << "  (4) 查看: ncu-ui nsight_metrics.ncu-rep       # 需要显示器\n";
    cout << "  或者纯命令行 (无需 GUI):\n";
    cout << "  >> sudo ncu --kernel-name profile_example --launch-skip 2 --launch-count 2 ./nsight_profiling\n";
    cout << "========================================\n";

    return 0;
}
