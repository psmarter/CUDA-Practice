// RMSNorm (Root Mean Square Normalization) - LLaMA 系列标配归一化层
//
// 公式：RMSNorm(x) = x / sqrt(mean(x^2) + ε) * γ
//   其中 γ 是可学习的缩放参数（weight）
//
// 与 LayerNorm 的区别：RMSNorm 不做去均值操作，计算量更小约 30%
// 被 LLaMA、Qwen、Mistral 等主流大模型采用
#include <code_abbreviation.h>

// ========================= GPU Kernel =========================

// RMSNorm 朴素实现（GPU kernel，手写）
// 每个 Block 处理一行（一个 token 的隐藏层向量）
__global__ void rmsnorm_naive(CPFloat x, CPFloat weight, PFloat out,
                               CInt num_tokens, CInt hidden_dim, CFloat eps) {
    int row = blockIdx.x;
    if (row >= num_tokens) return;

    CPFloat x_row = x + row * hidden_dim;
    PFloat out_row = out + row * hidden_dim;

    // 1. 计算 mean(x^2)：单线程循环求和
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        sum_sq += x_row[i] * x_row[i];
    }
    float rms = rsqrtf(sum_sq / hidden_dim + eps);

    // 2. 归一化 + 缩放
    for (int i = 0; i < hidden_dim; ++i) {
        out_row[i] = x_row[i] * rms * weight[i];
    }
}

// RMSNorm Warp 级优化（GPU kernel，手写）
// 使用 warp shuffle 进行归约，支持 hidden_dim <= 1024
__global__ void rmsnorm_warp(CPFloat x, CPFloat weight, PFloat out,
                              CInt num_tokens, CInt hidden_dim, CFloat eps) {
    int row = blockIdx.x;
    if (row >= num_tokens) return;

    CPFloat x_row = x + row * hidden_dim;
    PFloat out_row = out + row * hidden_dim;

    extern __shared__ float sdata[];

    // 1. 每个线程负责若干元素的平方和
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = x_row[i];
        sum_sq += val * val;
    }

    // Warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    // 跨 Warp 归约，通过 Shared Memory
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int num_warps = blockDim.x / warpSize;

    if (lane_id == 0) {
        sdata[warp_id] = sum_sq;
    }
    __syncthreads();

    // 第一个 Warp 做最终归约
    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }
    }

    // 广播 RMS 因子
    if (threadIdx.x == 0) {
        sdata[0] = rsqrtf(sum_sq / hidden_dim + eps);
    }
    __syncthreads();
    float rms = sdata[0];

    // 2. 归一化 + 缩放
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        out_row[i] = x_row[i] * rms * weight[i];
    }
}

// ========================= CPU 参考实现 =========================

// RMSNorm CPU 实现（手写）
void rmsnorm_cpu(CPFloat x, CPFloat weight, PFloat out,
                  CInt num_tokens, CInt hidden_dim, CFloat eps) {
    for (int row = 0; row < num_tokens; ++row) {
        CPFloat x_row = x + row * hidden_dim;
        PFloat out_row = out + row * hidden_dim;

        // mean(x^2)
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            sum_sq += x_row[i] * x_row[i];
        }
        float rms = 1.0f / sqrtf(sum_sq / hidden_dim + eps);

        // 归一化 + 缩放
        for (int i = 0; i < hidden_dim; ++i) {
            out_row[i] = x_row[i] * rms * weight[i];
        }
    }
}

// ========================= 验证函数 =========================

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = 1e-4f) {
    int error_count = 0;
    float max_diff = 0.0f;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float diff = fabs(gpu_result[i] - cpu_result[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > epsilon) error_count++;
    }

    if (error_count > 0) {
        cout << "✗ " << kernel_name << " FAILED: " << error_count << " 个元素超出误差阈值 (max=" << max_diff << ")\n";
        return false;
    }

    cout << "✓ " << kernel_name << " PASSED (最大误差 " << max_diff << ")\n";
    return true;
}

// ========================= GPU 封装 =========================

// RMSNorm Naive GPU 封装（GPU，手写）
GpuTimingResult rmsnorm_naive_gpu(CRMatrix h_x, CRMatrix h_weight, RMatrix h_out,
                                   CInt num_tokens, CInt hidden_dim, CInt iterations) {
    CSize size_x = num_tokens * hidden_dim * FSIZE;
    CSize size_w = hidden_dim * FSIZE;

    PFloat d_x = nullptr, d_weight = nullptr, d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_x, size_x));
    CUDA_CHECK(cudaMalloc((void**)&d_weight, size_w));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size_x));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), size_w, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 grid(num_tokens);
    const dim3 block(1);  // Naive: 单线程处理一行

    rmsnorm_naive<<<grid, block>>>(d_x, d_weight, d_out, num_tokens, hidden_dim, EPSILON);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        rmsnorm_naive<<<grid, block>>>(d_x, d_weight, d_out, num_tokens, hidden_dim, EPSILON);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size_x, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_out));
    return result;
}

// RMSNorm Warp GPU 封装（GPU，手写）
GpuTimingResult rmsnorm_warp_gpu(CRMatrix h_x, CRMatrix h_weight, RMatrix h_out,
                                  CInt num_tokens, CInt hidden_dim, CInt iterations) {
    CSize size_x = num_tokens * hidden_dim * FSIZE;
    CSize size_w = hidden_dim * FSIZE;

    PFloat d_x = nullptr, d_weight = nullptr, d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_x, size_x));
    CUDA_CHECK(cudaMalloc((void**)&d_weight, size_w));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size_x));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), size_w, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    // 使用 256 个线程并行处理一行
    CInt block_size = min(256, hidden_dim);
    const dim3 grid(num_tokens);
    const dim3 block(block_size);
    CSize smem_size = (block_size / 32 + 1) * FSIZE;

    rmsnorm_warp<<<grid, block, smem_size>>>(d_x, d_weight, d_out, num_tokens, hidden_dim, EPSILON);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        rmsnorm_warp<<<grid, block, smem_size>>>(d_x, d_weight, d_out, num_tokens, hidden_dim, EPSILON);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size_x, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_out));
    return result;
}

// ========================= 主函数 =========================

// 主函数（部分手写，部分AI 生成）
int main() {
    // LLaMA-7B 典型配置
    CInt num_tokens = 2048;   // 序列长度
    CInt hidden_dim = 4096;   // 隐藏层维度
    CInt iterations = 50;
    CInt total_elements = num_tokens * hidden_dim;
    CSize size_bytes = total_elements * FSIZE;
    const double size_mb = size_bytes / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "   RMSNorm 性能基准测试\n";
    cout << "========================================\n";
    cout << "Token 数量：" << num_tokens << "\n";
    cout << "隐藏层维度：" << hidden_dim << "\n";
    cout << "数据大小：" << fixed << setprecision(2) << size_mb << " MB\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 数据初始化
    Matrix h_x(total_elements);
    Matrix h_weight(hidden_dim);
    Matrix h_out_cpu(total_elements, 0.0f);
    Matrix h_out_naive(total_elements, 0.0f);
    Matrix h_out_warp(total_elements, 0.0f);

    srand(42);
    for (int i = 0; i < total_elements; ++i) {
        h_x[i] = static_cast<float>(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (int i = 0; i < hidden_dim; ++i) {
        h_weight[i] = 1.0f + static_cast<float>(rand() % 100) / 1000.0f;  // ~1.0
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    rmsnorm_cpu(h_x.data(), h_weight.data(), h_out_cpu.data(), num_tokens, hidden_dim, EPSILON);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：" << setw(8) << cpu_time_ms << " ms\n\n";

    // GPU 版本 1：Naive RMSNorm
    cout << "--- GPU 版本 1: Naive RMSNorm (单线程/行) ---\n";
    GpuTimingResult res_naive = rmsnorm_naive_gpu(h_x, h_weight, h_out_naive, num_tokens, hidden_dim, iterations);
    cout << "Kernel 执行时间：" << setw(8) << res_naive.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "加速比：" << setprecision(1) << cpu_time_ms / res_naive.kernel_ms << "x\n\n";

    // GPU 版本 2：Warp RMSNorm
    cout << "--- GPU 版本 2: Warp-level RMSNorm (256线程/行, warp shuffle) ---\n";
    GpuTimingResult res_warp = rmsnorm_warp_gpu(h_x, h_weight, h_out_warp, num_tokens, hidden_dim, iterations);
    cout << "Kernel 执行时间：" << setw(8) << res_warp.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "加速比 vs CPU：" << setprecision(1) << cpu_time_ms / res_warp.kernel_ms << "x\n";
    cout << "加速比 vs Naive：" << setprecision(2) << res_naive.kernel_ms / res_warp.kernel_ms << "x\n\n";

    // 带宽分析
    double bytes = 2.0 * total_elements * FSIZE + hidden_dim * FSIZE;  // 读x + 读w + 写out
    double bw_naive = (bytes / 1e9) / (res_naive.kernel_ms / 1000.0);
    double bw_warp = (bytes / 1e9) / (res_warp.kernel_ms / 1000.0);
    cout << "--- 带宽分析 ---\n";
    cout << "Naive 有效带宽：" << setprecision(2) << bw_naive << " GB/s\n";
    cout << "Warp  有效带宽：" << setprecision(2) << bw_warp << " GB/s\n\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_out_naive, h_out_cpu, "Naive RMSNorm");
    bool pass2 = verify_results(h_out_warp, h_out_cpu, "Warp RMSNorm");

    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";
    return 0;
}
