// RoPE (Rotary Position Embedding) - 旋转位置编码 CUDA 实现
// LLaMA / GPT-NeoX / Qwen 等主流 LLM 的标配位置编码方案
//
// 核心思想：将位置信息通过旋转矩阵编码到 Q/K 向量中
//   x_rotated[2i]   = x[2i] * cos(θ_i) - x[2i+1] * sin(θ_i)
//   x_rotated[2i+1] = x[2i] * sin(θ_i) + x[2i+1] * cos(θ_i)
//   其中 θ_i = pos / (10000^(2i/d))
#include <code_abbreviation.h>

// ========================= GPU Kernel =========================

// RoPE 朴素实现（GPU kernel，手写）
// 每个线程处理一对相邻的维度 (2i, 2i+1)
__global__ void rope_naive(PFloat x, CInt seq_len, CInt num_heads, CInt head_dim, CFloat base = 10000.0f) {
    // 线程索引映射
    int pos = blockIdx.x;                     // 当前 token 位置 [0, seq_len)
    int head = blockIdx.y;                    // 当前 head [0, num_heads)
    int half_idx = threadIdx.x;               // 当前维度对的索引 [0, head_dim/2)

    if (half_idx >= head_dim / 2) return;

    // 计算旋转角度 θ_i = pos / (base^(2i/d))
    float freq = 1.0f / powf(base, (2.0f * half_idx) / head_dim);
    float theta = pos * freq;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    // 定位到当前 (pos, head) 对应的向量起始位置
    int offset = (pos * num_heads + head) * head_dim;
    int idx0 = offset + 2 * half_idx;
    int idx1 = offset + 2 * half_idx + 1;

    // 旋转
    float x0 = x[idx0];
    float x1 = x[idx1];
    x[idx0] = x0 * cos_theta - x1 * sin_theta;
    x[idx1] = x0 * sin_theta + x1 * cos_theta;
}

// RoPE 向量化实现（GPU kernel，手写）
// 使用 float2 一次读写两个相邻元素，减少全局内存事务
__global__ void rope_vectorized(PFloat x, CInt seq_len, CInt num_heads, CInt head_dim, CFloat base = 10000.0f) {
    int pos = blockIdx.x;
    int head = blockIdx.y;
    int half_idx = threadIdx.x;

    if (half_idx >= head_dim / 2) return;

    float freq = 1.0f / powf(base, (2.0f * half_idx) / head_dim);
    float theta = pos * freq;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    int offset = (pos * num_heads + head) * head_dim;
    // 使用 float2 进行合并读写
    float2* x2 = reinterpret_cast<float2*>(&x[offset]);
    float2 val = x2[half_idx];

    float2 rotated;
    rotated.x = val.x * cos_theta - val.y * sin_theta;
    rotated.y = val.x * sin_theta + val.y * cos_theta;
    x2[half_idx] = rotated;
}

// ========================= CPU 参考实现 =========================

// RoPE CPU 实现（手写）
void rope_cpu(PFloat x, CInt seq_len, CInt num_heads, CInt head_dim, CFloat base = 10000.0f) {
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int head = 0; head < num_heads; ++head) {
            int offset = (pos * num_heads + head) * head_dim;
            for (int i = 0; i < head_dim / 2; ++i) {
                float freq = 1.0f / powf(base, (2.0f * i) / head_dim);
                float theta = pos * freq;
                float cos_theta = cosf(theta);
                float sin_theta = sinf(theta);

                float x0 = x[offset + 2 * i];
                float x1 = x[offset + 2 * i + 1];
                x[offset + 2 * i]     = x0 * cos_theta - x1 * sin_theta;
                x[offset + 2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
            }
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

// RoPE GPU 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult rope_gpu(Matrix& h_x, CInt seq_len, CInt num_heads, CInt head_dim,
                         CInt iterations, KernelFunc kernel) {
    CSize total_elements = seq_len * num_heads * head_dim;
    CSize size_bytes = total_elements * FSIZE;

    PFloat d_x = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_x, size_bytes));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_bytes, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    // Grid: (seq_len, num_heads), Block: (head_dim / 2)
    const dim3 grid(seq_len, num_heads);
    const dim3 block(head_dim / 2);

    // 预热
    kernel<<<grid, block>>>(d_x, seq_len, num_heads, head_dim, 10000.0f);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // 每次迭代前需要重新拷贝数据（RoPE 是 in-place 操作）
    float total_kernel_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size_bytes, cudaMemcpyHostToDevice));
        timerKernel.start();
        kernel<<<grid, block>>>(d_x, seq_len, num_heads, head_dim, 10000.0f);
        timerKernel.stop();
        total_kernel_ms += timerKernel.elapsed_ms();
    }
    CUDA_CHECK_LAST();
    result.kernel_ms = total_kernel_ms / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, size_bytes, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_x));
    return result;
}

// ========================= 主函数 =========================

// 主函数（部分手写，部分AI 生成）
int main() {
    // LLaMA-7B 典型配置
    CInt seq_len = 2048;
    CInt num_heads = 32;
    CInt head_dim = 128;  // 必须为偶数
    CInt iterations = 50;
    CInt total_elements = seq_len * num_heads * head_dim;
    CSize size_bytes = total_elements * FSIZE;
    const double size_mb = size_bytes / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "   RoPE 位置编码 性能基准测试\n";
    cout << "========================================\n";
    cout << "序列长度：" << seq_len << "\n";
    cout << "Head 数量：" << num_heads << "\n";
    cout << "Head 维度：" << head_dim << "\n";
    cout << "数据大小：" << fixed << setprecision(2) << size_mb << " MB\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 生成输入数据
    Matrix h_x_cpu(total_elements);
    srand(42);
    for (int i = 0; i < total_elements; ++i) {
        h_x_cpu[i] = static_cast<float>(rand() % 1000) / 1000.0f - 0.5f;
    }
    Matrix h_x_naive = h_x_cpu;  // 拷贝用于 GPU naive
    Matrix h_x_vec = h_x_cpu;    // 拷贝用于 GPU vectorized

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    rope_cpu(h_x_cpu.data(), seq_len, num_heads, head_dim);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：" << setw(8) << cpu_time_ms << " ms\n\n";

    // GPU 版本 1：Naive RoPE
    cout << "--- GPU 版本 1: Naive RoPE ---\n";
    GpuTimingResult res_naive = rope_gpu(h_x_naive, seq_len, num_heads, head_dim, iterations, rope_naive);
    cout << "Kernel 执行时间：" << setw(8) << res_naive.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "加速比：" << setprecision(1) << cpu_time_ms / res_naive.kernel_ms << "x\n\n";

    // GPU 版本 2：Vectorized RoPE
    cout << "--- GPU 版本 2: Vectorized RoPE (float2) ---\n";
    GpuTimingResult res_vec = rope_gpu(h_x_vec, seq_len, num_heads, head_dim, iterations, rope_vectorized);
    cout << "Kernel 执行时间：" << setw(8) << res_vec.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "加速比 vs CPU：" << setprecision(1) << cpu_time_ms / res_vec.kernel_ms << "x\n";
    cout << "加速比 vs Naive：" << setprecision(2) << res_naive.kernel_ms / res_vec.kernel_ms << "x\n\n";

    // 有效带宽分析
    double bytes = 2.0 * total_elements * FSIZE;  // 读+写
    double bw_naive = (bytes / 1e9) / (res_naive.kernel_ms / 1000.0);
    double bw_vec = (bytes / 1e9) / (res_vec.kernel_ms / 1000.0);
    cout << "--- 带宽分析 ---\n";
    cout << "Naive 有效带宽：  " << setprecision(2) << bw_naive << " GB/s\n";
    cout << "Vector 有效带宽： " << setprecision(2) << bw_vec << " GB/s\n\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_x_naive, h_x_cpu, "Naive RoPE");
    bool pass2 = verify_results(h_x_vec, h_x_cpu, "Vectorized RoPE");

    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";
    return 0;
}
