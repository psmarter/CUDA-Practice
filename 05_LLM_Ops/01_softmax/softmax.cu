// Softmax - LLM 注意力机制核心算子
#include <code_abbreviation.h>

// 朴素 Softmax：两次标准的 Block 归约（Shared Memory） （GPU kernel，手写）
__global__ void naive_softmax(CPFloat input, PFloat output, CInt batch, CInt seq_len) {
    __shared__ float shared_data[BLOCK_SIZE];
    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;

    // 找到当前行的最大值
    float local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, input[row * seq_len + i]);
    }
    shared_data[tid] = local_max;
    __syncthreads();

    // 归约求最大值
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    float block_max = shared_data[0];
    __syncthreads();

    // 计算指数和
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_sum += expf(input[row * seq_len + i] - block_max);
    }
    shared_data[tid] = local_sum;
    __syncthreads();

    // 归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    float block_sum = shared_data[0];
    __syncthreads();

    // 计算 Softmax 输出
    for (int i = tid; i < seq_len; i += blockDim.x) {
        output[row * seq_len + i] = expf(input[row * seq_len + i] - block_max) / block_sum;
    }
}

// Online Softmax：单次局部遍历合并 max 和 sum （GPU kernel，手写）
__global__ void online_softmax(CPFloat input, PFloat output, CInt batch, CInt seq_len) {
    __shared__ float shared_max[BLOCK_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE];

    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;

    // 局部遍历计算 max 和 sum
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = input[row * seq_len + i];
        float new_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }
    shared_max[tid] = local_max;
    shared_sum[tid] = local_sum;
    __syncthreads();

    // 归约求全局 max 和 sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float new_max = fmaxf(shared_max[tid], shared_max[tid + stride]);
            shared_sum[tid] = shared_sum[tid] * expf(shared_max[tid] - new_max) + shared_sum[tid + stride] * expf(shared_max[tid + stride] - new_max);
            shared_max[tid] = new_max;
        }
        __syncthreads();
    }

    float block_max = shared_max[0];
    float block_sum = shared_sum[0];
    __syncthreads();

    // 计算 Softmax 输出
    for (int i = tid; i < seq_len; i += blockDim.x) {
        output[row * seq_len + i] = expf(input[row * seq_len + i] - block_max) / block_sum;
    }
}

// Warp-Reduce Softmax：基于 Warp Primitives 实现极致性能的归约（GPU kernel，手写）
__global__ void warp_reduce_softmax(CPFloat input, PFloat output, CInt batch, CInt seq_len)  {
    extern __shared__ float shared_data[];
    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;
    int lane = tid % 32; // Warp 内线程 ID
    int warp_id = tid / 32; // Warp ID

    // Warp 级归约
    float local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, input[row * seq_len + i]);
    }

    // Warp 内归约求 max
    float warp_max = warp_reduce_max(local_max);
    if (lane == 0) shared_data[warp_id] = warp_max;
    __syncthreads();

    // Block 级归约求 max
    float block_max = (tid < blockDim.x / 32) ? shared_data[lane] : -INFINITY;
    if (warp_id == 0) block_max = warp_reduce_max(block_max);
    if (tid == 0) shared_data[0] = block_max; 
    __syncthreads();
    block_max = shared_data[0];

    // 计算指数和
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_sum += __expf(input[row * seq_len + i] - block_max);
    }

    // Warp 内归约求 sum
    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) shared_data[warp_id] = warp_sum;
    __syncthreads();

    // Block 级归约求 sum
    float block_sum = (tid < blockDim.x / 32) ? shared_data[lane] : 0.0f;
    if (warp_id == 0) block_sum = warp_reduce_sum(block_sum);
    if (tid == 0) shared_data[0] = block_sum;
    __syncthreads();
    block_sum = shared_data[0];

    // 计算 Softmax 输出
    for (int i = tid; i < seq_len; i += blockDim.x) {
        output[row * seq_len + i] = __expf(input[row * seq_len + i] - block_max) / block_sum;
    }
}

// Warp-per-row Softmax：针对小 seq_len (例如 128) 的高效策略 (GPU kernel, 手写)
__global__ void warp_per_row_softmax(CPFloat input, PFloat output, CInt batch, CInt seq_len) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; // 每个 Warp 处理一行
    if (row >= batch) return;
    int lane = threadIdx.x % 32; // Warp 内线程 ID

    // 计算当前行的 max
    float local_max = -INFINITY;
    for (int i = lane; i < seq_len; i += 32) {
        local_max = fmaxf(local_max, input[row * seq_len + i]);
    }

    // Warp 内归约求 max，只有 0 号线程拿到了真实的 row_max
    float row_max = warp_reduce_max(local_max);

    // 广播 row_max 给 Warp 内所有线程（统一读取 0 号线程手里的 row_max 值，并覆盖到自己的 row_max 变量中）
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    // 计算指数和
    float local_sum = 0.0f;
    for (int i = lane; i < seq_len; i += 32) {
        local_sum += __expf(input[row * seq_len + i] - row_max);
    }

    // Warp 内归约求 sum，只有 0 号线程拿到了真实的 row_sum
    float row_sum = warp_reduce_sum(local_sum);

    // 广播 row_sum 给 Warp 内所有线程
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // 计算 Softmax 输出
    for (int i = lane; i < seq_len; i += 32) {
        output[row * seq_len + i] = __expf(input[row * seq_len + i] - row_max) / row_sum;
    }
}

// Softmax（CPU，手写）
void softmax_cpu(CPFloat input, PFloat output, CInt batch, CInt seq_len) {
    for (int i = 0; i < batch; ++i) {
        float max_val = -INFINITY;

        // 1. 找最大值
        for (int j = 0; j < seq_len; ++j) {
            max_val = fmaxf(max_val, input[i * seq_len + j]);
        }
        
        float exp_sum = 0.0f;

        // 2. 减去最大值并计算 exp，暂时存入 output 数组中（避免第三个循环再算一次开销极大的 expf）
        for (int j = 0; j < seq_len; ++j) {
            float e = expf(input[i * seq_len + j] - max_val);
            exp_sum += e;
            output[i * seq_len + j] = e; 
        }

        // 3. 统一除以 exp_sum 得到最终概率
        for (int j = 0; j < seq_len; ++j) {
            output[i * seq_len + j] /= exp_sum;
        }
    }
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, const string& kernel_name, CFloat epsilon = 1e-3f) {
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配\n";
        return false;
    }

    int error_count = 0;
    float max_diff = 0.0f;
    int max_diff_idx = 0;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        float diff = fabs(gpu_result[i] - cpu_result[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = static_cast<int>(i);
        }
        if (diff > epsilon) {
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

    cout << "✓ " << kernel_name << " PASSED: 结果验证通过 (最大误差 " << max_diff << ")\n";
    return true;
}

// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      // Host to Device 传输时间
    float kernel_ms;   // Kernel 执行时间（多次平均）
    float d2h_ms;      // Device to Host 传输时间
    float total_ms;    // 总时间
};

// 通用 Softmax GPU 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult softmax_gpu(CRMatrix h_input, RMatrix h_output, CInt batch, CInt seq_len, CInt iterations, KernelFunc kernel) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    CSize size_inout = batch * seq_len * FSIZE;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_inout));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_inout));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    // H2D
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_inout, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    // Kernel 配置 - 1个 Block 处理 1 行(序列)
    const dim3 block(BLOCK_SIZE);
    const dim3 grid(batch);

    // 计算动态 shared memory 大小 (取决于具体 kernel 需要)
    // 这里我们传入最多可能用到的共享内存大小，Warp Reduce Softmax 取决于 warps 数量
    CSize smem_size = (BLOCK_SIZE / 32) * FSIZE; 

    // 预热
    kernel<<<grid, block, smem_size>>>(d_input, d_output, batch, seq_len);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    // Kernel 计时
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block, smem_size>>>(d_input, d_output, batch, seq_len);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    // D2H
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_inout, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    // 总时间
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    // 释放
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

// 通用 Warp-per-row Softmax GPU 封装（GPU，部分手写）
template<typename KernelFunc>
GpuTimingResult warp_per_row_softmax_gpu(CRMatrix h_input, RMatrix h_output, CInt batch, CInt seq_len, CInt iterations, KernelFunc kernel) {
    PFloat d_input = nullptr;
    PFloat d_output = nullptr;
    CSize size_inout = batch * seq_len * FSIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_input, size_inout));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_inout));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_inout, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    // Kernel 配置 - 1个 Warp 处理 1 行，1个 Block 可以处理 (BLOCK_SIZE / 32) 行
    CInt warps_per_block = BLOCK_SIZE / 32;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid(cdiv(batch, warps_per_block));

    kernel<<<grid, block>>>(d_input, d_output, batch, seq_len);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_input, d_output, batch, seq_len);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_inout, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt batch = 128; // Batch size
    CInt seq_len = 4096; // 序列长度（模拟大模型上下文长序列）
    CInt n = batch * seq_len;
    CInt iterations = 100;

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      Softmax 性能基准测试\n";
    cout << "========================================\n";
    cout << "Batch 大小：" << batch << "\n";
    cout << "序列长度：" << seq_len << "\n";
    cout << "总元素数：" << n << "\n";
    cout << "单矩阵大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_input(n);
    Matrix h_out_cpu(n, 0.0f);
    Matrix h_out_naive(n, 0.0f);
    Matrix h_out_online(n, 0.0f);
    Matrix h_out_warp_reduce(n, 0.0f);
    Matrix h_out_warp_per_row(n, 0.0f);

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(rand() % 100) / 100.0f;
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    softmax_cpu(h_input.data(), h_out_cpu.data(), batch, seq_len);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1：Naive Softmax
    cout << "--- GPU 版本 1: Naive Softmax (Shared Memory Reduce) ---\n";
    GpuTimingResult res_naive = softmax_gpu(h_input, h_out_naive, batch, seq_len, iterations, naive_softmax);
    cout << "H2D 传输时间：   " << setw(8) << res_naive.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_naive.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_naive.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_naive.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2：Online Softmax
    cout << "--- GPU 版本 2: Online Softmax (Single-pass Reduce) ---\n";
    GpuTimingResult res_online = softmax_gpu(h_input, h_out_online, batch, seq_len, iterations, online_softmax);
    cout << "H2D 传输时间：   " << setw(8) << res_online.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_online.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_online.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_online.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 3：Warp Reduce Softmax
    cout << "--- GPU 版本 3: Warp Reduce Softmax (原 Fused Softmax) ---\n";
    GpuTimingResult res_warp_reduce = softmax_gpu(h_input, h_out_warp_reduce, batch, seq_len, iterations, warp_reduce_softmax);
    cout << "H2D 传输时间：   " << setw(8) << res_warp_reduce.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_warp_reduce.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_warp_reduce.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_warp_reduce.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 4：Warp-per-row Softmax
    cout << "--- GPU 版本 4: Warp-per-row Softmax ---\n";
    GpuTimingResult res_warp_per_row = warp_per_row_softmax_gpu(h_input, h_out_warp_per_row, batch, seq_len, iterations, warp_per_row_softmax);
    cout << "H2D 传输时间：   " << setw(8) << res_warp_per_row.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_warp_per_row.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_warp_per_row.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_warp_per_row.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_naive = cpu_time_ms / res_naive.kernel_ms;
    double speedup_warp_reduce = cpu_time_ms / res_warp_reduce.kernel_ms;
    cout << "CPU vs Warp Reduce GPU 加速比：" << setprecision(2) << speedup_warp_reduce << "x\n";

    double bytes_rw = 2.0 * n * FSIZE; // 读入 Input + 写出 Output
    double bw_naive = (bytes_rw / 1e9) / (res_naive.kernel_ms / 1000.0);
    double bw_warp_reduce = (bytes_rw / 1e9) / (res_warp_reduce.kernel_ms / 1000.0);
    double bw_warp_per_row = (bytes_rw / 1e9) / (res_warp_per_row.kernel_ms / 1000.0);
    cout << "Naive GPU 有效带宽：        " << setprecision(2) << bw_naive << " GB/s\n";
    cout << "Warp Reduce GPU 有效带宽：  " << setprecision(2) << bw_warp_reduce << " GB/s\n";
    cout << "Warp-per-row GPU 有效带宽：" << setprecision(2) << bw_warp_per_row << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Naive Softmax:        " << setw(8) << setprecision(4) << res_naive.kernel_ms << " ms (基准)\n";
    cout << "Online Softmax:       " << setw(8) << res_online.kernel_ms << " ms ("
         << setprecision(2) << res_naive.kernel_ms / res_online.kernel_ms << "x)\n";
    cout << "Warp Reduce Softmax:  " << setw(8) << res_warp_reduce.kernel_ms << " ms ("
         << setprecision(2) << res_naive.kernel_ms / res_warp_reduce.kernel_ms << "x)\n";
    cout << "Warp-per-row Softmax: " << setw(8) << res_warp_per_row.kernel_ms << " ms ("
         << setprecision(2) << res_naive.kernel_ms / res_warp_per_row.kernel_ms << "x)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_out_naive, h_out_cpu, "Naive Softmax");
    bool pass2 = verify_results(h_out_online, h_out_cpu, "Online Softmax");
    bool pass3 = verify_results(h_out_warp_reduce, h_out_cpu, "Warp Reduce Softmax");
    bool pass4 = verify_results(h_out_warp_per_row, h_out_cpu, "Warp-per-row Softmax");

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2 && pass3 && pass4) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
