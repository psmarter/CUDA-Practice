// LayerNorm - LLM 归一化层核心算子
#include <code_abbreviation.h>

// 朴素 LayerNorm：共享内存两次 Block 归约（GPU kernel，手写）
__global__ void naive_layernorm(CPFloat input, CPFloat gamma, CPFloat beta, 
                                 PFloat output, CInt batch, CInt hidden_size) {
    __shared__ float shared_mean[BLOCK_SIZE];
    __shared__ float shared_var[BLOCK_SIZE];

    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;

    // 计算均值
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        local_sum += input[row * hidden_size + i];
    }
    shared_mean[tid] = local_sum;
    __syncthreads();

    // 归约计算全局均值
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mean[tid] += shared_mean[tid + stride];
        }
        __syncthreads();
    }
    float block_mean = shared_mean[0] / hidden_size;
    __syncthreads();

    // 计算方差
    float local_var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = input[row * hidden_size + i] - block_mean;
        local_var_sum += diff * diff;
    }
    shared_var[tid] = local_var_sum;
    __syncthreads();

    // 归约计算全局方差
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_var[tid] += shared_var[tid + stride];
        }
        __syncthreads();
    }
    float block_var = shared_var[0] / hidden_size;
    __syncthreads();

    // 计算归一化输出
    float inv_std = rsqrtf(block_var + EPSILON); // 使用硬件加速版 rsqrtf
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (input[row * hidden_size + i] - block_mean) * inv_std;
        output[row * hidden_size + i] = normalized * gamma[i] + beta[i];
    }
}

// 供 Welford 算法使用的核心结构与设备函数
struct WelfordData {
    float mean;
    float m2;
    float count;
};

// Welford 算法核心函数：合并两个 WelfordData 结构
__device__ __forceinline__ WelfordData welford_combine(WelfordData a, WelfordData b) {
    // __forceinline__ 强制内联，减少函数调用开销
    WelfordData res;
    res.count = a.count + b.count;

    if (res.count == 0.0f) {
        res.mean = 0.0f;
        res.m2 = 0.0f;
        return res;
    }

    float delta = b.mean - a.mean;
    res.mean = a.mean + delta * b.count / res.count;
    res.m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / res.count;
    return res;
}

// 融合 LayerNorm：Welford 在线算法，单次遍历求均值与方差（GPU kernel，手写）
__global__ void welford_layernorm(CPFloat input, CPFloat gamma, CPFloat beta,
                                 PFloat output, CInt batch, CInt hidden_size) {
    __shared__ WelfordData shared_data[BLOCK_SIZE];

    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;

    // 每个线程计算局部 WelfordData
    WelfordData local_data = {0.0f, 0.0f, 0.0f};
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = input[row * hidden_size + i];
        local_data.count += 1.0f;
        float delta = x - local_data.mean;
        local_data.mean += delta / local_data.count;
        local_data.m2 += delta * (x - local_data.mean);
    }
    shared_data[tid] = local_data;
    __syncthreads();

    // 归约合并 WelfordData
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = welford_combine(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    WelfordData block_data = shared_data[0];
    float block_var = block_data.m2 / hidden_size;
    float inv_std = rsqrtf(block_var + EPSILON); // 使用硬件加速版 rsqrtf
    __syncthreads();

    // 计算归一化输出
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (input[row * hidden_size + i] - block_data.mean) * inv_std;
        output[row * hidden_size + i] = normalized * gamma[i] + beta[i];
    }
}

// Warp 原语版本的 welford_combine
__device__ __forceinline__ WelfordData warp_reduce_welford(WelfordData data) {
    for (int offset = 16; offset > 0; offset /= 2) {
        WelfordData other;
        other.mean = __shfl_down_sync(0xffffffff, data.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffff, data.m2, offset);
        other.count = __shfl_down_sync(0xffffffff, data.count, offset);
        data = welford_combine(data, other);
    }
    return data;
}

// Warp-Reduce LayerNorm：基于 Warp Primitives 寄存器级归约（GPU kernel，手写）
__global__ void warp_reduce_layernorm(CPFloat input, CPFloat gamma, CPFloat beta,
                                      PFloat output, CInt batch, CInt hidden_size) {
    extern __shared__ WelfordData shared_welford[];

    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // 每个线程计算局部 WelfordData
    WelfordData local_data = {0.0f, 0.0f, 0.0f};
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x = input[row * hidden_size + i];
        local_data.count += 1.0f;
        float delta = x - local_data.mean;
        local_data.mean += delta / local_data.count;
        local_data.m2 += delta * (x - local_data.mean);
    }

    // Warp 内归约
    WelfordData warp_data = warp_reduce_welford(local_data);
    if (lane == 0) {
        shared_welford[warp_id] = warp_data;
    }
    __syncthreads();

    // Block 内归约
    WelfordData block_data = (tid < blockDim.x / 32) ? shared_welford[lane] : WelfordData{0.0f, 0.0f, 0.0f};
    if (warp_id == 0) {
        block_data = warp_reduce_welford(block_data);
    }
    if (tid == 0) shared_welford[0] = block_data;
    __syncthreads();

    block_data = shared_welford[0];

    // 计算归一化输出
    float block_var = block_data.m2 / hidden_size;
    float inv_std = rsqrtf(block_var + EPSILON); // 使用硬件加速版 rsqrtf
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (input[row * hidden_size + i] - block_data.mean) * inv_std;
        output[row * hidden_size + i] = normalized * gamma[i] + beta[i];
    }
}

// Warp-per-row LayerNorm：针对小 hidden_size 的高效策略（GPU kernel，手写）
__global__ void warp_per_row_layernorm(CPFloat input, CPFloat gamma, CPFloat beta,
                                       PFloat output, CInt batch, CInt hidden_size) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= batch) return;
    int lane = threadIdx.x % 32;

    // 每个 Warp 处理一行
    WelfordData local_data = {0.0f, 0.0f, 0.0f};
    for (int i = lane; i < hidden_size; i += 32) {
        float x = input[row * hidden_size + i];
        local_data.count += 1.0f;
        float delta = x - local_data.mean;
        local_data.mean += delta / local_data.count;
        local_data.m2 += delta * (x - local_data.mean);
    }

    // Warp 内归约
    WelfordData warp_data = warp_reduce_welford(local_data);

    // 将0号线程持有的一整行归约结果广播给同属该 Warp 的其它线程
    warp_data.mean = __shfl_sync(0xffffffff, warp_data.mean, 0);
    warp_data.m2 = __shfl_sync(0xffffffff, warp_data.m2, 0);

    // 计算归一化输出
    float block_var = warp_data.m2 / hidden_size;
    float inv_std = rsqrtf(block_var + EPSILON); // 使用硬件加速版 rsqrtf
    
    for (int i = lane; i < hidden_size; i += 32) {
        float normalized = (input[row * hidden_size + i] - warp_data.mean) * inv_std;
        output[row * hidden_size + i] = normalized * gamma[i] + beta[i];
    }
}

// LayerNorm（CPU，手写）
void layernorm_cpu(CPFloat input, CPFloat gamma, CPFloat beta,
                   PFloat output, CInt batch, CInt hidden_size) {
    for (int i = 0; i < batch; ++i) {
        // 1. 求 Mean
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += input[i * hidden_size + j];
        }
        float mean = sum / hidden_size;

        // 2. 求 Var 并且计算 norm 避免重复扫描
        float var_sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            float diff = input[i * hidden_size + j] - mean;
            var_sum += diff * diff;
            output[i * hidden_size + j] = diff; // 暂存去均值后的结果
        }
        float var = var_sum / hidden_size;
        float inv_std = 1.0f / sqrtf(var + EPSILON);

        // 3. 输出仿射变换
        for (int j = 0; j < hidden_size; ++j) {
            int idx = i * hidden_size + j;
            float norm_val = output[idx] * inv_std; // 使用暂存的结果
            output[idx] = norm_val * gamma[j] + beta[j];
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

// 通用 LayerNorm GPU 封装（GPU，部分手写）
template<typename KernelFunc>
GpuTimingResult layernorm_gpu(CRMatrix h_input, CRMatrix h_gamma, CRMatrix h_beta, RMatrix h_output,
                              CInt batch, CInt hidden_size, CInt iterations, KernelFunc kernel, bool is_warp_reduce = false) {
    PFloat d_input = nullptr;
    PFloat d_gamma = nullptr;
    PFloat d_beta = nullptr;
    PFloat d_output = nullptr;
    CSize size_inout = batch * hidden_size * FSIZE;
    CSize size_params = hidden_size * FSIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_input, size_inout));
    CUDA_CHECK(cudaMalloc((void**)&d_gamma, size_params));
    CUDA_CHECK(cudaMalloc((void**)&d_beta, size_params));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_inout));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_inout, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), size_params, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), size_params, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(BLOCK_SIZE);
    const dim3 grid(batch);

    // 对于 Warp Reduce Kernel，需要分配供每一个 Warp 使用的一块 Shared Memory 计算空间
    CSize smem_size = is_warp_reduce ? (BLOCK_SIZE / 32) * sizeof(WelfordData) : 0; 
    
    kernel<<<grid, block, smem_size>>>(d_input, d_gamma, d_beta, d_output, batch, hidden_size);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block, smem_size>>>(d_input, d_gamma, d_beta, d_output, batch, hidden_size);
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
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

// 通用 Warp-per-row LayerNorm GPU 封装（GPU，部分手写）
template<typename KernelFunc>
GpuTimingResult warp_per_row_layernorm_gpu(CRMatrix h_input, CRMatrix h_gamma, CRMatrix h_beta, RMatrix h_output,
                                           CInt batch, CInt hidden_size, CInt iterations, KernelFunc kernel) {
    PFloat d_input = nullptr;
    PFloat d_gamma = nullptr;
    PFloat d_beta = nullptr;
    PFloat d_output = nullptr;
    CSize size_inout = batch * hidden_size * FSIZE;
    CSize size_params = hidden_size * FSIZE;

    CUDA_CHECK(cudaMalloc((void**)&d_input, size_inout));
    CUDA_CHECK(cudaMalloc((void**)&d_gamma, size_params));
    CUDA_CHECK(cudaMalloc((void**)&d_beta, size_params));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_inout));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_inout, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), size_params, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), size_params, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    // Kernel 配置 - 1个 Warp 处理 1 行(序列)，1个 Block 处理若干行
    CInt warps_per_block = BLOCK_SIZE / 32;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid(cdiv(batch, warps_per_block));

    kernel<<<grid, block>>>(d_input, d_gamma, d_beta, d_output, batch, hidden_size);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_input, d_gamma, d_beta, d_output, batch, hidden_size);
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
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_output));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt batch = 128;
    CInt hidden_size = 4096;
    CInt n = batch * hidden_size;
    CInt iterations = 100;

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      LayerNorm 性能基准测试\n";
    cout << "========================================\n";
    cout << "Batch 大小：" << batch << "\n";
    cout << "Hidden Size：" << hidden_size << "\n";
    cout << "总元素数：" << n << "\n";
    cout << "单矩阵大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << BLOCK_SIZE << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    Matrix h_input(n);
    Matrix h_gamma(hidden_size);
    Matrix h_beta(hidden_size);
    Matrix h_out_cpu(n, 0.0f);
    Matrix h_out_naive(n, 0.0f);
    Matrix h_out_welford(n, 0.0f);
    Matrix h_out_warp_reduce(n, 0.0f);
    Matrix h_out_warp_per_row(n, 0.0f);

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < hidden_size; ++i) {
        h_gamma[i] = 1.0f + static_cast<float>(rand() % 100) / 1000.0f; 
        h_beta[i] = static_cast<float>(rand() % 100) / 1000.0f;         
    }

    // CPU 计算（先执行 CPU，后执行 GPU）
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    layernorm_cpu(h_input.data(), h_gamma.data(), h_beta.data(), h_out_cpu.data(), batch, hidden_size);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1: Naive LayerNorm
    cout << "--- GPU 版本 1: Naive LayerNorm (Shared Memory Reduce) ---\n";
    GpuTimingResult res_naive = layernorm_gpu(h_input, h_gamma, h_beta, h_out_naive, batch, hidden_size, iterations, naive_layernorm, false);
    cout << "H2D 传输时间：   " << setw(8) << res_naive.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_naive.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_naive.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_naive.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2: Welford LayerNorm
    cout << "--- GPU 版本 2: Welford LayerNorm (单次遍历求均值方差) ---\n";
    GpuTimingResult res_welford = layernorm_gpu(h_input, h_gamma, h_beta, h_out_welford, batch, hidden_size, iterations, welford_layernorm, false);
    cout << "H2D 传输时间：   " << setw(8) << res_welford.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_welford.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_welford.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_welford.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 3: Warp Reduce LayerNorm
    cout << "--- GPU 版本 3: Warp Reduce LayerNorm (寄存器级极致规约) ---\n";
    GpuTimingResult res_warp_reduce = layernorm_gpu(h_input, h_gamma, h_beta, h_out_warp_reduce, batch, hidden_size, iterations, warp_reduce_layernorm, true);
    cout << "H2D 传输时间：   " << setw(8) << res_warp_reduce.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_warp_reduce.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_warp_reduce.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_warp_reduce.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 4: Warp-per-row LayerNorm
    cout << "--- GPU 版本 4: Warp-per-row LayerNorm (适配小Hidden Size) ---\n";
    GpuTimingResult res_warp_per_row = warp_per_row_layernorm_gpu(h_input, h_gamma, h_beta, h_out_warp_per_row, batch, hidden_size, iterations, warp_per_row_layernorm);
    cout << "H2D 传输时间：   " << setw(8) << res_warp_per_row.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_warp_per_row.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_warp_per_row.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_warp_per_row.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_warp_reduce = cpu_time_ms / res_warp_reduce.kernel_ms;
    cout << "CPU vs Warp Reduce GPU 加速比：" << setprecision(2) << speedup_warp_reduce << "x\n";

    // 带宽计算：读 Input + 读 Gamma + 读 Beta + 写 Output
    // 假设大矩阵中 Gamma 和 Beta 仅为 Hidden_size 的常数开销缓存读取，核心有效载荷依旧是 2 * N
    double bytes_rw = 2.0 * n * FSIZE; 
    double bw_naive = (bytes_rw / 1e9) / (res_naive.kernel_ms / 1000.0);
    double bw_welford = (bytes_rw / 1e9) / (res_welford.kernel_ms / 1000.0);
    double bw_warp_reduce = (bytes_rw / 1e9) / (res_warp_reduce.kernel_ms / 1000.0);
    double bw_warp_per_row = (bytes_rw / 1e9) / (res_warp_per_row.kernel_ms / 1000.0);
    
    cout << "Naive GPU 有效带宽：        " << setprecision(2) << bw_naive << " GB/s\n";
    cout << "Welford GPU 有效带宽：      " << setprecision(2) << bw_welford << " GB/s\n";
    cout << "Warp Reduce GPU 有效带宽：  " << setprecision(2) << bw_warp_reduce << " GB/s\n";
    cout << "Warp-per-row GPU 有效带宽：" << setprecision(2) << bw_warp_per_row << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Naive LayerNorm:        " << setw(8) << setprecision(4) << res_naive.kernel_ms << " ms (基准)\n";
    cout << "Welford LayerNorm:      " << setw(8) << res_welford.kernel_ms << " ms (" << setprecision(2) << res_naive.kernel_ms / res_welford.kernel_ms << "x)\n";
    cout << "Warp Reduce LayerNorm:  " << setw(8) << res_warp_reduce.kernel_ms << " ms (" << setprecision(2) << res_naive.kernel_ms / res_warp_reduce.kernel_ms << "x)\n";
    cout << "Warp-per-row LayerNorm: " << setw(8) << res_warp_per_row.kernel_ms << " ms (" << setprecision(2) << res_naive.kernel_ms / res_warp_per_row.kernel_ms << "x)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_out_naive, h_out_cpu, "Naive LayerNorm");
    bool pass2 = verify_results(h_out_welford, h_out_cpu, "Welford LayerNorm");
    bool pass3 = verify_results(h_out_warp_reduce, h_out_cpu, "Warp Reduce LayerNorm");
    bool pass4 = verify_results(h_out_warp_per_row, h_out_cpu, "Warp-per-row LayerNorm");

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2 && pass3 && pass4) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}