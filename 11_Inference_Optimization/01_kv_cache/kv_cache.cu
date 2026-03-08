// KV Cache - LLM 推理中的键值缓存管理
#include <code_abbreviation.h>
#include <random>

// KV Cache 配置
struct KVCacheConfig {
    int batch_size;
    int num_heads;
    int head_dim;
    int max_seq_len;
    int block_size;  // PagedAttention 块大小
};


// 朴素 Attention：直接在连续内存上读取KV Cache进行点积 (GPU Kernel，手写)
__global__ void naive_attention_kernel(
    CPFloat query,       // [batch, num_heads, head_dim]
    CPFloat k_cache,     // [batch, num_heads, max_seq_len, head_dim]
    CPFloat v_cache,     // [batch, num_heads, max_seq_len, head_dim]
    int* seq_lens,       // [batch] 每个序列的当前实际长度
    PFloat output,       // [batch, num_heads, head_dim]
    CInt batch_size, CInt num_heads, CInt head_dim, CInt max_seq_len) {
    
    int head_idx = blockIdx.x % num_heads;
    int batch_idx = blockIdx.x / num_heads;
    int tid = threadIdx.x; // 处理 head_dim
    
    if (batch_idx >= batch_size || head_idx >= num_heads || tid >= head_dim) return;

    int seq_len = seq_lens[batch_idx];
    float q_val = query[batch_idx * num_heads * head_dim + head_idx * head_dim + tid];
    
    // 简化版：仅对 K 计算 dot product 并寻找最大值或简单求和作为 Attention 示意
    // 真实场景是：S = Q * K^T, P = Softmax(S), O = P * V
    // 这里我们做一个简化的加权和来代表读写 K,V 的访存模式
    
    float acc = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        int kv_idx = batch_idx * (num_heads * max_seq_len * head_dim) + 
                     head_idx * (max_seq_len * head_dim) + 
                     i * head_dim + 
                     tid;
        float k_val = k_cache[kv_idx];
        float v_val = v_cache[kv_idx];
        
        // 假装 q*k 作为权重乘在 v 上
        acc += (q_val * k_val) * v_val; 
    }
    
    output[batch_idx * num_heads * head_dim + head_idx * head_dim + tid] = acc;
}

// Paged KV Cache 读取 kernel (PagedAttention) (GPU Kernel，手写)
__global__ void paged_attention_kernel(
    CPFloat query,       // [batch, num_heads, head_dim]
    float** k_blocks,    // 块指针数组 (物理块)
    float** v_blocks,
    int* block_table,    // [batch, max_blocks_per_seq] 逻辑到物理块映射
    int* seq_lens,       // [batch]
    PFloat output,       // [batch, num_heads, head_dim]
    CInt batch_size, CInt num_heads, CInt head_dim,
    CInt block_size, CInt max_blocks_per_seq) {
    
    int head_idx = blockIdx.x % num_heads;
    int batch_idx = blockIdx.x / num_heads;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || tid >= head_dim) return;

    int seq_len = seq_lens[batch_idx];
    float q_val = query[batch_idx * num_heads * head_dim + head_idx * head_dim + tid];
    
    float acc = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        // PagedAttention 核心逻辑：计算逻辑块号和块内偏移
        int logical_block_idx = i / block_size;
        int block_offset = i % block_size;
        
        // 查表获取物理块号
        int physical_block_idx = block_table[batch_idx * max_blocks_per_seq + logical_block_idx];
        
        // 获取物理块指针
        float* k_block = k_blocks[physical_block_idx];
        float* v_block = v_blocks[physical_block_idx];
        
        // 计算块内数据的准确索引：[num_heads, block_size, head_dim]
        int element_idx = head_idx * (block_size * head_dim) + 
                          block_offset * head_dim + 
                          tid;
                          
        float k_val = k_block[element_idx];
        float v_val = v_block[element_idx];
        
        acc += (q_val * k_val) * v_val;
    }
    
    output[batch_idx * num_heads * head_dim + head_idx * head_dim + tid] = acc;
}


// 朴素 KV Cache CPU 实现（CPU，手写）
void naive_attention_cpu(
    CRMatrix query, CRMatrix k_cache, CRMatrix v_cache, const vector<int>& seq_lens, RMatrix output,
    CInt batch_size, CInt num_heads, CInt head_dim, CInt max_seq_len) {
    
    for (int b = 0; b < batch_size; ++b) {
        int seq_len = seq_lens[b];
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                float q_val = query[b * num_heads * head_dim + h * head_dim + d];
                float acc = 0.0f;
                for (int i = 0; i < seq_len; ++i) {
                    int kv_idx = b * (num_heads * max_seq_len * head_dim) + 
                                 h * (max_seq_len * head_dim) + 
                                 i * head_dim + 
                                 d;
                    float k_val = k_cache[kv_idx];
                    float v_val = v_cache[kv_idx];
                    acc += (q_val * k_val) * v_val;
                }
                output[b * num_heads * head_dim + h * head_dim + d] = acc;
            }
        }
    }
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, CInt n, const string& kernel_name) {
    bool success = true;
    for (int i = 0; i < n; ++i) {
        // 由于累加误差较大，适当放宽 EPSILON
        if (std::abs(gpu_result[i] - cpu_result[i]) > 1e-3f) {
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

// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      // Host to Device 传输时间
    float kernel_ms;   // Kernel 执行时间（多次平均）
    float d2h_ms;      // Device to Host 传输时间
    float total_ms;    // 总时间
};


// Naive GPU 封装 (GPU，手写)
template<typename KernelFunc>
GpuTimingResult naive_gpu(
    CRMatrix h_query, CRMatrix h_k_cache, CRMatrix h_v_cache, const vector<int>& h_seq_lens,
    RMatrix h_output, CInt batch_size, CInt num_heads, CInt head_dim, CInt max_seq_len,
    CInt iterations, KernelFunc kernel) {
    
    PFloat d_query = nullptr, d_k_cache = nullptr, d_v_cache = nullptr, d_output = nullptr;
    int* d_seq_lens;
    
    CInt elem_q = batch_size * num_heads * head_dim;
    CInt elem_kv = batch_size * num_heads * max_seq_len * head_dim;
    
    CUDA_CHECK(cudaMalloc((void**)&d_query, elem_q * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_k_cache, elem_kv * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_v_cache, elem_kv * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_output, elem_q * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_seq_lens, batch_size * sizeof(int)));
    
    CUDA_CHECK(cudaMemset(d_output, 0, elem_q * FSIZE));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), elem_q * FSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), elem_kv * FSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), elem_kv * FSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens, h_seq_lens.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const dim3 block(head_dim); // 每个线程处理一个维度
    const dim3 grid(batch_size * num_heads); // 每个 block 处理一个头
    
    kernel<<<grid, block>>>(d_query, d_k_cache, d_v_cache, d_seq_lens, d_output, batch_size, num_heads, head_dim, max_seq_len);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_query, d_k_cache, d_v_cache, d_seq_lens, d_output, batch_size, num_heads, head_dim, max_seq_len);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, elem_q * FSIZE, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_seq_lens));
    
    return result;
}

// Paged GPU 封装 (GPU，手写)
template<typename KernelFunc>
GpuTimingResult paged_gpu(
    CRMatrix h_query, 
    float** h_k_blocks, float** h_v_blocks, int total_physical_blocks, CInt block_size_bytes,
    const vector<int>& h_block_table, const vector<int>& h_seq_lens,
    RMatrix h_output, CInt batch_size, CInt num_heads, CInt head_dim, CInt block_size, CInt max_blocks_per_seq,
    CInt iterations, KernelFunc kernel) {
    
    PFloat d_query, d_output;
    int *d_block_table, *d_seq_lens;
    
    // 我们需要在 GPU 上构建指针数组
    float** d_k_blocks_ptrs;
    float** d_v_blocks_ptrs;
    CUDA_CHECK(cudaMalloc((void**)&d_k_blocks_ptrs, total_physical_blocks * sizeof(float*)));
    CUDA_CHECK(cudaMalloc((void**)&d_v_blocks_ptrs, total_physical_blocks * sizeof(float*)));
    
    // 给每个物理块分配显存并在 CPU 上装配设备指针列表
    vector<float*> dev_k_ptrs(total_physical_blocks);
    vector<float*> dev_v_ptrs(total_physical_blocks);
    
    CInt elem_q = batch_size * num_heads * head_dim;
    
    CUDA_CHECK(cudaMalloc((void**)&d_query, elem_q * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_output, elem_q * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_block_table, batch_size * max_blocks_per_seq * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_seq_lens, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_output, 0, elem_q * FSIZE));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    // 拷贝物理块数据并挂载指针
    for (int i = 0; i < total_physical_blocks; ++i) {
        CUDA_CHECK(cudaMalloc((void**)&dev_k_ptrs[i], block_size_bytes));
        CUDA_CHECK(cudaMalloc((void**)&dev_v_ptrs[i], block_size_bytes));
        CUDA_CHECK(cudaMemcpy(dev_k_ptrs[i], h_k_blocks[i], block_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_v_ptrs[i], h_v_blocks[i], block_size_bytes, cudaMemcpyHostToDevice));
    }
    // 拷贝装配好的指针数组到 GPU
    CUDA_CHECK(cudaMemcpy(d_k_blocks_ptrs, dev_k_ptrs.data(), total_physical_blocks * sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_blocks_ptrs, dev_v_ptrs.data(), total_physical_blocks * sizeof(float*), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), elem_q * FSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table.data(), batch_size * max_blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens, h_seq_lens.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const dim3 block(head_dim);
    const dim3 grid(batch_size * num_heads);
    
    kernel<<<grid, block>>>(d_query, d_k_blocks_ptrs, d_v_blocks_ptrs, d_block_table, d_seq_lens, d_output, 
                            batch_size, num_heads, head_dim, block_size, max_blocks_per_seq);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_query, d_k_blocks_ptrs, d_v_blocks_ptrs, d_block_table, d_seq_lens, d_output, 
                                batch_size, num_heads, head_dim, block_size, max_blocks_per_seq);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, elem_q * FSIZE, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    // 释放内存
    for (int i = 0; i < total_physical_blocks; ++i) {
        CUDA_CHECK(cudaFree(dev_k_ptrs[i]));
        CUDA_CHECK(cudaFree(dev_v_ptrs[i]));
    }
    CUDA_CHECK(cudaFree(d_k_blocks_ptrs));
    CUDA_CHECK(cudaFree(d_v_blocks_ptrs));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_block_table));
    CUDA_CHECK(cudaFree(d_seq_lens));
    
    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    KVCacheConfig config;
    config.batch_size = 32;
    config.num_heads = 16;
    config.head_dim = 64;       // 保证等于甚至小于 blockDim 极限 (这里64 <= 1024)
    config.max_seq_len = 2048;  // 最大序列长度
    config.block_size = 16;     // 每个块存放 16 个 token
    
    CInt iterations = 100;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    CInt batch_size = config.batch_size;
    CInt num_heads = config.num_heads;
    CInt head_dim = config.head_dim;
    CInt max_seq_len = config.max_seq_len;
    CInt block_size = config.block_size;
    CInt max_blocks_per_seq = cdiv(max_seq_len, block_size);

    CInt elem_q = batch_size * num_heads * head_dim;
    CInt elem_kv_naive = batch_size * num_heads * max_seq_len * head_dim;
    
    const double naive_mb = (2.0 * elem_kv_naive * FSIZE) / (1024.0 * 1024.0); // K and V
    
    printDeviceInfo();

    // 构建真实的测试集：动态序列长度以暴露出 Naive 架构浪费显存的问题
    vector<int> seq_lens(batch_size);
    int total_actual_tokens = 0;
    for (int i = 0; i < batch_size; ++i) {
        seq_lens[i] = 128 + (gen() % 1920); // 随机长度 [128, 2048)
        total_actual_tokens += seq_lens[i];
    }
    
    // 计算 Paged 实际使用显存
    int total_physical_blocks = 0;
    vector<int> block_table(batch_size * max_blocks_per_seq, -1);
    
    for (int b = 0; b < batch_size; ++b) {
        int req_blocks = cdiv(seq_lens[b], block_size);
        for (int i = 0; i < req_blocks; ++i) {
            block_table[b * max_blocks_per_seq + i] = total_physical_blocks++;
        }
    }
    
    CInt block_size_bytes = num_heads * block_size * head_dim * FSIZE; // 每个物理块包含所有头的该 token 切片
    const double paged_mb = (2.0 * total_physical_blocks * block_size_bytes) / (1024.0 * 1024.0);
    
    cout << "========================================\n";
    cout << "      KV Cache 内存管理优化基准测试\n";
    cout << "========================================\n";
    cout << "Batch Size：" << batch_size << "\n";
    cout << "Num Heads：" << num_heads << "  Head Dim：" << head_dim << "\n";
    cout << "最大序列长度：" << max_seq_len << "  Block 大小：" << block_size << " 个 Token\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";
    cout << "--- 理论显存占用对比 ---\n";
    cout << "所有张量按完整长度 (" << max_seq_len << ") 预分配 (Naive):\n";
    cout << "> 预估 KV Cache 大小: " << fixed << setprecision(2) << naive_mb << " MB\n";
    cout << "使用分块按需分布并消除碎片 (Paged Attention):\n";
    cout << "> 预估 KV Cache 大小: " << fixed << setprecision(2) << paged_mb << " MB\n";
    cout << "> 节省显存: " << setprecision(2) << (1.0f - (paged_mb/naive_mb))*100.0f << "%\n";
    cout << "\n";

    // 内存初始化 (CPU 模拟)
    Matrix h_query(elem_q);
    for (int i = 0; i < elem_q; ++i) h_query[i] = dist(gen);
    
    Matrix h_k_naive(elem_kv_naive), h_v_naive(elem_kv_naive);
    for (int i = 0; i < elem_kv_naive; ++i) {
        h_k_naive[i] = dist(gen);
        h_v_naive[i] = dist(gen);
    }
    
    // 我们必须将生成的 Naive 数据映射到 Paged 模型中以便进行严格的正确性比对 (verify_result)
    vector<float*> h_k_blocks(total_physical_blocks);
    vector<float*> h_v_blocks(total_physical_blocks);
    
    int elements_per_block = num_heads * block_size * head_dim;
    for (int i = 0; i < total_physical_blocks; ++i) {
        h_k_blocks[i] = new float[elements_per_block];
        h_v_blocks[i] = new float[elements_per_block];
    }
    
    // 数据同步：将 Native 数据精准按照 Paged 格式打包装填
    for (int b = 0; b < batch_size; ++b) {
        int seq_len = seq_lens[b];
        for (int i = 0; i < seq_len; ++i) {
            int logical_block = i / block_size;
            int offset = i % block_size;
            int phys_block = block_table[b * max_blocks_per_seq + logical_block];
            
            for (int h = 0; h < num_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    // Naive Index
                    int naive_idx = b * (num_heads * max_seq_len * head_dim) + 
                                    h * (max_seq_len * head_dim) + 
                                    i * head_dim + d;
                                    
                    // Paged Index
                    int paged_idx = h * (block_size * head_dim) +
                                    offset * head_dim + d;
                                    
                    h_k_blocks[phys_block][paged_idx] = h_k_naive[naive_idx];
                    h_v_blocks[phys_block][paged_idx] = h_v_naive[naive_idx];
                }
            }
        }
    }
    
    Matrix h_cpu_output(elem_q, 0.0f);
    Matrix h_gpu_naive_output(elem_q, 0.0f);
    Matrix h_gpu_paged_output(elem_q, 0.0f);

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    naive_attention_cpu(h_query, h_k_naive, h_v_naive, seq_lens, h_cpu_output, batch_size, num_heads, head_dim, max_seq_len);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1：Naive 连续内存
    cout << "--- GPU 版本 1: Naive (静态分配连续内存) ---\n";
    GpuTimingResult res_naive = naive_gpu(h_query, h_k_naive, h_v_naive, seq_lens, h_gpu_naive_output,
                                         batch_size, num_heads, head_dim, max_seq_len, iterations, naive_attention_kernel);
    cout << "H2D 传输时间：   " << setw(8) << res_naive.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_naive.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_naive.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_naive.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2：Paged 按需分配块列表
    cout << "--- GPU 版本 2: PagedAttention 机制 ---\n";
    GpuTimingResult res_paged = paged_gpu(h_query, h_k_blocks.data(), h_v_blocks.data(), total_physical_blocks, block_size_bytes,
                                          block_table, seq_lens, h_gpu_paged_output, batch_size, num_heads, head_dim, block_size,
                                          max_blocks_per_seq, iterations, paged_attention_kernel);
    cout << "H2D 传输时间：   " << setw(8) << res_paged.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_paged.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_paged.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_paged.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_naive = cpu_time_ms / res_naive.kernel_ms;
    double speedup_paged = cpu_time_ms / res_paged.kernel_ms;
    cout << "CPU vs GPU (Naive) 加速比：" << setprecision(2) << speedup_naive << "x\n";
    cout << "CPU vs GPU (Paged) 加速比：" << setprecision(2) << speedup_paged << "x\n";

    // 带宽计算 (依据实际被接触的元素而不是总分配尺寸计算)
    double actual_bytes_accessed = (double)total_actual_tokens * num_heads * head_dim * FSIZE * 2.0; // KV cache readout per token + writes
    double gpu_bandwidth_naive = (actual_bytes_accessed / 1e9) / (res_naive.kernel_ms / 1000.0);
    double gpu_bandwidth_paged = (actual_bytes_accessed / 1e9) / (res_paged.kernel_ms / 1000.0);
    
    cout << "Naive 有效带宽：" << setw(8) << setprecision(2) << gpu_bandwidth_naive << " GB/s\n";
    cout << "Paged 有效带宽：" << setw(8) << setprecision(2) << gpu_bandwidth_paged << " GB/s\n";
    cout << "性能对比差异  ：Paged 相比较 Naive 耗时 " << setprecision(2) << (res_paged.kernel_ms / res_naive.kernel_ms) << "x (主要来自指针解引用的开销)\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_gpu_naive_output, h_cpu_output, elem_q, "Naive Attention");
    bool pass2 = verify_results(h_gpu_paged_output, h_cpu_output, elem_q, "Paged Attention");

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    //清理 CPU 端动态分配的数据
    for (int i = 0; i < total_physical_blocks; ++i) {
        delete[] h_k_blocks[i];
        delete[] h_v_blocks[i];
    }
    return 0;
}
