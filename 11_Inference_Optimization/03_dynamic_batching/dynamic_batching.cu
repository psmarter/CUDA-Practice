// Dynamic Batching - 动态批处理 / 连续批处理 (Continuous Batching)
#include <code_abbreviation.h>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <random>
#include <algorithm>


// 传统静态 Batching (Padding to Max Length)
// 假设输入已经被 Padding 到了 [batch_size, max_seq_len, head_dim]
__global__ void batched_attention_fixed(
    CPFloat query,        
    CPFloat key,        
    CPFloat value,
    const int* seq_lens,   // [batch_size] 用于 Masking
    PFloat output,
    CInt batch_size, CInt max_seq_len, CInt num_heads, CInt head_dim) {
    
    int batch_idx = blockIdx.x / num_heads;
    int head_idx = blockIdx.x % num_heads;
    int tid = threadIdx.x; // 处理 head_dim
    
    if (batch_idx >= batch_size || head_idx >= num_heads || tid >= head_dim) return;

    int actual_len = seq_lens[batch_idx];
    
    // 如果当前正在处理的序列位置超过了实际长度（即这部分是 Padding 数据），则进行大量无用计算或分支掩码
    // 这里简单示意一个计算流程，对于所有的 T 都去跑，但在真实场景中对超出 actual_len 的位置掩码置为 -inf
    
    float q_val = query[batch_idx * num_heads * head_dim + head_idx * head_dim + tid];
    float acc = 0.0f;
    
    for (int i = 0; i < max_seq_len; ++i) { // 悲观遍历到最大长度 (Padding waste)
        if (i < actual_len) { // 有效计算
            int kv_idx = batch_idx * (num_heads * max_seq_len * head_dim) + 
                         head_idx * (max_seq_len * head_dim) + 
                         i * head_dim + tid;
            float k_val = key[kv_idx];
            float v_val = value[kv_idx];
            acc += (q_val * k_val) * v_val; 
        } else {
            // Padding 计算浪费，计算单元空转
        }
    }
    
    output[batch_idx * num_heads * head_dim + head_idx * head_dim + tid] = acc;
}

// 变长序列批处理 kernel (Continuous Batching / Var-len array)
// 使用 packed tensor + offset 方式处理不同长度的序列，消除 Padding
__global__ void batched_attention_varlen(
    CPFloat query,         // [batch_size, num_heads, head_dim] 提问向量
    CPFloat key,           // [total_tokens, num_heads, head_dim] Packed Key
    CPFloat value,         // [total_tokens, num_heads, head_dim] Packed Value
    const int* seq_starts, // [batch_size + 1] 每个序列在 packed 数组中的起始 token 偏移
    PFloat output,         // [batch_size, num_heads, head_dim] 输出
    CInt batch_size, CInt num_heads, CInt head_dim) {
    
    // 依然令一个 block 处理一个 (batch, head)
    int batch_idx = blockIdx.x / num_heads;
    int head_idx = blockIdx.x % num_heads;
    int tid = threadIdx.x; // 处理 head_dim
    
    if (batch_idx >= batch_size || head_idx >= num_heads || tid >= head_dim) return;

    // 获取当前序列在这个 packed array 中的全局 token 起止范围
    int start_token_idx = seq_starts[batch_idx];
    int end_token_idx = seq_starts[batch_idx + 1];
    
    float q_val = query[batch_idx * num_heads * head_dim + head_idx * head_dim + tid];
    float acc = 0.0f;
    
    // 仅遍历有效 token！绝对没有任何 Padding 的浪费。
    for (int token_idx = start_token_idx; token_idx < end_token_idx; ++token_idx) {
        // 由于是 Packed，所以 Key 的第一维直接是 token_idx
        int kv_idx = token_idx * (num_heads * head_dim) + 
                     head_idx * head_dim + 
                     tid;
                     
        float k_val = key[kv_idx];
        float v_val = value[kv_idx];
        
        acc += (q_val * k_val) * v_val;
    }
    
    output[batch_idx * num_heads * head_dim + head_idx * head_dim + tid] = acc;
}



// VarLen CPU
void varlen_attention_cpu(
    CRMatrix query, CRMatrix key, CRMatrix value,
    const std::vector<int>& seq_starts, RMatrix output,
    CInt batch_size, CInt num_heads, CInt head_dim) {
    
    for (int b = 0; b < batch_size; ++b) {
        int start_token = seq_starts[b];
        int end_token = seq_starts[b + 1];
        
        for (int h = 0; h < num_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                float q_val = query[b * num_heads * head_dim + h * head_dim + d];
                float acc = 0.0f;
                
                for (int t = start_token; t < end_token; ++t) {
                    int kv_idx = t * (num_heads * head_dim) + h * head_dim + d;
                    acc += (q_val * key[kv_idx]) * value[kv_idx];
                }
                
                output[b * num_heads * head_dim + h * head_dim + d] = acc;
            }
        }
    }
}

// 附加概念：投机采样（Speculative Decoding）模拟器 (运行在 CPU 端架构)
class SpeculativeDecoder {
public:
    int generated_tokens = 0;
    
    // 假设的小模型（纯逻辑示意）
    std::vector<int> draft_model_generate(int k_tokens) {
        // 在实际应用中，小模型极速生成一批备选 token
        return std::vector<int>(k_tokens, 1); // 假装全部生成了 ID 1
    }
    
    // 假设的大模型验证（由于并行性，验证 K 个 token 此处相当于一次 forward，远快于大模型自回归 K 次）
    int target_model_verify(const std::vector<int>& draft_tokens) {
        // 对比并判断哪些 token 被大模型接受。发现错误则截断。
        // （在此模拟接受率为 70% 左右）
        int accepted = 0;
        for(size_t i=0; i<draft_tokens.size(); ++i) {
            if((rand() % 100) < 70) accepted++;
            else break;
        }
        return accepted + 1; // 总是能额外产生1个必须是对的 token
    }
    
    void speculative_decode_simulate(int target_len, int max_draft_tokens) {
        while(generated_tokens < target_len) {
            // 1. 小模型生成 K 个候选
            auto drafts = draft_model_generate(max_draft_tokens);
            // 2. 大模型验证
            int valid_count = target_model_verify(drafts);
            // 3. 接受 token 步进
            generated_tokens += valid_count;
        }
    }
};

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, CInt n, const string& kernel_name) {
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(gpu_result[i] - cpu_result[i]) > 1e-3f) {
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



// Varlen GPU 封装 (GPU，手写)
template<typename KernelFunc>
GpuTimingResult varlen_gpu(
    CRMatrix h_query, CRMatrix h_key, CRMatrix h_value, const std::vector<int>& h_seq_starts,
    RMatrix h_output, CInt batch_size, CInt total_tokens, CInt num_heads, CInt head_dim,
    CInt iterations, KernelFunc kernel) {
    
    PFloat d_query = nullptr, d_key = nullptr, d_value = nullptr, d_output = nullptr;
    int* d_seq_starts = nullptr;
    
    CInt elem_q = batch_size * num_heads * head_dim;
    CInt elem_kv = total_tokens * num_heads * head_dim;
    
    CUDA_CHECK(cudaMalloc((void**)&d_query, elem_q * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_key, elem_kv * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_value, elem_kv * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_output, elem_q * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_seq_starts, (batch_size + 1) * sizeof(int)));
    
    CUDA_CHECK(cudaMemset(d_output, 0, elem_q * FSIZE));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), elem_q * FSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, h_key.data(), elem_kv * FSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), elem_kv * FSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_starts, h_seq_starts.data(), (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const dim3 block(head_dim); 
    const dim3 grid(batch_size * num_heads); 
    
    kernel<<<grid, block>>>(d_query, d_key, d_value, d_seq_starts, d_output, batch_size, num_heads, head_dim);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_query, d_key, d_value, d_seq_starts, d_output, batch_size, num_heads, head_dim);
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
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_seq_starts));
    
    return result;
}

// 纯测试用的 Fixed 长度 GPU 封装，仅提供相同的参数签名来公平对比 (GPU，手写)
GpuTimingResult fixed_gpu(
    CRMatrix h_query, CRMatrix h_key, CRMatrix h_value, const std::vector<int>& h_seq_lens,
    RMatrix h_output, CInt batch_size, CInt max_seq_len, CInt num_heads, CInt head_dim,
    CInt iterations) {
    
    PFloat d_query = nullptr, d_key = nullptr, d_value = nullptr, d_output = nullptr;
    int* d_seq_lens = nullptr;
    
    CInt elem_q = batch_size * num_heads * head_dim;
    CInt elem_kv = batch_size * max_seq_len * num_heads * head_dim; // 这里的显存分配必须按最大长度进行 Full Pad
    
    CUDA_CHECK(cudaMalloc((void**)&d_query, elem_q * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_key, elem_kv * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_value, elem_kv * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_output, elem_q * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_seq_lens, batch_size * sizeof(int)));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), elem_q * FSIZE, cudaMemcpyHostToDevice));
    // 直接把打包的数据装载进去充当 pad 数据（实战里这通常伴随着大量零值，但不影响我们测算性能和时间）
    CUDA_CHECK(cudaMemcpy(d_key, h_key.data(), std::min(elem_kv, (int)h_key.size()) * FSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), std::min(elem_kv, (int)h_value.size()) * FSIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens, h_seq_lens.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const dim3 block(head_dim); 
    const dim3 grid(batch_size * num_heads); 
    
    batched_attention_fixed<<<grid, block>>>(d_query, d_key, d_value, d_seq_lens, d_output, batch_size, max_seq_len, num_heads, head_dim);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        batched_attention_fixed<<<grid, block>>>(d_query, d_key, d_value, d_seq_lens, d_output, batch_size, max_seq_len, num_heads, head_dim);
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
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_seq_lens));
    
    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt batch_size = 128;
    CInt num_heads = 32;
    CInt head_dim = 128;
    CInt max_seq_len = 1024;  // 静态 Padding 到最大长度（4096 会导致 OOM，降低以允许对比）
    CInt iterations = 100;

    std::mt19937 gen(2026);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    printDeviceInfo();

    // 建立一个倾斜的请求序列长度分布（真实网络环境）：大部分都很短，少部分极长（触发了长尾效应 Padding Waste）
    std::vector<int> seq_lens(batch_size);
    std::vector<int> seq_starts(batch_size + 1, 0);
    
    int total_actual_tokens = 0;
    for (int i = 0; i < batch_size; ++i) {
        if (i % 10 == 0) seq_lens[i] = max_seq_len; // 10% 的长序列
        else seq_lens[i] = 128 + gen() % 256;       // 90% 的短序列平均 ~256
        
        seq_starts[i + 1] = seq_starts[i] + seq_lens[i];
        total_actual_tokens += seq_lens[i];
    }

    CInt padded_total_tokens = batch_size * max_seq_len; // 静态传统批处理下的 Token 计算量
    double padded_mb = ((long long)padded_total_tokens * num_heads * head_dim * FSIZE * 2.0) / (1024.0 * 1024.0);
    double varlen_mb = ((long long)total_actual_tokens * num_heads * head_dim * FSIZE * 2.0) / (1024.0 * 1024.0);

    cout << "========================================\n";
    cout << "  动态/连续批处理 (Continuous Batching) 基准测试\n";
    cout << "========================================\n";
    cout << "Batch 规模  ：" << batch_size << " 个并发请求\n";
    cout << "单请求极长  ：" << max_seq_len << "\n";
    cout << "网络结构    ：Num Heads=" << num_heads << ", Head Dim=" << head_dim << "\n";
    cout << "Kernel 迭代 ：" << iterations << " 次\n";
    cout << "\n";
    cout << "--- 理论负载与显存开销 ---\n";
    cout << "1. 静态 / 基础批次调度 (Static Padding):\n";
    cout << "   等待集齐 " << batch_size << " 个请求并向最长维度对齐 (" << max_seq_len << ")。\n";
    cout << "   所需处理的 Token 载量: " << padded_total_tokens << " [" << fixed << setprecision(2) << padded_mb << " MB]\n";
    cout << "2. 动态 / Inflight 连续调度 (Continuous Packed Tensor):\n";
    cout << "   摒弃所有 0-Padding，合并有效 Token 放入 Flatten 数组。\n";
    cout << "   实际需计算的 Token 载量: " << total_actual_tokens << " [" << fixed << setprecision(2) << varlen_mb << " MB]\n";
    cout << ">>> 预估节省计算量 (FLOPS/Mem): " << setprecision(2) << (1.0f - ((float)total_actual_tokens / padded_total_tokens))*100.0f << "%\n";
    cout << "\n";

    // 设置张量
    CInt elem_q = batch_size * num_heads * head_dim;
    Matrix h_query(elem_q);
    for (int i = 0; i < elem_q; ++i) h_query[i] = dist(gen);
    
    // 我们仅按实际大小分配 packed 数据进行测试（给 static padding 送这部分数据以模拟实际处理速度）
    CInt elem_kv_varlen = total_actual_tokens * num_heads * head_dim;
    Matrix h_key(elem_kv_varlen);
    Matrix h_value(elem_kv_varlen);
    for (int i = 0; i < elem_kv_varlen; ++i) {
        h_key[i] = dist(gen);
        h_value[i] = dist(gen);
    }
    
    Matrix h_cpu_output(elem_q, 0.0f);
    Matrix h_gpu_fixed_output(elem_q, 0.0f);
    Matrix h_gpu_varlen_output(elem_q, 0.0f);

    // CPU 计算
    cout << "--- CPU 计时 (真实 Token) ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    varlen_attention_cpu(h_query, h_key, h_value, seq_starts, h_cpu_output, batch_size, num_heads, head_dim);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 测试：静态长度
    cout << "--- GPU 版本 1: 静态批处理 (Static Padding to Max Length) ---\n";
    GpuTimingResult res_fixed = fixed_gpu(h_query, h_key, h_value, seq_lens, h_gpu_fixed_output,
                                          batch_size, max_seq_len, num_heads, head_dim, iterations);
    cout << "H2D 传输时间：   " << setw(8) << res_fixed.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_fixed.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_fixed.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_fixed.total_ms << " ms\n";
    cout << "\n";

    // GPU 测试：变长扁平化长度
    cout << "--- GPU 版本 2: 动态批处理 / Varlen Packed Tensor ---\n";
    GpuTimingResult res_varlen = varlen_gpu(h_query, h_key, h_value, seq_starts, h_gpu_varlen_output,
                                            batch_size, total_actual_tokens, num_heads, head_dim, iterations, batched_attention_varlen);
    cout << "H2D 传输时间：   " << setw(8) << res_varlen.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_varlen.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_varlen.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_varlen.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    cout << "Kernel 耗时对比 : Static " << setprecision(4) << res_fixed.kernel_ms 
         << " ms  vs  Varlen " << res_varlen.kernel_ms << " ms\n";
    cout << ">> Static 内部通过分支跳过 Padding，因此两者 Kernel 速度接近。\n";
    cout << ">> Continuous Batching 的核心收益是显存节省（" << setprecision(2) 
         << (1.0 - (double)total_actual_tokens / padded_total_tokens) * 100.0 
         << "%），使同一 GPU 能服务更多并发请求。\n\n";

    // 两个 kernel 实际访存量相同（都只访问有效 token），用 varlen_mb 计算带宽
    double bw_fixed = (varlen_mb / 1024.0) / (res_fixed.kernel_ms / 1000.0);
    double bw_varlen = (varlen_mb / 1024.0) / (res_varlen.kernel_ms / 1000.0);
    cout << "Static 实际有效带宽：" << setw(8) << setprecision(2) << bw_fixed << " GB/s\n";
    cout << "Varlen 实际有效带宽：" << setw(8) << setprecision(2) << bw_varlen << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";
    cout << "--- 显存占用对比 (核心指标) ---\n";
    cout << "Static Padding 显存  ：" << setprecision(2) << padded_mb << " MB\n";
    cout << "Varlen Packed  显存  ：" << setprecision(2) << varlen_mb << " MB\n";
    cout << ">> 节省显存 " << (1.0 - (double)total_actual_tokens / padded_total_tokens) * 100.0 
         << "%，等效于可多服务 " << setprecision(1) << (double)padded_total_tokens / total_actual_tokens 
         << "x 的并发请求\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass = verify_results(h_gpu_varlen_output, h_cpu_output, elem_q, "Var-Len Attention");

    if (pass) {
        cout << "✓ GPU/CPU Variadic-Length Attention 结果验证通过\n";
    } else {
        cout << "✗ GPU/CPU 验证存在差异\n";
    }
    cout << "\n========================================\n";

    return 0;
}
