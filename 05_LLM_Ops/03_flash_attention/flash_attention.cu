// Flash Attention - 高效注意力机制核心算子
#include <code_abbreviation.h>

// -----------------------------------------------------------------------------------------
// 朴素 Attention (QKV)
// -----------------------------------------------------------------------------------------

// 第 1 步：Q * K^T 矩阵乘（GPU kernel，手写）
__global__ void naive_bmm_qk(CPFloat Q, CPFloat K, PFloat S, CInt batch, CInt heads, CInt seq_len, CInt head_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算当前线程处理的 batch 和 head 索引
    int h = blockIdx.z % heads; 
    int b = blockIdx.z / heads; 

    // 计算 Q * K^T 的元素 S[row, col]
    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        int qkv_offset = (b * heads *seq_len * head_dim) + (h * seq_len * head_dim);
        for (int i = 0; i < head_dim; ++i) {
            sum += Q[qkv_offset + row * head_dim + i] * K[qkv_offset + col * head_dim + i];
        }

        // 缩放因子
        float scale = rsqrtf(static_cast<float>(head_dim));
        S[(b * heads * seq_len * seq_len) + (h * seq_len * seq_len) + (row * seq_len) + col] = sum * scale;
    }
}

// 第 2 步：对 S 矩阵求 Softmax（GPU kernel，手写）
__global__ void naive_softmax_kernel(PFloat S, CInt batch, CInt heads, CInt seq_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    int h = blockIdx.y;
    int b = blockIdx.z;

    if (row < seq_len) {
        int s_offset = (b * heads * seq_len * seq_len) + (h * seq_len * seq_len);
        float* s_row = &S[s_offset + row * seq_len];

        // 计算行最大值
        float max_val = -INFINITY;
        for (int i = 0; i < seq_len; ++i) {
            max_val = fmaxf(max_val, s_row[i]);
        }

        // 计算行的指数和
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            s_row[i] = __expf(s_row[i] - max_val); // 减去最大值以提高数值稳定性
            sum_exp += s_row[i];
        }

        // 归一化
        for (int i = 0; i < seq_len; ++i) {
            s_row[i] /= sum_exp;
        }
    }
}

// 第 3 步：P * V 矩阵乘（GPU kernel，手写）
__global__ void naive_bmm_pv(CPFloat P, CPFloat V, PFloat O, CInt batch, CInt heads, CInt seq_len, CInt head_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算当前线程处理的 batch 和 head 索引
    int h = blockIdx.z % heads; 
    int b = blockIdx.z / heads; 

    // 计算 P * V 的元素 O[row, col]
    if (row < seq_len && col < head_dim) {
        float sum = 0.0f;
        int p_offset = (b * heads * seq_len * seq_len) + (h * seq_len * seq_len);
        int v_offset = (b * heads * seq_len * head_dim) + (h * seq_len * head_dim);
        
        for (int i = 0; i < seq_len; ++i) {
            sum += P[p_offset + row * seq_len + i] * V[v_offset + i * head_dim + col];
        }
        O[v_offset + row * head_dim + col] = sum;
    }
}

// Flash Attention V1 (分块 + 重计算) （GPU kernel，手写）
__global__ void flash_attention(CPFloat Q, CPFloat K, CPFloat V, PFloat O,
                                 CInt batch, CInt heads, CInt seq_len, CInt head_dim) {
    // 动态共享内存布局
    extern __shared__ float sMEM[]; 
    float* s_Q = sMEM;                      
    float* s_K = s_Q + BR * head_dim;       
    float* s_V = s_K + BC * head_dim;       
    
    // 要求 Launch 配置满足 blockDim.x == BR == BC
    int tx = threadIdx.x;     
    int bx = blockIdx.x;      
    int head = blockIdx.y;    
    int b = blockIdx.z;       
    
    int qkv_offset = (b * heads * seq_len * head_dim) + (head * seq_len * head_dim);
    int row_q = bx * BR + tx;
    
    // 把当前 Q 块搬运到 Shared Memory：每个线程搬运自己负责的那一行
    if (row_q < seq_len) {
        for (int i = 0; i < head_dim; ++i) {
            s_Q[tx * head_dim + i] = Q[qkv_offset + row_q * head_dim + i];
        }
    }
    
    // Softmax 维护变量
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    float scale = rsqrtf((float)head_dim);
    int num_blocks_K = (seq_len + BC - 1) / BC;
    
    // 先将 O的当前行在 Global Memory 里初始化为 0
    if (row_q < seq_len) {
        for (int d = 0; d < head_dim; ++d) {
            O[qkv_offset + row_q * head_dim + d] = 0.0f;
        }
    }
    
    for (int j = 0; j < num_blocks_K; ++j) {
        int row_k = j * BC + tx;
        
        // 因此每个线程搬一行
        if (row_k < seq_len) {
            for (int i = 0; i < head_dim; ++i) {
                s_K[tx * head_dim + i] = K[qkv_offset + row_k * head_dim + i];
                s_V[tx * head_dim + i] = V[qkv_offset + row_k * head_dim + i];
            }
        }
        __syncthreads();
        
        if (row_q < seq_len) {
            float m_i_new = m_i;
            
            // 在寄存器存一个小数组，只存 BC (例如 32) 个元素。
            float s_ij[BC]; // BC=32，耗费 32 个寄存器，可接受。
            
            for (int k_idx = 0; k_idx < BC; ++k_idx) {
                if (j * BC + k_idx < seq_len) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        sum += s_Q[tx * head_dim + d] * s_K[k_idx * head_dim + d];
                    }
                    s_ij[k_idx] = sum * scale;
                    m_i_new = fmaxf(m_i_new, s_ij[k_idx]);
                }
            }
            
            // 计算 exp
            float p_sum = 0.0f;
            for (int k_idx = 0; k_idx < BC; ++k_idx) {
                if (j * BC + k_idx < seq_len) {
                    float p = __expf(s_ij[k_idx] - m_i_new);
                    s_ij[k_idx] = p; // s_ij 现在就是 P
                    p_sum += p;
                }
            }
            
            float exp_diff = __expf(m_i - m_i_new);
            float l_i_new = exp_diff * l_i + p_sum;
            
            // O_new = O_old * exp_diff + P * V
            for (int d = 0; d < head_dim; ++d) {
                float pv_sum = 0.0f;
                for (int k_idx = 0; k_idx < BC; ++k_idx) {
                    if (j * BC + k_idx < seq_len) {
                        pv_sum += s_ij[k_idx] * s_V[k_idx * head_dim + d];
                    }
                }

                float old_o = O[qkv_offset + row_q * head_dim + d];
                O[qkv_offset + row_q * head_dim + d] = old_o * exp_diff + pv_sum;
            }
            
            m_i = m_i_new;
            l_i = l_i_new;
        }
        __syncthreads(); 
    }

    if (row_q < seq_len) {
        for (int d = 0; d < head_dim; ++d) {
            O[qkv_offset + row_q * head_dim + d] /= l_i;
        }
    }
}

// Flash Attention V3 (打磨版) （GPU kernel，手写）
__global__ void flash_attention_v3(CPFloat Q, CPFloat K, CPFloat V, PFloat O,
                                    CInt batch, CInt heads, CInt seq_len, CInt head_dim) {
    // 让 128 个线程 (4个 Wapr) 的 Macro-Block 吃同样大小的 K/V (BC*D), 但吞噬 4 倍大的 Q (BR_v3*D)
    // 这样 HBM 中 K, V 矩阵的重复访问量（Memory Traffic）被压低 4 倍
    const int BR_V3 = WARPS_PER_BLOCK * BR; // 128 行 Q
    
    extern __shared__ float sMEM[]; 
    float* s_Q = sMEM;                              
    float* s_K = s_Q + BR_V3 * head_dim;       
    float* s_V = s_K + BC * head_dim;       
    
    int tx = threadIdx.x;      // 0 ~ 127
    int bx = blockIdx.x;       // Q 的大块号
    int head = blockIdx.y;    
    int b = blockIdx.z;       
    
    int qkv_offset = (b * heads * seq_len * head_dim) + (head * seq_len * head_dim);
    int row_q = bx * BR_V3 + tx; 
    
    const int D_FLOAT4 = head_dim / 4; 
    if (row_q < seq_len) {
        const float4* Q_f4 = reinterpret_cast<const float4*>(&Q[qkv_offset + row_q * head_dim]);
        float4* s_Q_f4 = reinterpret_cast<float4*>(&s_Q[tx * head_dim]);
        #pragma unroll
        for (int i = 0; i < D_FLOAT4; ++i) {
            s_Q_f4[i] = Q_f4[i];
        }
    }
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    float scale = rsqrtf((float)head_dim);
    int num_blocks_K = (seq_len + BC - 1) / BC;
    
    for (int j = 0; j < num_blocks_K; ++j) {
        int row_k_start = j * BC;
        
        int lines_per_thread = (BC * D_FLOAT4) / blockDim.x; // 32*32/128 = 8 个 float4
        
        #pragma unroll
        for (int step = 0; step < lines_per_thread; ++step) {
            int task_id = tx + step * blockDim.x;  // 该线程分配到的第 task_id 个 float4 任务
            int k_row_local = task_id / D_FLOAT4;  // 位于 Shared Memory 中的行号 (0~31)
            int k_col_f4 = task_id % D_FLOAT4;     // 位于该行中的第几个 float4 (0~31)
            
            int global_k_row = row_k_start + k_row_local; // 这是他在 K 里的实际行号
            
            if (global_k_row < seq_len) {
                 const float4* K_f4 = reinterpret_cast<const float4*>(&K[qkv_offset + global_k_row * head_dim]);
                 const float4* V_f4 = reinterpret_cast<const float4*>(&V[qkv_offset + global_k_row * head_dim]);
                 
                 float4* s_K_f4 = reinterpret_cast<float4*>(&s_K[k_row_local * head_dim]);
                 float4* s_V_f4 = reinterpret_cast<float4*>(&s_V[k_row_local * head_dim]);
                 
                 s_K_f4[k_col_f4] = K_f4[k_col_f4];
                 s_V_f4[k_col_f4] = V_f4[k_col_f4];
            }
        }
        __syncthreads(); // 必须等，4 个 Warp 搬完了 K/V，现在 128 个人开吃
        
        // 下面的逻辑与 V2 基本一致，只是这 128 个线程各自算自己那一行的 Q
        if (row_q < seq_len) {
            float m_i_new = m_i;
            float s_ij[BC]; 
            
            #pragma unroll 4
            for (int k_idx = 0; k_idx < BC; ++k_idx) {
                if (row_k_start + k_idx < seq_len) {
                    float sum = 0.0f;
                   
                    const float4* q_vec = reinterpret_cast<const float4*>(&s_Q[tx * head_dim]);
                    const float4* k_vec = reinterpret_cast<const float4*>(&s_K[k_idx * head_dim]);
                    
                    #pragma unroll
                    for (int d4 = 0; d4 < D_FLOAT4; ++d4) {
                        float4 qv = q_vec[d4];
                        float4 kv = k_vec[d4];
                        sum += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                    }
                    s_ij[k_idx] = sum * scale;
                    m_i_new = fmaxf(m_i_new, s_ij[k_idx]);
                }
            }
            
            float p_sum = 0.0f;
            #pragma unroll
            for (int k_idx = 0; k_idx < BC; ++k_idx) {
                if (row_k_start + k_idx < seq_len) {
                    float p = __expf(s_ij[k_idx] - m_i_new);
                    s_ij[k_idx] = p; 
                    p_sum += p;
                }
            }
            
            float exp_diff = __expf(m_i - m_i_new);
            float l_i_new = exp_diff * l_i + p_sum;
            
            #pragma unroll
            for (int d_start = 0; d_start < head_dim; d_start += D_TILE) {
                for (int d = d_start; d < d_start + D_TILE && d < head_dim; ++d) {
                    float pv_sum = 0.0f;
                    #pragma unroll 4
                    for (int k_idx = 0; k_idx < BC; ++k_idx) {
                        if (row_k_start + k_idx < seq_len) {
                            pv_sum += s_ij[k_idx] * s_V[k_idx * head_dim + d];
                        }
                    }

                    float old_o = O[qkv_offset + row_q * head_dim + d];
                    O[qkv_offset + row_q * head_dim + d] = old_o * exp_diff + pv_sum;
                }
            }
            m_i = m_i_new;
            l_i = l_i_new;
        }
        __syncthreads(); // 等这一刻度的 O 算完，再由全体 Warp 去换下一批 K/V
    }
    
    if (row_q < seq_len) {
        float4* O_f4 = reinterpret_cast<float4*>(&O[qkv_offset + row_q * head_dim]);
        #pragma unroll
        for (int d4 = 0; d4 < D_FLOAT4; ++d4) {
            float4 out_val = O_f4[d4];
            out_val.x /= l_i;
            out_val.y /= l_i;
            out_val.z /= l_i;
            out_val.w /= l_i;
            O_f4[d4] = out_val;
        }
    }
}

// Attention 参考实现（CPU，手写）
void attention_cpu(CPFloat Q, CPFloat K, CPFloat V, PFloat O,
                   CInt batch, CInt heads, CInt seq_len, CInt head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int qkv_offset = (b * heads * seq_len * head_dim) + (h * seq_len * head_dim);
            
            for (int i = 0; i < seq_len; ++i) { // 遍历 Q 的每一行
                
                std::vector<float> S_i(seq_len);
                float max_val = -INFINITY;
                
                // 1. 求 S = Q * K^T
                for (int j = 0; j < seq_len; ++j) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        sum += Q[qkv_offset + i * head_dim + d] * K[qkv_offset + j * head_dim + d];
                    }
                    S_i[j] = sum * scale;
                    max_val = max(max_val, S_i[j]);
                }
                
                // 2. Softmax = exp(S - max) / sum(exp)
                float l_i = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    S_i[j] = expf(S_i[j] - max_val);
                    l_i += S_i[j];
                }
                for (int j = 0; j < seq_len; ++j) {
                    S_i[j] /= l_i; // 此时 S_i 就是权重 P
                }
                
                // 3. O = P * V
                for (int d = 0; d < head_dim; ++d) {
                    float out_val = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        out_val += S_i[j] * V[qkv_offset + j * head_dim + d];
                    }
                    O[qkv_offset + i * head_dim + d] = out_val;
                }
            }
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

    cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result[0] << " (期望 " << cpu_result[0] << ")\n";
    return true;
}

// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      
    float kernel_ms;   
    float d2h_ms;      
    float total_ms;    
};

// 朴素 Attention 封装（GPU，手写）
GpuTimingResult naive_attention_gpu(CRMatrix h_Q, CRMatrix h_K, CRMatrix h_V, RMatrix h_O, 
                                     CInt batch, CInt heads, CInt seq_len, CInt head_dim, CInt iterations) {
    PFloat d_Q = nullptr, d_K = nullptr, d_V = nullptr, d_O = nullptr;
    PFloat d_S = nullptr; 
    CSize size_qkv = batch * heads * seq_len * head_dim * FSIZE;
    CSize size_s = batch * heads * seq_len * seq_len * FSIZE; 
    
    CUDA_CHECK(cudaMalloc((void**)&d_Q, size_qkv));
    CUDA_CHECK(cudaMalloc((void**)&d_K, size_qkv));
    CUDA_CHECK(cudaMalloc((void**)&d_V, size_qkv));
    CUDA_CHECK(cudaMalloc((void**)&d_O, size_qkv));
    CUDA_CHECK(cudaMalloc((void**)&d_S, size_s));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), size_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), size_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), size_qkv, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    // QK 矩阵乘配置 (2D Tile)
    const dim3 block_qk(16, 16);
    const dim3 grid_qk(cdiv(seq_len, 16), cdiv(seq_len, 16), batch * heads);
    
    // Softmax 配置 (1D Rowwise)
    const dim3 block_soft(128); // 优化了线程数量，避免越界
    const dim3 grid_soft(cdiv(seq_len, 128), heads, batch);
    
    // PV 矩阵乘配置 (2D Tile)
    const dim3 block_pv(16, 16);
    const dim3 grid_pv(cdiv(head_dim, 16), cdiv(seq_len, 16), batch * heads);
    
    // 预热 
    naive_bmm_qk<<<grid_qk, block_qk>>>(d_Q, d_K, d_S, batch, heads, seq_len, head_dim);
    naive_softmax_kernel<<<grid_soft, block_soft>>>(d_S, batch, heads, seq_len);
    naive_bmm_pv<<<grid_pv, block_pv>>>(d_S, d_V, d_O, batch, heads, seq_len, head_dim);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        naive_bmm_qk<<<grid_qk, block_qk>>>(d_Q, d_K, d_S, batch, heads, seq_len, head_dim);
        naive_softmax_kernel<<<grid_soft, block_soft>>>(d_S, batch, heads, seq_len);
        naive_bmm_pv<<<grid_pv, block_pv>>>(d_S, d_V, d_O, batch, heads, seq_len, head_dim);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, size_qkv, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); 
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O)); 
    CUDA_CHECK(cudaFree(d_S)); 
    return result;
}

// Flash Attention V1 封装（GPU，手写）
GpuTimingResult flash_attention_gpu(CRMatrix h_Q, CRMatrix h_K, CRMatrix h_V, RMatrix h_O, 
                                     CInt batch, CInt heads, CInt seq_len, CInt head_dim, CInt iterations) {
    PFloat d_Q = nullptr, d_K = nullptr, d_V = nullptr, d_O = nullptr;
    CSize size_qkv = batch * heads * seq_len * head_dim * FSIZE; 
    
    CUDA_CHECK(cudaMalloc((void**)&d_Q, size_qkv));
    CUDA_CHECK(cudaMalloc((void**)&d_K, size_qkv));
    CUDA_CHECK(cudaMalloc((void**)&d_V, size_qkv));
    CUDA_CHECK(cudaMalloc((void**)&d_O, size_qkv));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), size_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), size_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), size_qkv, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    // 设置 Kernel Launch 配置
    const dim3 block(BR);
    const dim3 grid(cdiv(seq_len, BR), heads, batch);
    
    // 给 Q, K, V 各分配一个 Block 大小的 Shared Memory：BR*dim + BC*dim + BC*dim
    CSize smem_size = (BR * head_dim + 2 * BC * head_dim) * FSIZE;
    
    flash_attention<<<grid, block, smem_size>>>(d_Q, d_K, d_V, d_O, batch, heads, seq_len, head_dim);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        flash_attention<<<grid, block, smem_size>>>(d_Q, d_K, d_V, d_O, batch, heads, seq_len, head_dim);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, size_qkv, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); 
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O)); 
    return result;
}

// Flash Attention V3 封装（GPU，手写）
GpuTimingResult flash_attention_v3_gpu(CRMatrix h_Q, CRMatrix h_K, CRMatrix h_V, RMatrix h_O, 
                                     CInt batch, CInt heads, CInt seq_len, CInt head_dim, CInt iterations) {
    if (head_dim % 4 != 0) {
        return GpuTimingResult{};
    }

    PFloat d_Q = nullptr, d_K = nullptr, d_V = nullptr, d_O = nullptr;
    CSize size_qkv = batch * heads * seq_len * head_dim * FSIZE; 
    
    CUDA_CHECK(cudaMalloc((void**)&d_Q, size_qkv));
    CUDA_CHECK(cudaMalloc((void**)&d_K, size_qkv));
    CUDA_CHECK(cudaMalloc((void**)&d_V, size_qkv));
    CUDA_CHECK(cudaMalloc((void**)&d_O, size_qkv));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), size_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), size_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), size_qkv, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    const int BR_V3 = WARPS_PER_BLOCK * BR; // 128
    const dim3 block(BR_V3); 
    const dim3 grid(cdiv(seq_len, BR_V3), heads, batch); // Grid 小了四倍
    
    // SMEM: Q (128行)*dim + K (32行)*dim + V (32行)*dim 
    CSize smem_size = (BR_V3 * head_dim + 2 * BC * head_dim) * FSIZE;
    
    flash_attention_v3<<<grid, block, smem_size>>>(d_Q, d_K, d_V, d_O, batch, heads, seq_len, head_dim);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        flash_attention_v3<<<grid, block, smem_size>>>(d_Q, d_K, d_V, d_O, batch, heads, seq_len, head_dim);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, size_qkv, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); 
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O)); 
    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 890
    cudaFuncSetAttribute(flash_attention_v3, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
#endif
#endif

    CInt batch = 2;
    CInt heads = 4;
    CInt seq_len = 2048; 
    CInt head_dim = 64; 
    CInt n = batch * heads * seq_len * head_dim; // QKV 的基础形状
    CInt iterations = 50; // 重矩阵乘计算密集量大

    CSize size_qkv = n * FSIZE;
    CSize size_s = batch * heads * seq_len * seq_len * FSIZE; 
    const double size_qkv_mb = size_qkv / (1024.0 * 1024.0);
    const double size_s_mb = size_s / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      Flash Attention 性能基准测试\n";
    cout << "========================================\n";
    cout << "Batch 大小：" << batch << "\n";
    cout << "Heads 数量：" << heads << "\n";
    cout << "序列长度 (Seq_Len)：" << seq_len << "\n";
    cout << "头维度维度 (Head_Dim)：" << head_dim << "\n";
    cout << "输入矩阵体积 (Q)：" << fixed << setprecision(2) << size_qkv_mb << " MB\n";
    cout << "[警告] 朴素版 N*N 中间变量体积：" << size_s_mb << " MB\n";
    cout << "Flash Attention 块大小：" << "BR=" << BR << " BC=" << BC << "\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    Matrix h_Q(n);
    Matrix h_K(n);
    Matrix h_V(n);
    Matrix h_O_cpu(n, 0.0f);
    Matrix h_O_naive(n, 0.0f);
    Matrix h_O_flash(n, 0.0f);
    Matrix h_O_flash_v3(n, 0.0f);

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_Q[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_K[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_V[i] = static_cast<float>(rand() % 100) / 100.0f;
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    attention_cpu(h_Q.data(), h_K.data(), h_V.data(), h_O_cpu.data(), batch, heads, seq_len, head_dim);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1: 朴素注意力
    cout << "--- GPU 版本 1: Naive Attention (模拟全显存遍历) ---\n";
    GpuTimingResult res_naive = naive_attention_gpu(h_Q, h_K, h_V, h_O_naive, batch, heads, seq_len, head_dim, iterations);
    cout << "H2D 传输时间：   " << setw(8) << res_naive.h2d_ms << " ms\n";
    cout << "Kernel 累加时间：" << setw(8) << res_naive.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_naive.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_naive.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2: Flash Attention V1
    cout << "--- GPU 版本 2: Flash Attention V1 (SRAM Tiling + 重计算) ---\n";
    GpuTimingResult res_flash = flash_attention_gpu(h_Q, h_K, h_V, h_O_flash, batch, heads, seq_len, head_dim, iterations);
    cout << "H2D 传输时间：   " << setw(8) << res_flash.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_flash.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_flash.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_flash.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 3: Flash Attention V3
    cout << "--- GPU 版本 3: Flash Attention V3 (Macro-Block + Vectorization) ---\n";
    GpuTimingResult res_flash_v3 = flash_attention_v3_gpu(h_Q, h_K, h_V, h_O_flash_v3, batch, heads, seq_len, head_dim, iterations);
    if(res_flash_v3.kernel_ms == 0) {
        cout << "[错误] Head Dim " << head_dim << " 不是 4 的倍数，不支持 V3 运行！\n";
        return 0;
    }
    cout << "H2D 传输时间：   " << setw(8) << res_flash_v3.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_flash_v3.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_flash_v3.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_flash_v3.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_naive = cpu_time_ms / res_naive.kernel_ms;
    double speedup_flash_v3 = cpu_time_ms / res_flash_v3.kernel_ms;
    cout << "CPU vs Flash V3 加速比：" << setprecision(2) << speedup_flash_v3 << "x\n";

    // 注意：算力墙而非带宽墙
    double bytes_load = 4.0 * batch * heads * seq_len * head_dim * FSIZE; // 理论最少载荷：读QKV，写O
    double bw_naive = (bytes_load / 1e9) / (res_naive.kernel_ms / 1000.0);
    double bw_flash = (bytes_load / 1e9) / (res_flash.kernel_ms / 1000.0);
    double bw_flash_v3 = (bytes_load / 1e9) / (res_flash_v3.kernel_ms / 1000.0);
    
    cout << "Naive GPU 有效推断带宽：" << setprecision(2) << bw_naive << " GB/s\n";
    cout << "Flash V1  有效推断带宽：" << setprecision(2) << bw_flash << " GB/s\n";
    cout << "Flash V3  有效推断带宽：" << setprecision(2) << bw_flash_v3 << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "Naive Attention (3 Steps): " << setw(8) << setprecision(4) << res_naive.kernel_ms << " ms (基准)\n";
    cout << "Flash Attention V1:        " << setw(8) << res_flash.kernel_ms << " ms (" 
         << setprecision(2) << res_naive.kernel_ms / res_flash.kernel_ms << "x)\n";
    cout << "Flash Attention V3:        " << setw(8) << res_flash_v3.kernel_ms << " ms (" 
         << setprecision(2) << res_naive.kernel_ms / res_flash_v3.kernel_ms << "x)\n";
    cout << "\n";

    // 结果验证: 涉及到双重矩阵乘与连续极长序列的精度衰减
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_O_naive, h_O_cpu, "Naive Attention", 1e-2f);
    bool pass2 = verify_results(h_O_flash, h_O_cpu, "Flash Attention V1", 1e-2f);
    bool pass3 = verify_results(h_O_flash_v3, h_O_cpu, "Flash Attention V3 (Macro-Block)", 2e-2f);

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2 && pass3) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}