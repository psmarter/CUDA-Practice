# 11_Inference_Optimization: 推理系统的心脏级优化

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

与训练过程（侧重高吞吐）不同，生成式大模型在推理过程中呈现出自回归（Auto-Regressive）的特性，即 Memory Bound 极其严重、每次生成的 Batch Size 很小、并且拥有越来越长的上下文依赖。本章旨在学习业界针对生成式推理所发明的三大法宝。解决上述痛点，是大规模部署高并发 AI 服务的核心壁垒。

目录下的实现覆盖了当前最主流的工业级加速范式：

- `01_kv_cache/`：演示大模型中最核心的以空间换时间的机制——保存历史 Token 生成的 K 和 V 特征，避免 $O(N^2)$ 的冗余重算，但也带来巨大的显存占用。
- `02_kernel_fusion/`：演示如何将一些元素级别的短效算子（如 Add, Mul, Activation）手动打包为单个 Kernel，缩减多次 Kernel Launch 和中转显存拉取的延迟。
- `03_dynamic_batching/`：演示在服务端如何异步积攒来自不同用户的请求，凑齐一定的 Batch Size 统一发起计算以提升算术强度（Compute/Memory Ratio）。

## 2. 原理推导与数学表达 (Math & Logic)

以 KV Cache 的复用性推演为例：
设第 $t$ 步生成时，需要预测 $t+1$ 的 Token。
如果没有 KV Cache，计算 Attention 权重需要：$Q_t \times [K_0, K_1, \ldots, K_t]$。
为此必须先重算出全部的 $K_{0 \dots t}$，浪费大量计算资源。
引入 KV Cache 后，在第 $t$ 步的计算仅为：
$$ K_{\text{new}} = W_k \cdot x_t, \quad V_{\text{new}} = W_v \cdot x_t $$
然后立刻将 $K_{\text{new}}, V_{\text{new}}$ 并入 Cache 列表的尾端，直接提取完整 Cache 参与 Attention。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

以 融合内核（Kernel Fusion）的显存节约分析为例：

```text
[无融合的传统路线]
Device Mem (X) ----读取----> [Kernel_1 (加偏置)] ----写回----> Device Mem (X1)
Device Mem (X1)----读取----> [Kernel_2 (ReLU)] ----写回----> Device Mem (Y)
(发生了 2 次全容量内存写回与 2 行启动延迟！)

[融合后的算子执行 Kernel_Fused(X, Y)]
Device Mem (X) -----> | 加载到寄存器 T_X |
                      |   T_Y = T_X + B |
                      |   T_Z = max(0, T_Y) |
           <--------- | 写回 Device Mem (Y) |
(只经过了 1 次完整的内存交互！)
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `01_kv_cache/kv_cache.cu` 中的循环缓存写入处理：

```cpp
// 典型的自回归外层逐步生成循环
for (int step = 0; step < GEN_LEN; ++step) {
    // 1. 各个头单独算出这一个 Step 新生成的 K 和 V 向量
    float current_k = ...;
    float current_v = ...;
    
    // 2. 将这几位新元素精确拼接到庞大的预分配 Cache 张量的尾部
    // 注意 Cache 张量的分配大小为 MAX_SEQ_LEN
    int cache_idx = (batch_id * MAX_SEQ_LEN + current_seq_len) * hidden_dim + tid;
    kv_cache_k[cache_idx] = current_k;
    kv_cache_v[cache_idx] = current_v;

    // 3. 将整个庞大（动态增长）的 Cache 抛给 Attention 去计算当前步概率
    compute_attention<<<...>>>(q, kv_cache_k, kv_cache_v, current_seq_len);
    
    current_seq_len++; // 序列长长，下次拼接到后一格
}
```

## 5. 性能基准与分析视角 (Performance & Profiling)

- **基准**：对比没有维护 KV Cache 的暴力每一次全序列预测法，以及拆分小 Kernel 运行的基准时间。
- **典型分析**：使用 NCU 观察，KV Cache 会导致极高的 DRAM 读取容量，但省下了海量的算力时间（`sm__cycles_elapsed` 陡降）。Kernel Fusion 则会明显看到 `kernel_launch` 开销次数缩减，且 `dram__bytes.sum` 的读写总量大幅降低（从 $O(K\times N)$ 降至 $O(N)$）。

## 6. 编译指引与参考资料 (Compile & References)

```bash
nvcc -O3 -arch=sm_89 kernel_fusion.cu -o run_fusion
# 用于查看小算子的启动开销以及显存交互体积
ncu --metrics dram__sectors_read.sum,dram__sectors_write.sum ./run_fusion
```

- 参考资料: HuggingFace: "KV Cache in Large Language Models".
