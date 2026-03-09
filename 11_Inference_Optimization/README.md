# 11_Inference_Optimization: 大语言模型推理核心优化范式

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

自回归生成（Auto-Regressive Generation）是大模型推理的灵魂，但其往往伴随着严重的 Memory Bound （内存墙限制）。每一次生成循环都只能吞吐一个 Token，这导致计算单元大部分时间在等待显存数据的搬运。为了打破僵局并在工业界实现高并发吞吐，业界摸索出了一系列针对推理引擎的优化策略组合拳。

本章将带你深入当前 AI 系统（如 vLLM、TensorRT-LLM）所采用的核心底层技术：
- `01_kv_cache/`：演示大模型最基本的空间换时间策略——保留历史键值对，并利用类 OS 虚拟内存的 **PagedAttention** 技术消除预分配带来的极其严重的内部碎片。
- `02_kernel_fusion/`：演示**算子融合 (Kernel Fusion)** 技术在显存受限场景下的威力，将冗余显存换乘转换为寄存器级的低级通信。
- `03_dynamic_batching/`：演示连续批处理 (Continuous Batching) 与 **Varlen (Variadic Length) Attention**，通过打包非规则长度张量，直接抹除传统 Padding 带来的无效计算与存储浪费。

## 2. 原理推导与数学表达 (Math & Logic)

**1. Paged KV Cache 的内存切分：**
传统的静态分配采用 $[B, H, L_{\text{max}}, D]$ 布局，如果真实生成的序列只走了 $10$，剩下的 $L_{\text{max}} - 10$ 全是内存浪费。
分页机制（Paged Cache）将长度 $L$ 切分为固定大小的 Block (例如 $B=16$)：
$$ \text{Total Blocks} = \lceil L / 16 \rceil $$
Attention 时，通过一张类似页表 (Block Table) 的映射结构：
$$ \text{Physical\_Addr}(t) = \text{BlockTable}[B_i][\text{Head}][t \bmod 16][\text{Dim}] $$
此结构可完全打散碎存池，使得显存按需分配，零碎片。

**2. 变长注意力模型 (Varlen Attention)：**
设系统接受 3 个并发请求，长度为 $[10, 50, 20]$。传统 Static Padding 会用 $0$ 填满到全局最大长度 $50$，需分配 $3 \times 50 = 150$ 空间。
动态批处理使用连续 1D 张量压缩 (Packed 1D Tensor)：
$$ \text{Position Array}: [0, \dots, 9 \mid 0, \dots, 49 \mid 0, \dots, 19] $$
共只需 $10 + 50 + 20 = 80$ 单位的显存大小，直接在单 Kernel 内分发计算，将 $0$ Padding 开销彻底抹除。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

以 融合内核（Kernel Fusion）的显存节约分析为例：

```text
[无融合的传统路线：Add -> ReLU -> Scale]
Global Mem (A) ----读取----> [Kernel_1 (加法)] ----写回----> Global Mem (A+B)
Global Mem (A+B)---读取----> [Kernel_2 (ReLU)] ----写回----> Global Mem (ReLU)
Global Mem (ReLU)--读取----> [Kernel_3 (缩放)] ----写回----> Global Mem (Out)
(发生了 3 次来回的 DRAM 数据倾倒，带宽消耗极大！)

[融合后的算子执行 Fused_Kernel]
Global Mem (A, B) -----> | 加载到寄存器 Register T_A, T_B |
                         |   T_Y = T_A + T_B (Add)        |
                         |   T_Z = max(0, T_Y) (ReLU)     |
                         |   T_R = T_Z * Scale (Scale)    |
              <--------- | 仅仅一次写回 Global Mem (Out)  |
(中间状态全在极速寄存器中流转，达到带宽理论物理极限！)
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `01_kv_cache/kv_cache.cu` 中的 Paged 寻址逻辑：

```cpp
// 典型的 PagedAttention 虚拟寻址过程
__global__ void paged_kv_cache_kernel(
    const float* __restrict__ block_pool, // 全局杂乱的内存块池子
    const int* __restrict__ block_table,  // 对应每个序列分配的物理块号
    float* __restrict__ output,
    int seq_len, int block_size, int hidden_dim) {
    
    // ...
    // 给定当前需要取的 token 的逻辑索引 step
    int step = /* 循环变量 */;
    
    // 1. 算出它落在哪个逻辑块
    int logical_block_idx = step / block_size;
    int offset_in_block = step % block_size;
    
    // 2. 查表(Translation Lookaside Buffer的软件化)得到真实物理块号
    int physical_block_idx = block_table[batch_idx * max_blocks_per_seq + logical_block_idx];
    
    // 3. 计算极其离散但完全无碎片的真实显存地址进行载入
    int physical_offset = (physical_block_idx * block_size + offset_in_block) * hidden_dim + dim_idx;
    float k_val = block_pool[physical_offset]; 
}
```

## 5. 性能基准与分析视角 (Performance & Profiling)

在 RTX 4090 上深度压测三大策略的实际量化收益：

**1. PagedAttention 显存收益 (`01_kv_cache`)：**
* 常规(按2048最长静态全量分配): 512.00 MB
* Paged(依16步长按需碎片分配): 317.75 MB
* **结论**: 牺牲了约 23% 的执行时间换取了多达 **37.94% 的显存节约**。在大模型中，极度稀缺的是显存（决定能开多大 Batch），时间换空间的 Paged KV 成为唯一选择。

**2. 算子融合加速 (`02_kernel_fusion`)：**
* 非融合链路(Add->ReLU->Scale): 4.06 ms (有效带宽 396.84 GB/s)
* 融合算子: **1.73 ms (**有效带宽 **932.93 GB/s**)
* **加速比**: **2.35 倍**，不仅消除 Kernel Launch，且由于避免了将中间状态写入重读，彻底突破访存天花板。

**3. 连续变长批处理 (`03_dynamic_batching`)：**
* 测试 128 并发不同长短句子。
* 传统的静态 0-Padding 需对齐至最长，分配显存高达 **4096 MB**，而处理长串 0 浪费了巨量算力。
* 使用 1D Flatten 的 Var-len Attention，显存暴降至 **1311.22 MB (节省 67.99%)**。省下来的资源使得我们能将并发处理量一举提升 **3.1倍**。

## 6. 编译指引与参考资料 (Compile & References)

```bash
mkdir build && cd build
cmake .. && make kv_cache kernel_fusion dynamic_batching

./11_Inference_Optimization/01_kv_cache/kv_cache
./11_Inference_Optimization/02_kernel_fusion/kernel_fusion
./11_Inference_Optimization/03_dynamic_batching/dynamic_batching
```

**参考资料:**
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention (SOSP 2023)](https://github.com/vllm-project/vllm)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [NVIDIA TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)