---
title: "11_Inference_Optimization：算子融合、PagedAttention 与 Continuous Batching"
date: 2026-03-12 10:00:00
tags: [CUDA, 高性能计算, LLM, 推理优化, Kernel Fusion, PagedAttention, Continuous Batching, KV Cache]
categories: 深度学习系统架构
---

## 本文目标

读完本文，你将能够：

- 理解推理 Decoding 阶段 Memory Bound 的微观成因
- 掌握 Kernel Fusion（算子融合）降低无效数据搬运的原理
- 理解 KV Cache 管理中 PagedAttention 解决显存碎片的逻辑
- 了解 Continuous Batching 如何消除 Padding 带来的无效计算与存储开销

## 对应代码路径

> **硬件环境**：NVIDIA RTX 4090 (Ada Lovelace, sm_89)
> 128 SMs | FP32 82.6 TFLOPS | HBM 1008 GB/s | L2 72 MB | Roofline 拐点 81.9 FLOP/Byte

| 源文件 | Kernel 名称 | 核心技术 | 测试规模 |
|--------|-------------|----------|----------|
| `11_Inference_Optimization/02_kernel_fusion/kernel_fusion.cu` | `add_kernel`<br>`relu_kernel`<br>`scale_kernel`<br>`fused_add_relu_scale` | 算子融合 (Kernel Fusion) | $N=134,217,728$ |
| `11_Inference_Optimization/01_kv_cache/kv_cache.cu` | `naive_attention_kernel`<br>`paged_attention_kernel` | 分页注意力 (PagedAttention) 与 KV Cache | `seq_len` $\le 2048$<br>`block_size=16` |
| `11_Inference_Optimization/03_dynamic_batching/dynamic_batching.cu` | `batched_attention_fixed`<br>`batched_attention_varlen` | 连续/动态批处理 (Continuous Batching)<br>消除 Padding | `batch=128`<br>`max_seq=1024` |

> Kernel 名称与源码中 `__global__` 函数签名完全一致。

## Baseline

**问题陈述**：在千亿参数大语言模型（LLM）的自回归生成（Decoding）阶段，由于每次仅生成 1 个 Token，运算模式从适合并行的高算术强度 GEMM 退化为了 GEMV（矩阵乘向量）。

按照 Roofline 模型计算，此时的算术强度 $I$ 极低（接近 $1.0\text{ FLOP/Byte}$），远低于 RTX 4090 的拐点 $81.9\text{ FLOP/Byte}$。系统陷入极其严重的 Memory Bound。我们用原生的序列操作（Unfused）、朴素显存分配（Naive KV Cache）和静态对齐批处理（Static Padding）作为这三个维度的 Baseline。

| Baseline 类别 | 测试场景 | 指标 | 值 | 数据来源 |
|---------------|----------|------|----|----------|
| 未融合 (Unfused) | Add+ReLU+Scale | Kernel 耗时 | 4.06 ms | [实测] Results/11_Inference_Optimization.md |
| 未融合 (Unfused) | Add+ReLU+Scale | 业务有效带宽| 396.79 GB/s | [实测] Results/11_Inference_Optimization.md |
| 朴素 KV Cache | 预分配最大长度 | 预估显存占用| 512.00 MB | [实测] Results/11_Inference_Optimization.md |
| 静态 Padding | 对齐到最大长度 | 无效计算量占比| 67.99% | [实测] Results/11_Inference_Optimization.md |

## 瓶颈分析

在推理引擎的瓶颈定位中，必须将计算效率低下与显存浪费分离分析：

1. **中间变量落盘导致带宽利用率低 (Kernel Fusion 优化点)**
   - 在 `Add -> ReLU -> Scale` 的序列调用中，即便是极其简单的逐元素操作，框架也需要多次将数据从 HBM 读入 SM，再写回 HBM 作为中间变量（如 `tmp1`, `tmp2`）。这些 Memory Round-trip 占用了宝贵的带宽，导致实际有效带宽仅跑出 396.79 GB/s [实测]。
2. **静态连续预分配导致的显存碎片 (PagedAttention 优化点)**
   - 传统的 Attention 实现要求 KV Cache 在内存中必须地址连续。由于无法预测用户实际回答的总长度，系统必须按 `max_seq_len`（如 2048）进行完整预留。对于大量短请求而言，未使用的预留块形成了夸张的内部碎片，极大限制了系统的并发上限。
3. **Padding 带来的死算与死存 (Continuous Batching 优化点)**
   - 为了以 Batch 形式下发 GEMV 任务，原始架构会将不同长度的请求使用 `<PAD>` 强制拉长到批次内的最大长度（例如 1024）。这产生了占比高达 67.99% [实测] 的占位符气泡，不仅浪费显存空间，还令总线上塞满了无意义的内存读写请求。

## 优化思路

针对上述三个维度的瓶颈，推理侧给出了对应的工程化系统解法。

### 优化 1：Kernel Fusion 算子融合

**解决的瓶颈**：无效的中间变量落盘拖垮了极度稀缺的 HBM 带宽。
**核心思想**：利用 CUDA 线程内部的高速 Registers。打破框架原本模块化的三次 Kernel 调用限制，强行将逻辑串联合并在一个 `__global__` 函数内。使数据直接在 Registers 内流转完成所有操作后，仅进行最后一次全局回写。
**预期收益**：大幅减少 `LDG` 和 `STG` 数量，大幅降低 Kernel 耗时 [理论]。

### 优化 2：PagedAttention 内存分页机制

**解决的瓶颈**：显存内部碎片导致批次并发量上不去（Capacity Bound）。
**核心思想**：打破要求 KV 张量在物理显存上连续存放的铁律。将总容量切分为定长的小 Block（例如每块 16 个 Token）。在 Kernel 寻址侧，引入 `block_table` 映射表机制；在需要读取指定位置数据时，进行即时的逻辑索引到物理指针的多重解引用转换。
**预期收益**：以微小的解引用算力开销和访存迟延，换回海量的可用显存空间 [理论]。

### 优化 3：Continuous Batching 连续动态批打平

**解决的瓶颈**：长短不一的请求合并计算时， Padding 补齐产生的死算与显存浪费。
**核心思想**：彻底摧毁 $3D$ 的 Batch 维度概念。将所有不同长度的请求内的有效 Token 全部首尾相连，压扁重构为紧凑的 $1D$ 长数组（Packed Tensor `[total_actual_tokens, ...] `）。在 CUDA Kernel 中传入 `seq_starts` 锚点数组供线程划界并自行解算坐标。
**预期收益**：杜绝一切无效的 PAD 参数计算，根据真实验证负载严格节省计算量和显存底座厚度 [理论]。

## 关键代码解释

### 内存分页转换 (PagedAttention)

```cpp
// 来源：11_Inference_Optimization/01_kv_cache/kv_cache.cu : L73-L93
    float acc = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        // [1] 在最内侧循环对 Logical 的词位置解算出逻辑 Block 编号及偏移量
        int logical_block_idx = i / block_size;
        int block_offset = i % block_size;
        
        // [2] 查询映射表，由逻辑 ID 换算拿取离散在四面八方的真实物理块 ID
        int physical_block_idx = block_table[batch_idx * max_blocks_per_seq + logical_block_idx];
        
        // [3] 直接提取 C++ 二级指针数组中的物理块基址指针
        float* k_block = k_blocks[physical_block_idx];
        float* v_block = v_blocks[physical_block_idx];
        
        // [4] 计算该物理块内部的精准元素偏移并完成取值
        int element_idx = head_idx * (block_size * head_dim) + 
                          block_offset * head_dim + tid;
        float k_val = k_block[element_idx];
        // ...
    }
```

**要点解读**：

- `[1]` 到 `[3]` 是 PagedAttention 的心脏逻辑。在原本可以通过线性递增指针的紧凑循环里，它强行引入了整数除法、模运算以及两次全局显存的非连续内存跳转（拿去查 `block_table` 和提领 `k_blocks`）。这在传统的 GPU 代码优化原则里是大忌，正是这种置诸死地而后生的交换逻辑铸就了 VLLM 架构。

### 降维展平坐标系 (Continuous Batching)

```cpp
// 来源：11_Inference_Optimization/03_dynamic_batching/dynamic_batching.cu : L68-L86
    // [1] 获取当前序列在这个 packed array 中的全局 token 起止范围
    int start_token_idx = seq_starts[batch_idx];
    int end_token_idx = seq_starts[batch_idx + 1];
    
    float q_val = query[batch_idx * num_heads * head_dim + head_idx * head_dim + tid];
    float acc = 0.0f;
    
    // [2] 仅遍历纯净的有效 token 区间！拒绝 Padding
    for (int token_idx = start_token_idx; token_idx < end_token_idx; ++token_idx) {
        // [3] 由于维度的强行压实坍塌，直接使用合并后的前端 1D ID 进行寻址坐标乘运算
        int kv_idx = token_idx * (num_heads * head_dim) + 
                     head_idx * head_dim + tid;
                     
        float k_val = key[kv_idx];
        float v_val = value[kv_idx];
        acc += (q_val * k_val) * v_val;
    }
```

**要点解读**：

- 打平后的 `[total_actual_tokens, num_heads * head_dim]` 数据结构中，GPU 线程依赖由外部 CPU 送入进来的 `seq_starts` 一维锚序列（`[1]`）作为切分尺。内环循环（`[2]`）已经变得清清爽爽，由于没有任何掺水的 0 节点，所以甚至不需要任何的分支预测 `if` 去掩码判断。

## 结果与边界

### 性能对比

> **测试条件**：RTX 4090, CUDA 12.x
> **数据来源**：`Results/11_Inference_Optimization.md` 原始日志

**1. 算子融合 (Kernel Fusion) Benchmark**

*针对特征尺寸 N=134M 元素进行 `Add -> ReLU -> Scale`*

| 实现版本 | Kernel 平均耗时 | 业务有效带宽 | 加速比 | 数据性质 |
|----------|---------------|--------------|--------|----------|
| 非融合序列 (原生调度) | 4.06 ms | 396.79 GB/s | 1.00x | [实测] |
| Fused Kernel | 1.73 ms | 932.85 GB/s | **2.35x** | [实测] |

单纯抹去中间无意义显存回写，即可在不更改任何上游逻辑前提下令全链路提速 $2.35$ 倍，并将有效内存吞吐直推物理边界极值 932.85 GB/s [实测]。

**2. PagedAttention Benchmark** (容量置换代价分析)

*Batch=32 (含长尾随机请求)*

| 实现架构 | Kernel 耗时 | 等效带宽 | 显存容量占用 | 数据性质 |
|----------|------------|----------|-------------|----------|
| Naive (盲目连续对齐) | 0.37 ms| 898.12 GB/s| 512.00 MB | [实测] |
| Paged (查表寻链法)   | 0.45 ms| 735.04 GB/s| **317.75 MB** | [实测] |

PagedArchitecture 架构在指标上产生了极其反常识的“倒退”：因繁琐的查表阻断合并访存，其实测耗时劣化了 $1.22$ 倍。但在这约 $0.08\text{ ms}$ 的舍弃换取到的却是：容量核心直接节省高达 **37.94%** [实测]！使得集群不加卡即能强行拔高承载上线。

**3. Continuous Batching Benchmark**

*包含 128 条并发指令混纺（极小部分长达 1024，大部分 200 出头）*

| 实现架构 | Kernel 执行耗时 | 核心底层算子用料与显存 | 数据性质 |
|----------|---------------|----------------------|----------|
| Static Padding 方案 | 1.52 ms | **4096.00 MB (吞没131k token)** | [实测] |
| Continuous 降维方案 | 1.69 ms | **1311.22 MB (缩减至41k token)** | [实测] |

在静态批处理中，即便凭借底层 FPU 微指令规避（由于 `if-else` 分支屏蔽）让其时间上仍然能跑到 1.52ms。但其显卡资源被极其蛮力地锁死了整整 4GB [实测]。连续批处理运用扁平化打击，挤全净水使得占据水位缩退至 1311 MB。资源节省比足达 **67.99%** [实测]。

### 边界条件与局限

- **CUDA Graph 兼容冲突**：系统最下层的终极调度掩体 CUDA Graphs 要求一切必须遵循静止不可变更的 Shape 原则。而 PagedAttention 和 Continuous Batching 本质是对 Shape 与内存指引的极度动态狂飙。工程落盘中需将架构分切开，只在固定 $b=1$ 的 Decoding 阶段用多个静态 Graph 模板去包揽这些不确定的内存链动态推演。

## 常见误区

1. **误区**：在发现程序慢的时候，只有 `sm__throughput` 这一项代表芯片的核心状态。
   **实际**：在所有大模型的生成后期 Decoding，其最核心瓶颈全部卡死在 `dram__throughput` 吞吐指标上。单纯迷信高 TFLOPS 无用武之地。
2. **误区**：既然 Paged 架构造成了内存读的碎裂甚至降频，我们可以通过做 Tiling 给它接回来以求两全其美。
   **实际**：这在 KV Cache 的长条向向量中基本等同无效功。其本质不在于计算架构编排松散，在于根本无法预测这几十名用户的上下文会被随机散落在哪。其属于无可避免的数据架构级开销税。

## 系列导航

### 前置阅读

| 文章 | 关系 |
|------|------|
| [13_Performance_Analysis_Roofline_Occupancy.md](13_Performance_Analysis_Roofline_Occupancy.md) | 在诊断篇已详细讲解的 $I$ 和带宽天花板底层 |

### 推荐后续

| 文章 | 关系 |
|------|------|
| [14_CUTLASS_TemplateGEMM_CuTe.md](14_CUTLASS_TemplateGEMM_CuTe.md) | 即便已经化简成了无 Padding 版本，最后纯血内核矩阵的压榨仍需要依赖工程端点 |
