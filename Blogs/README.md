# Blogs — CUDA 深度技术博客

> 15 篇系统化的中文技术文章，对应项目 15 个主题模块，覆盖**数学推导、硬件原理、Kernel 实现与性能分析**。每篇文章严格按照 9 节结构规范，从底层物理约束出发到工程实现收尾，旨在基于数据回答性能边界的根因问题。

## 文章索引

| # | 文章 | 主题 | 核心内容 |
| :---: | :--- | :--- | :--- |
| 01 | [01_Basics：从 Vector Add 到 Tiled GEMM](01_Basics_Concepts_and_Tiling.md) | 基础概念与分块 | 内存带宽 vs 算力、算术强度 Roofline、Grid-Stride Loop、Shared Memory Tiling |
| 02 | [02_Reduction：并行归约的三次进化](02_Reduction_Tree_Algo_and_Coarsening.md) | 归约与线程粗化 | 树状归约、Warp Divergence 消除、收敛展开、线程粗化 |
| 03 | [03_Scan：并行前缀和的两条路线](03_Scan_Kogge_Stone_and_MultiBlock.md) | Scan 算法 | Inclusive/Exclusive Scan、Kogge-Stone (低 Span) vs Brent-Kung (低 Work) |
| 04 | [04_GEMM_Optimization：寄存器外积极限](04_GEMM_Optimization_Register_Tiling.md) | GEMM 优化全链路 | 一维粗化 → 二维寄存器分块 → 外积累加 → Double Buffering流水线 |
| 05 | [05_LLM_Ops：Transformer 核心五算子](05_LLM_Ops_FlashAttention_and_Norms.md) | 大模型算子 | Online Softmax、Welford LayerNorm、RMSNorm、RoPE、Flash Attention 原理 |
| 06 | [06_Warp_Primitives：无锁寄存器通信](06_Warp_Primitives_Register_Shuffle.md) | Warp 原语 | `__shfl_xor` / `__shfl_up` / `__shfl_down`、Block Reduce 无同步、Block Scan |
| 07 | [07_Quantization：INT8/FP16 混合精度工程](07_Quantization_FP16_INT8_dp4a.md) | 量化推理 | FP16 带宽翻倍、INT8 `dp4a` 指令吞吐、Per-Tensor vs Per-Channel 量化 |
| 08 | [08_Advanced：CUDA Graphs 与 Streams 调度](08_Advanced_CUDAGraphs_Streams_Extensions.md) | 系统级调度 | Multi-Stream 异步流水线、CUDA Graphs 减免 Kernel 启动开销、PyTorch C++ 扩展 |
| 09 | [09_Tensor_Core：WMMA API 与指令峰值](09_Tensor_Core_WMMA_Mixed_Precision.md) | 张量核心 | WMMA 16×16×16 指令拆解、Fragment 寄存器级编程、FP16 混合精度算力提速 |
| 10 | [10_Memory_Optimization：三维访存优化](10_Memory_Optimization_Coalescing_BankConflict.md) | 内存编排 | 128B Cache Line 内存合并 (Coalescing)、Bank Conflict 计算与避免 Padding |
| 11 | [11_Inference_Optimization：算子融合与 KV Cache](11_Inference_Optimization_Fusion_KVCache.md) | 推理系统优化 | Decoding 算术瓶颈分析、Kernel Fusion、PagedAttention 内存池、Continuous Batching 降维 |
| 12 | [12_Standard_Libraries：cuBLAS/cuFFT/Thrust](12_Standard_Libraries_cuBLAS_cuFFT_Thrust.md) | 标准库 | cuBLAS 列主序系统、cuFFT 规划缓存、Thrust 排序与算法的工程级取舍 |
| 13 | [13_Performance_Analysis：Roofline 理论与分析](13_Performance_Analysis_Roofline_Occupancy.md) | 性能调优 | Roofline 理论拐点计算 ($81.9\text{ FLOP/B}$)、Occupancy 占比考量、Nsight Compute (ncu) 吞吐实战 |
| 14 | [14_CUTLASS_TemplateGEMM：工业级代码生成](14_CUTLASS_TemplateGEMM_CuTe.md) | CUTLASS | 矩阵引擎的 Template 抽象阶段、CuTe 代数布局设计 (Layout)、高级元编程开发 |
| 15 | [15_Multi_GPU：Ring AllReduce 数学推导与 NCCL](15_Multi_GPU_NCCL_AllReduce.md) | 多显卡通信 | Parameter Server 星型拓扑的物理阻塞、Ring Topo 的两阶段常数通信限界、NCCL API 落地 |

## 推荐学习路线

根据阅读目标，提供四条经时间验证的高效学习流：

### 路线一：从零入门并行编程

> **适用群体**：CUDA 初学者。关注内存编排基础和线程并发思维。

```text
01 基础概念 → 02 归约 → 03 前缀和 → 06 Warp 原语 → 10 访存优化
```

### 路线二：GEMM 极致优化专精

> **适用群体**：算子工程师。深度解析矩阵计算底层，从纯缓存到寄存器再到硅片汇编级别引擎。

```text
01 Tiling 基础 → 04 寄存器分块 → 09 Tensor Core → 14 CUTLASS → 12 cuBLAS (验证)
```

### 路线三：系统性能分析与诊断调优

> **适用群体**：全栈优化师。重在识别瓶颈（Memory Bound / Compute Bound）。

```text
13 Roofline 模型诊断 → 10 访存优化 → 04 GEMM 优化剖析 → 12 标准库对标
```

### 路线四：大语言模型 (LLM) 推理底层逻辑

> **适用群体**：AI 基础设施架构师。以大模型特征作为出发点。

```text
05 大模型五算子 → 07 量化技术 → 11 推理三剑客优化 → 08 异步调度 → 15 单机多卡通信拓扑
```

## 内容架构特色

- **以数据定义标准**：杜绝"极快"或"碾压"等模糊字眼，坚用 `28.2ms`、`81.9 FLOP/Byte` 等硬件强关联度量。
- **理论实测分离**：所有量化验证均会明确标准为 `[理论]` 推算上限还是 `[实测]` 二进制跑分下限。 
- **反直觉与误区**：专门剖析例如「浮点运算单元吃满 100% 但总体业务吞吐却阻塞」、「异步 Copy 操作却降低性能」的系统性错觉。
- **完整的代码锚点**：一切公式必有源文件的特定 `__global__` 函数代码和 Nsight 系统测点对齐。
