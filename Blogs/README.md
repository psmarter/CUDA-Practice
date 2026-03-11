# Blogs — CUDA 深度技术博客

> 15 篇系统化的中文技术文章，对应项目 15 个主题模块，覆盖**数学推导、硬件原理、Kernel 实现与性能分析**。每篇文章从底层物理约束出发、到工程实现收尾，旨在回答"为什么这么快/慢"的根因问题。

## 文章索引

| # | 文章 | 主题 | 核心内容 |
| :---: | :--- | :--- | :--- |
| 01 | [从 Vector Add 到 Tiled GEMM](01_Basics_Concepts_and_Tiling.md) | 基础概念与分块 | 带宽 vs 算力、算术强度、Grid-Stride Loop、Shared Memory Tiling |
| 02 | [并行归约的三次进化](02_Reduction_Tree_Algo_and_Coarsening.md) | 归约与线程粗化 | 树状归约、Warp Divergence 消除、收敛展开、线程粗化 |
| 03 | [并行前缀和的两条路线](03_Scan_Kogge_Stone_and_MultiBlock.md) | Scan 算法 | Inclusive/Exclusive Scan、Kogge-Stone (低 Span) vs Brent-Kung (低 Work) |
| 04 | [寄存器外积的极限](04_GEMM_Optimization_Register_Tiling.md) | GEMM 优化全链路 | 一维粗化 → 二维寄存器分块 → 外积累加 → cuBLAS 对标 |
| 05 | [Transformer 核心五算子](05_LLM_Ops_FlashAttention_and_Norms.md) | 大模型算子 | Online Softmax、Welford LayerNorm、RMSNorm、RoPE、Flash Attention V1/V3 |
| 06 | [无锁寄存器通信](06_Warp_Primitives_Register_Shuffle.md) | Warp 原语 | `__shfl_xor` / `__shfl_up` / `__shfl_down`、Block Reduce 无同步、Block Scan |
| 07 | [混合精度工程](07_Quantization_FP16_INT8_dp4a.md) | 量化推理 | FP16 带宽翻倍、INT8 dp4a 指令、Per-Tensor vs Per-Channel 量化 |
| 08 | [系统调度优化](08_Advanced_CUDAGraphs_Streams_Extensions.md) | 高级特性 | Multi-Stream 流水线、CUDA Graphs 减驱动开销、PyTorch C++ Extension |
| 09 | [硅片矩阵乘法](09_Tensor_Core_WMMA_Mixed_Precision.md) | 张量核心 | HMMA 16×16×16 指令、Fragment 集体编程、FP16→FP32 混合精度 |
| 10 | [三维访存优化](10_Memory_Optimization_Coalescing_BankConflict.md) | 内存层次 | 128B Cache Line 合并、Bank Conflict Padding、`cp.async` 异步流水线 |
| 11 | [LLM 推理三驾马车](11_Inference_Optimization_Fusion_KVCache.md) | 推理优化 | Roofline 下的 Decoding 瓶颈、Kernel Fusion、PagedAttention、Continuous Batching |
| 12 | [站在巨人肩上](12_Standard_Libraries_cuBLAS_cuFFT_Thrust.md) | 标准库 | cuBLAS 列主序陷阱、批量 GEMM 转置等价、cuFFT Plan 开销、Thrust 执行策略 |
| 13 | [GPU 诊断流](13_Performance_Analysis_Roofline_Occupancy.md) | 性能分析 | Roofline 拐点 (85.33 FLOP/Byte)、Occupancy 过度迷思、Nsight Compute 实战 |
| 14 | [工业级代码生成](14_CUTLASS_TemplateGEMM_CuTe.md) | CUTLASS 框架 | 四层抽象 (Arch→Instruction→Tile→Epilogue)、CuTe 代数布局、模板元编程 |
| 15 | [Ring AllReduce 数学](15_Multi_GPU_NCCL_AllReduce.md) | 多卡通信 | Parameter Server 星型拓扑瓶颈、Ring 拓扑线性度保证、NCCL 实现 |

## 推荐学习路线

根据学习目标，可以选择不同的阅读顺序：

### 路线一：从零入门 CUDA

> 适合初学者，建立完整的并行编程思维

```
01 基础概念 → 02 归约 → 03 前缀和 → 06 Warp 原语 → 10 访存优化
```

### 路线二：GEMM 优化专精

> 适合需要深入矩阵计算优化的开发者

```
01 Tiling 基础 → 04 寄存器分块 → 09 Tensor Core → 14 CUTLASS → 12 cuBLAS
```

### 路线三：LLM 推理系统

> 适合 AI 基础设施工程师

```
05 五大算子 → 07 量化 → 11 推理优化 → 08 系统调度 → 15 多卡通信
```

### 路线四：性能分析与调优

> 适合需要做性能诊断和优化决策的工程师

```
13 Roofline 与 Occupancy → 10 访存优化 → 04 GEMM 优化 → 12 标准库对标
```

## 内容特色

- **数学推导**：关键算法均给出完整的公式推导 (LaTeX)，如 Flash Attention 的 IO 复杂度、Ring AllReduce 通信量
- **硬件原理**：从 Cache Line、Bank、Warp 调度器等物理层面解释性能现象
- **反直觉案例**：揭示"高 Occupancy ≠ 高性能"、"异步拷贝不一定快"等工业经验
- **代码配套**：每篇文章对应项目同编号目录下的完整可编译 CUDA 源码
- **全中文撰写**：在英文资料主导的 GPU 优化领域提供系统化的中文技术解读
