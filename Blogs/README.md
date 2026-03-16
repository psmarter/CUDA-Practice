# Blogs — CUDA 深度技术博客

本目录为 **CUDA-Practice** 配套的 16 篇中文技术文章（1 篇导读 + 15 篇专题），覆盖数学推导、硬件原理、Kernel 实现与性能分析，从带宽墙到多卡通信形成完整学习路线。

**在线博客**：**[Smarter's Blog — https://smarter.xin/](https://smarter.xin/)**  
系列文章在站内以「CUDA-Practice」分类发布，与仓库章节一一对应，推荐配合代码与 `Results/` 实测数据阅读。

---

## 文章索引

| # | 文章 | 主题 | 核心内容 |
| :---: | :--- | :--- | :--- |
| 00 | [00 系列导读与学习路线](CUDA实践-00-系列导读与学习路线.md) | 系列导读 | 15 篇专题依赖关系、适合人群、阅读顺序与推荐路线 |
| 01 | [01 基础概念与分块](CUDA实践-01-基础概念与分块.md) | 基础概念与分块 | 合并访存、带宽墙与 Roofline、Shared Memory Tiling、两次 __syncthreads |
| 02 | [02 归约与线程粗化](CUDA实践-02-归约与线程粗化.md) | 归约与线程粗化 | 树状归约、Warp Divergence 消除、收敛索引、多 Block atomicAdd、线程粗化、FMA 与 L2 |
| 03 | [03 前缀和与多块扫描](CUDA实践-03-前缀和与多块扫描.md) | Scan 算法 | Kogge-Stone vs Brent-Kung、双缓冲防 RAW、coarse_scan 与 3-Pass 多 Block |
| 04 | [04 矩阵乘优化与寄存器分块](CUDA实践-04-矩阵乘优化与寄存器分块.md) | GEMM 优化全链路 | 一维粗化 → 二维寄存器分块 → 外积累加 → Double Buffering |
| 05 | [05 大模型算子与注意力归一化](CUDA实践-05-大模型算子与注意力归一化.md) | 大模型算子 | Online Softmax、Welford LayerNorm、RMSNorm、RoPE、Flash Attention 原理 |
| 06 | [06 线程束原语与寄存器通信](CUDA实践-06-线程束原语与寄存器通信.md) | Warp 原语 | __shfl_xor / __shfl_up / __shfl_down、Block Reduce、Block Scan |
| 07 | [07 量化、半精度与整数推理](CUDA实践-07-量化半精度与整数推理.md) | 量化推理 | FP16 带宽、INT8 dp4a、Per-Tensor vs Per-Channel 量化 |
| 08 | [08 多流、图执行与扩展开发](CUDA实践-08-多流图执行与扩展开发.md) | 系统级调度 | Multi-Stream、CUDA Graphs、PyTorch C++ 扩展 |
| 09 | [09 张量核心与混合精度](CUDA实践-09-张量核心与混合精度.md) | 张量核心 | WMMA 16×16×16、Fragment、FP16 混合精度 |
| 10 | [10 访存优化与共享内存冲突](CUDA实践-10-访存优化与共享内存冲突.md) | 内存编排 | 合并访存、Bank Conflict 与 Padding |
| 11 | [11 推理优化、融合与键值缓存](CUDA实践-11-推理优化融合与键值缓存.md) | 推理系统 | Kernel Fusion、PagedAttention、Continuous Batching |
| 12 | [12 标准库与工程实践](CUDA实践-12-标准库与工程实践.md) | 标准库 | cuBLAS 列主序、cuFFT、Thrust 排序与归约 |
| 13 | [13 性能分析、屋顶线与占用率](CUDA实践-13-性能分析屋顶线与占用率.md) | 性能调优 | Roofline、Occupancy、Nsight Compute (ncu) |
| 14 | [14 模板矩阵乘与代数布局](CUDA实践-14-模板矩阵乘与代数布局.md) | CUTLASS | Template 抽象、CuTe Layout、元编程 |
| 15 | [15 多卡通信与全归约](CUDA实践-15-多卡通信与全归约.md) | 多卡通信 | Ring/Parameter Server、NCCL API |

## 推荐学习路线

- **入门**：00 导读 → 01 基础 → 02 归约 → 03 前缀和 → 06 Warp 原语 → 10 访存优化  
- **GEMM 专精**：00 → 01 → 04 寄存器分块 → 09 Tensor Core → 14 CUTLASS → 12 cuBLAS 验证  
- **性能诊断**：00 → 13 Roofline → 10 访存 → 04 GEMM 剖析 → 12 标准库  
- **LLM 推理**：00 → 05 大模型算子 → 07 量化 → 11 推理优化 → 08 多流 → 15 多卡

## 内容特点

- 数据驱动：用具体指标（如 28.2 ms、81.9 FLOP/B）替代模糊描述。  
- 理论/实测区分：标明 [理论] 与 [实测] 来源。  
- 代码锚点：公式与结论对应仓库中的 Kernel 或 `Results/` 报告。
