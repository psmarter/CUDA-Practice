# 01_Basics 核心概念与共享内存优化解析

## 1. 背景与动机
在 CUDA 编程中，计算能力往往不是瓶颈，访存（Memory Access）才是。本篇博客详细拆解了如何从一维的向量加法，跨越到二维的朴素矩阵乘法，最后通过 Tiling 技术压榨硬件性能。

## 2. 核心优化：Tiling 与 Shared Memory
在朴素的矩阵乘法中，为了计算 $C$ 的一个元素，我们需要从全局内存中读取 $A$ 的一行和 $B$ 的一列。如果有 1024 个元素，全局内存将被重复读取 1024 次。
通过引入 **Shared Memory**（L1 级缓存，位于 SM 内部，速度极快）：
- 我们让一个 Block 协作把数据加载到 Shared Memory。
- 使用 `__syncthreads()` 等待加载完成。
- Block 内的线程直接从 Shared Memory 读取数据参与计算。

## 3. 性能验证闭环
在 RTX 4090 上，针对 1024x1024 的矩阵乘法进行实测：
- CPU (单线程基础版)：~2105 ms
- GPU (Tiled)：~0.33 ms
- GPU 总吞吐达到了 6592 GFLOPS。这说明 Tiling 技术极为有效地接管了内存数据馈送，让运算单元不至于长时间闲置（stall）。

## 4. 总结与展望
掌握 Tiling 是进军后续更高级的 GEMM 优化（如 Register Tiling, WMMA/Tensor Core）的必经之路。在 `04_GEMM_Optimization` 中，我们将进一步榨干寄存器级别的局部性。
