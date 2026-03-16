---
title: CUDA-Practice：09 硬件原生矩阵指令与混合精度协同
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - 高性能计算
  - Tensor Core
  - WMMA
  - 混合精度
  - FP16
  - GEMM
categories:
  - CUDA-Practice
cover: /img/Nvidia_CUDA_Logo.jpg
abbrlink: 78e375e8
date: 2026-03-11 12:00:00
---

## 本文目标

读完本文，你将能够：

- 理解 Tensor Core 的微架构原理及其对应的硬件级 WMMA（Warp Matrix Multiply-Accumulate）指令逻辑
- 掌握 `wmma::fragment` 这一核心抽象机制，理解数据在 Warp 内各线程间如何分布
- 理解 FP16 计算与 FP32 累加的混合精度（Mixed Precision）设计，以及它如何防止数值上溢和截断下溢
- 通过实测数据观察到 WMMA 优化带来的吞吐提升，并理解其物理天花板

## 对应代码路径

> **硬件环境**：NVIDIA RTX 4090 (Ada Lovelace, sm_89)
> 128 SMs | FP32 82.6 TFLOPS | HBM 1008 GB/s | L2 72 MB | Roofline 拐点 81.9 FLOP/Byte

| 源文件 | Kernel 名称 | 核心技术 | 测试规模 |
|--------|-------------|----------|----------|
| `09_Tensor_Core/01_wmma_gemm/wmma_gemm.cu` | `wmma_gemm_naive` | `<mma.h>` 命名空间、Fragment 存取 | M=N=K=2048 |
| `09_Tensor_Core/02_mixed_precision/mixed_precision.cu` | `gemm_fp32_kernel` | FP32 CUDA Core 基准测试 | M=N=K=1024 |
| `09_Tensor_Core/02_mixed_precision/mixed_precision.cu` | `wmma_mixed_gemm_kernel` | FP16 输入 + FP32 累加混合精度防溢出 | M=N=K=1024 |

> Kernel 名称与源码中 `__global__` 函数签名完全一致。
>
> **本篇在系列中的位置**：承接 [04 矩阵乘优化与寄存器分块](/posts/1a09f6f/) 的寄存器分块、[07 量化、半精度与整数推理](/posts/ef325d2f/) 的 FP16/INT8 数据类型，本篇从**硬件指令级**把 GEMM 从 CUDA Core 标量 FMA 升级到 **Tensor Core WMMA**（16×16×16 矩阵乘加），并说明混合精度（FP16 输入 + FP32 累加）如何兼顾吞吐与数值稳定。后续 [14 模板矩阵乘与代数布局](/posts/f1b57921/) 会在工业级模板下进一步压榨 WMMA；[11 推理优化、融合与键值缓存](/posts/9729c03f/) 则把 Tensor Core 置于完整推理流水线中理解。

## Baseline

**问题陈述**：在传统的 CUDA Core 优化中，即便是最极致的 Register Tiling，其算力瓶颈也被 82.6 TFLOPS 的物理天花板和单条 FMA 指令（仅处理两个标量元素）的效率所限制。当我们将矩阵乘法计算量推至极限时，单纯依靠标量指令无法满足庞大的算力缺口。

为了彰显 WMMA 的作用，我们首先建立一个基于 CUDA Core 的 FP32 标量实现作为基准。

**Baseline 实现**：`gemm_fp32_kernel`，位于 `09_Tensor_Core/02_mixed_precision/mixed_precision.cu`。此实现是一个朴素的、每个线程独立完成一个输出元素计算的实现逻辑。

| 指标 | 值 | 数据来源 |
|------|----|----------|
| Kernel 耗时 | 0.39 ms | [实测] Results/09_Tensor_Core.md |
| 有效算力 | 5.45 TFLOPS | [实测] Results/09_Tensor_Core.md |
| 有效访存带宽 | 31.96 GB/s | [实测] Results/09_Tensor_Core.md |

## 瓶颈分析

在 `gemm_fp32_kernel` 中，每个线程独立读取数据并执行 FMA，没有利用硬件提供的矩阵级指令加速能力：

1. **算术强度与指令效率不足**
   - Naive FP32 GEMM 中，单个线程每次读写所能贡献的计算量（算术强度）较低。
   - 每个时钟周期只能执行零散的标量运算。对于规模为 $M=N=K=1024$ 的操作，其计算量约为 2.14 GFLOPs，但受制于未优化的反复内存访问和低吞吐量的 FMA 标量指令发射，带宽仅跑出 31.96 GB/s [实测]。
   - RTX 4090 具有 ~165 TFLOPS（无稀疏）的 FP16 Tensor Core 峰值算力 [理论]，这意味现有普通核心实现的算力上限远低于实际硬件潜力。

## 优化思路

为了突破普通标量计算的瓶颈，我们引入 Tensor Core 的原生 WMMA 加速指令，并配合混合精度以保证结果准确。

### 优化 1：引入 WMMA 硬件级矩阵指令

**解决的瓶颈**：标量 FMA 指令计算效率低下。
**核心思想**：通过调用 `<mma.h>` 里的 `wmma::mma_sync` 等 API，强制 32 个线程的 Warp 联合完成一个 $16 \times 16 \times 16$ 尺寸矩阵乘加操作（$D = A \times B + C$）。这种操作只需极少的指令周期即可吞吐 8192 次浮点运算，大幅提高指令级并行度。
**预期收益**：极大缩短 Kernel 耗时，大幅提升等效计算量至 30 TFLOPS 以上 [理论]。

### 优化 2：采用混合精度防护算术溢出

**解决的瓶颈**：原生的纯 FP16 乘加会导致长链条累加下的数值截断错误，进而造成大数吃小数（Swamping）现象。
**核心思想**：让数据入口采用 FP16 读取以节省全局访存带宽，而在硬件内部执行矩阵乘积累加阶段采用浮点 32 位（FP32）的高位宽 `wmma::accumulator<float>`。
**预期收益**：在获得高运算吞吐和高能效带宽的同时，保证数值精度始终处于安全容差范围内 [理论]。

## 关键代码解释

以下代码展示了基于混合精度的 Tensor Core 实现逻辑：

```cpp
// 来源：09_Tensor_Core/02_mixed_precision/mixed_precision.cu : L23-L43
__global__ void wmma_mixed_gemm_kernel(const half* A, const half* B, float* C, CInt M, CInt N, CInt K) {
    // [1] 将当前 Warp 映射到输出矩阵 C 的 16x16 Tile 上
    CInt warpM = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_M;
    CInt warpN = (blockIdx.x * blockDim.x / 32) * WMMA_N; 

    if (warpM >= M || warpN >= N) return;

    // [2] 声明矩阵碎片：输入使用 half（FP16），累加器使用 float（FP32）
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // [3] 进行 K 维度的外层遍历滑动收割
    for (int k = 0; k < K; k += WMMA_K) {
        // Warp 内 32 个线程齐步协调，从全局显存装载 16x16 块的独立寄存器元素网格中
        wmma::load_matrix_sync(a_frag, A + warpM * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN, N);
        // [4] 执行 Tensor Core 内部混合乘加
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存储完成全精度积累后的最终瓦片回到 C 矩阵
    wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N, wmma::mem_row_major);
}
```

**要点解读**：

- `[1]`：每个 Warp 整建制地处理一块输出结果，所以 `warpN` 中使用 `blockDim.x / 32` 作为 Warp 的水平偏移量计算单位。
- `[2]`：`wmma::fragment` 不是一个可见的普通数组结构，它实际上在编译器底层会被解算成 32 个线程的私有寄存器集合。
- `[3]` 和 `[4]`：`wmma::load_matrix_sync` 和 `wmma::mma_sync` 均为自带屏障性质的 Warp 层级同步动作，所以代码流中并不需要单独再补充 `__syncwarp()`。

## 结果与边界

### 性能对比

> **测试条件**：RTX 4090, CUDA 12.x, 迭代 100 次取平均值
> **数据来源**：`Results/09_Tensor_Core.md` 原始日志

**1. 2048x2048 规模 Naive WMMA (FP16)**

| 版本 | Kernel 耗时 | 有效算力 | vs Baseline | 数据性质 |
|------|------------|----------------|-------------|----------|
| `wmma_gemm_naive` | 0.56 ms | 30.50 TFLOPS | - | [实测] |

**2. 1024x1024 规模 混合精度对决**

| 版本 | Kernel 耗时 | 有效算力 | 有效带宽 | vs FP32 Baseline | 数据性质 |
|------|------------|----------------|----------|------------------|----------|
| Baseline (纯 FP32) | 0.39 ms | 5.45 TFLOPS | 31.96 GB/s | 1.00x | [实测] |
| WMMA Mixed Precision | 0.05 ms | 39.36 TFLOPS | 153.73 GB/s | 7.21x | [实测] |

在使用同等直白矩阵拆解下，简单切换到硬件级 WMMA 并开启混合精度，即可使运算速度提升 7.21 倍。带宽也等比例放大到 153.73 GB/s [实测]。这是因为输入源从 4 Bytes 的 `float` 切分至 2 Bytes `half`，单次 Global Memory 请求取出的矩阵元素增加了一倍。

### 边界条件与局限

- **物理性能墙**：虽然在 1024 规模获得了 39.36 TFLOPS [实测]，但这与 RTX 4090 本身约 165 TFLOPS 的物理极限仍相差甚远。其根因依然在于数据缺乏高等级的 Shared Memory Tiling。硬件底层 `mma_sync` 发射时仍被 `load_matrix_sync` 来自 Global Memory 端的访存延迟长链路牵连，造成了庞大的 Tensor 核心时钟空转。
- **架构依赖**：WMMA 对于矩阵的尺寸（例如 M16N16K16 的形状）有严格的调用限制，如果矩阵实际长宽不贴合 16 或 8 的整倍数，则必须作矩阵边界外扩 (Padding) 补充 0。这会造成额外不必要的填充损失耗时。

## 常见误区

1. **误区**：利用 WMMA 加快运算时，只要直接传入纯 FP16 的类型矩阵进行全链条操作即可，无须配置 Mixed Precision。
   **实际**：对于任何包含数千深度迭代的长乘加运算链，连续的 FP16 + FP16 极易发生大量尾数信息在挤压对齐时因位数短缺而溢出或截断失效，造成模型数值异常。必须通过 FP16 + FP32 的混合精度来进行精度保障。

2. **误区**：为了获取一个 Fragment 上的所有数据以打印调试，可以直接用普通的数组索引迭代访问 `c_frag.x[i]`。
   **实际**：Fragment 数据是以系统抽象的模式被打碎分配到各线程私有寄存器中交织存在的，底层映射是封闭不可预先确定的，因此任何基于手动遍历提取元素的尝试拿到的是无实际业务意义的数据排列。

3. **误区**：有了 Tensor Core 加持，我可以用它无节制加速所有长条形向量乘法（如 `M=1` 的 Vector-Matrix 操作）。
   **实际**：Tensor Core 的流水线设计专攻正方形或小长方体的对称数据块（Tile）。使用它强行做高度失衡的向量操作可能连流水线利用率的零头都塞不满，不仅会白白损耗性能边界对齐与转换开销，在解码等特定推理场景下还不如使用传统的 CUDA 标量指令来的快速平稳。

## 系列导航

### 前置阅读

| 文章 | 与本篇的衔接 |
|------|--------------|
| [04 矩阵乘优化与寄存器分块](/posts/1a09f6f/) | 先掌握 CUDA Core 下的 Register Tiling、外积累加与 Double Buffering，再理解本篇如何用 WMMA 在**指令级**替代标量 FMA，实现同语义下的数量级算力跃升 |
| [07 量化、半精度与整数推理](/posts/ef325d2f/) | FP16 数据类型与带宽收益、数值范围与舍入；本篇 WMMA 的 FP16 输入 + FP32 累加正是该思路在 Tensor Core 上的落地 |

### 推荐后续

| 文章 | 与本篇的衔接 |
|------|--------------|
| [14 模板矩阵乘与代数布局](/posts/f1b57921/) | 工业级模板如何对 WMMA 做 Shared Memory Tiling、流水线与多级分块，把本篇的 Naive WMMA 推到接近硬件峰值 |
| [11 推理优化、融合与键值缓存](/posts/9729c03f/) | 推理系统中 Tensor Core 与算子融合、KV Cache 的配合，形成端到端 LLM 推理优化视角 |

---

## 顺序导航

- 上一篇：[CUDA实践-08-多流图执行与扩展开发](/posts/b1c0c6a3/)
- 下一篇：[CUDA实践-10-访存优化与共享内存冲突](/posts/5b6f891d/)
