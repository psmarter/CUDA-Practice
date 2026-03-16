---
title: CUDA-Practice：13 Roofline 定天界，Occupancy 辨虚实——量化的 GPU 诊断流
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - 高性能计算
  - Roofline Model
  - Occupancy
  - ILP
  - Nsight Compute
  - 性能分析
  - Memory Bound
  - Compute Bound
categories:
  - CUDA-Practice
cover: /img/Nvidia_CUDA_Logo.jpg
abbrlink: 803b94d6
date: 2026-03-11 12:30:00
---

## 本文目标

读完本文，你将能够：

- 运用 Roofline 模型定量判断算子是受限于内存带宽（Memory Bound）还是计算能力（Compute Bound）
- 破除对 100% 占用率（Occupancy）的盲目追求，理解指令级并行（ILP）在隐藏高迟延访存时的物理依据
- 使用 Nsight Compute 的关键指标（如 `l1tex_t_sectors_per_request`）直接定位全局合并访存失效的根源

## 对应代码路径

> **硬件环境**：NVIDIA RTX 4090 (Ada Lovelace, sm_89)
> 128 SMs | FP32 82.6 TFLOPS | HBM 1008 GB/s | L2 72 MB | Roofline 拐点 81.9 FLOP/Byte

| 源文件 | Kernel 名称 | 核心技术 | 测试规模 |
|--------|-------------|----------|----------|
| `13_Performance_Analysis/02_roofline/roofline.cu` | `memory_bound_kernel` | Vector Add (Memory Bound 典型) | N=10,000,000 |
| `13_Performance_Analysis/02_roofline/roofline.cu` | `compute_bound_kernel` | Naive GEMM (Compute Bound 典型) | M=N=K=1024 |
| `13_Performance_Analysis/01_occupancy/occupancy.cu` | `configurable_kernel`<br>`shared_memory_kernel`<br>`register_limited_kernel`| 动态配置 Block Size 与 ILP，观察 Occupancy 和带宽影响 | N=10,000,000 |
| `13_Performance_Analysis/03_nsight_profiling/nsight_profiling.cu` | `profile_example_kernel_bad`<br>`profile_example_kernel_good` | 非合并访存 (Stride=32) 对比标准合并访存的性能探针 | N=10,000,000 |

> Kernel 名称与源码中 `__global__` 函数签名完全一致。
>
> **本篇在系列中的位置**：承接 [01 基础概念与分块](/posts/7608f1b0/)、[04 矩阵乘优化与寄存器分块](/posts/1a09f6f/)、[10 访存优化与共享内存冲突](/posts/5b6f891d/) 中对「算子实现与访存形态」的具体优化，本篇抽象出统一的 **性能建模与诊断视角**——通过 Roofline/Occupancy/Nsight 工具，回答「我的算子还可以快多少？」和「到底是算力不够还是带宽/实现出问题」。后续 [11 推理优化、融合与键值缓存](/posts/9729c03f/)、[12 标准库与工程实践](/posts/a1e20e80/) 会在推理系统与标准库层面复用这些分析方法。

## Baseline

**问题陈述**：在遇到性能未达预期的 Kernel 时，开发者常常通过猜测来添加 `__shared__` 或调整 `blockDim`。本篇博客不讨论具体业务，而是建立一套基于数据的诊断 Baseline：先通过 Roofline 计算天花板，再用代码实际跑分验证。

**Baseline 实现**：我们以 `roofline.cu` 中的 `memory_bound_kernel` (Vector Add) 和 `compute_bound_kernel` (Naive GEMM) 共同作为基准，对比它们的物理指标。

| 算子 | 指标 | 值 | 数据来源 |
|------|------|----|----------|
| Vector Add | Kernel 耗时 | 0.13 ms | [实测] Results/13_Performance_Analysis.md |
| Vector Add | 实际运行速度 | 78.72 GFLOPS | [实测] Results/13_Performance_Analysis.md |
| Naive GEMM | Kernel 耗时 | 0.41 ms | [实测] Results/13_Performance_Analysis.md |
| Naive GEMM | 实际运行速度 | 5.23 TFLOPS | [实测] Results/13_Performance_Analysis.md |

## 瓶颈分析

性能诊断的第一步，是计算算术强度（Arithmetic Intensity，$I$），即每搬运 1 字节数据能执行多少次浮点运算（FLOPs）。
我们将计算结果与 RTX 4090 的 **Roofline 拐点 81.9 FLOP/Byte [理论]** 结合对比。

### 案例 A：极限 Memory Bound 的 Vector Add

- 算术强度估算：每次计算读取 2 个 float（8 字节），写回 1 个 float（4 字节），总共 12 字节访存。计算为 1 次加法（1 FLOP）。
- 此算子的理论算术强度 $I = 1 / 12 \approx 0.083 \text{ FLOP/Byte}$ [理论]。
- 0.083 远小于拐点阈值 81.9，因此处于 **Memory Bound** 区域。
- 理论性能上限 $P = 0.083 \times 1008 \text{ GB/s} \approx 83.7 \text{ GFLOPS}$ [理论]。
- 我们的实测性能 78.72 GFLOPS 达到了理论峰值的 94%！这意味着显存带宽的物理潜力已经被全额透支。如果不减少算子的读写需求，无论如何修改 CUDA Core 指令均无法提速。

### 案例 B：虚假的 Compute Bound 的 Naive GEMM

- 算术强度估算：Naive GEMM 内层没有使用 Tiling 缓存复用。每次迭代计算都从全局显存重取数据。如果假定所有数据均完美通过缓存命中（仅最理想的情况），其算术强度 $I = 2N^3 / (3N^2 \times 4) = N / 6$。对于 N=1024，$I \approx 170.67 \text{ FLOP/Byte}$ [理论]。
- 这看似远大于 81.9，属于 Compute Bound。
- 理论天花板原本应为 $82.6 \text{ TFLOPS}$ [理论]。然而实测只有可怜的 5.23 TFLOPS。
- 瓶颈在于 $170.67$ 只是基于完美缓存的假象。由于 Naive 算法的显存重访率极高，实际传输的物理流量暴增，它实际跌回了 Memory Bound 斜坡底端。突破口必须是引入 Shared Memory 提高缓存命中（请参考 [04 矩阵乘优化与寄存器分块](/posts/1a09f6f/)）。

## 优化思路

通过上述 Roofline 的宏观判定，针对 Memory Bound 和由于不当执行引发性能跌落的情况，我们分别设计相关的实验来破除“常识”并提供纠错思路。

### 优化 1：利用 ILP（指令级并行）隐藏延迟

**解决的瓶颈**：破除对 100% Occupancy（高占用率才能掩盖延迟）的盲目追求。
**核心思想**：与其压缩单个线程资源来换取大量的并发 Warp 数，不如在同一个线程内放开手脚，通过展开循环引发多条无数据依赖的显存读取指令 (Instruction-Level Parallelism，ILP)，利用硬件内部加载单元的长流水线实现单核自我掩蔽。
**预期收益**：即便是在极低 Occupancy 下，也能跑满甚至超额榨取系统等效带宽 [理论]。

### 优化 2：彻底消灭非合并访存（Uncoalesced Memory Access）

**解决的瓶颈**：显存读取时的严重效率折损。
**核心思想**：避免同一 Warp 中不同线程的跳步跨越寻址，将跨步寻址重构为连续索引。以此适配底层缓存列强制 $32$ 字节包（Sector）硬性读取规则。
**预期收益**：极大提高 `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld` 指标健康度并成倍抬升带宽表现 [实测]。

## 关键代码解释

### ILP 极简并行发射

```cpp
// 来源：13_Performance_Analysis/01_occupancy/occupancy.cu : L8-L24
template<int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void configurable_kernel(CPFloat input, PFloat output, CInt n) {
    float items[ITEMS_PER_THREAD];
    
    int base_idx = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD;
    
    // [1] 利用 unroll 和寄存器数组形成密集独立的读取指令串，压发底层流水线
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = base_idx + i * BLOCK_SIZE + threadIdx.x;
        if (idx < n) {
            items[i] = input[idx];
        } else {
            items[i] = 0.0f;
        }
    }
    // ...
}
```

**要点解读**：

- `[1]`：编译器会将这里的 `#pragma unroll` 连同 `ITEMS_PER_THREAD` 展开成十几条连续且无上下游依赖的 `LDG.E` 加载汇编指令。这促使硬件不必等待上一跳加载返回即立刻下发后续读取请求，达成指令级别的深度并发响应（ILP）。这种方式下即便总活跃线程数很少，也能拉满甚至跑爆总线。

### 合并访存与跳跃寻址刺客

```cpp
// 来源：13_Performance_Analysis/03_nsight_profiling/nsight_profiling.cu : L68-L76
__global__ void profile_example_kernel_bad(CPFloat input, PFloat output, CInt n, CInt stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int chunk = n / stride; 
    // [1] 极其恶劣跳跃转换，强行错开同一 Warp 里的线性寻址
    int mapped_idx = (idx % stride) * chunk + (idx / stride);
    if (mapped_idx >= n) return;
    
    float val = input[mapped_idx];
```

**要点解读**：

- `[1]`：引入了一个基于输入 `stride=32` 的严重离散化重定向索引。这意味着 `thread0` 请求 `addr[0]` 时，`thread1` 会强行请求 `addr[chunk]`。Warp 的访存请求在物理板上支离破碎，最终导致有效数据利用率直线崩塌到不到百分之几（32 分之 1）。

## 结果与边界

### 性能对比

> **测试条件**：RTX 4090, CUDA 12.x, 迭代 100 次取平均值
> **数据来源**：`Results/13_Performance_Analysis.md` 原始日志

**1. Occupancy vs ILP 对抗 (配置均处理 10M 元素总量)**

| 版本 (每块线程人数, 每人承担包袱) | Kernel 耗时 | 理论计算 Occupancy | 等效带宽读写 | 数据性质 |
|-----------------------------------|-------------|------------------|--------------|----------|
| Config 1 `<256, 1>` (高满载无 ILP)  | 0.07 ms     | 100%             | 1230.12 GB/s | [实测]   |
| Config 2 `<256, 4>` (中等兼顾)      | 0.06 ms     | 100%             | 1324.67 GB/s | [实测]   |
| **Config 3 `<64, 16>` (极限 ILP)**  | **0.06 ms** | **100% (由于未超限)** | **1365.92 GB/s** | [实测]   |
| Config 4 `<256, 1>`伴随 32KB SMEM | 0.08 ms     | 50%              | 1020.48 GB/s | [实测]   |

在 Config 3 的高 ILP 推算中出现了超峰值带宽的 1365.92 GB/s [实测]。因为在连续的 100 轮循环测算中，这约 76 MB 的数据集碰巧命中了 RTX 4090 高达 72 MB 的 L2 Cache。内部 Cache 的瞬发吞吐远超 HBM，因此造就了表现爆表的有效读写。

**2. 探针下访存模式实验**

| 版本 | Kernel 耗时 | 有效测绘总线带宽 | 核心异常特征 | 数据性质 |
|------|-------------|----------------|--------------|----------|
| `profile_example_kernel_bad` | 0.29 ms | 273.54 GB/s | Stride=32 离散化 | [实测] |
| `profile_example_kernel_good`| 0.07 ms | 1227.03 GB/s | 极板标准合并访存 | [实测] |

完全相同执行结构中，仅仅只是因为将跳跃寻址变更回直接的线性对齐读取，整体吞吐量足足激增并产生碾压性级别的 **4.49 倍提升** [实测]。

### 边界条件与局限

- 当代码极其复杂包含巨量寄存器声明且不得不做高频切换时，若因为盲目去追求极度的高 ILP 将导致 `Spill to local memory` 现象发生（寄存器溢出回显存）。这将带来一次数百微秒的深重恶性迟延，反而不如利用一定的 Occupancy 实行多兵团切换妥当。

## 常见误区

1. **误区**：ncu 给我的算子定性为 Memory Bound（内存受限），所以我只能去买更好的显卡以获得更高的 HBM 带宽。
   **实际**：在基础算术中绝大部分初级计算皆呈现出 Memory Bound。但很多时候，内存总线上实际上运输的都是中间态的过程变量。我们可以通过算子融合机制（Kernel Fusion）去把这些原本落在外存上的变量合并留存在 SM 内部生命周期（例如 Shared Memory）里抵消掉，人为拔高算法的内生算术强度逼近拐点。

2. **误区**：为了获取高吞吐速度，我们编写所有核函数参数配置都应无脑调高使其 Occupancy 达成 100%。
   **实际**：Occupancy 本质只是掩盖迟延的一个选项之一。如果你的人均寄存器宽容度很大或者内存读取链存在很明显的 `#pragma unroll` 并发机会空间，就算降级牺牲一部分 Occupancy 取代之也依然能够跑爆系统的物理上限总线。

3. **误区**：不管我是 4 字节跳跃提取还是几十连号合并取，只要提取的数据总量是一样的带宽应该相差无几的。
   **实际**：这忽略了最致命的 GPU 显存底层通信最小包裹结构（Sector），硬件规定 L1 每发一次车硬规定打包 32 个字节（或多或少随构架演进略有区分）。你跳几步，这些包裹周边没利用到的数据都会同归于尽变成无用的垃圾吞吐流阻断大动脉循环，令真正承载带宽急剧跳水衰退。

## 系列导航

### 前置阅读

| 文章 | 与本篇的衔接 |
|------|--------------|
| [01 基础概念与分块](/posts/7608f1b0/) | 提供初步的 Memory Bound/Compute Bound 直觉，为本篇的 Roofline 数学化诊断做铺垫 |
| [04 矩阵乘优化与寄存器分块](/posts/1a09f6f/) | Naive GEMM vs Tiled GEMM 的性能差异，可以用本篇 Roofline/Occupancy 框架重新审视 |
| [10 访存优化与共享内存冲突](/posts/5b6f891d/) | 本篇指出问题（非合并访存、Bank/带宽瓶颈），10 章从 Global/Shared/Async Copy 三层给出具体解法 |

### 推荐后续

| 文章 | 与本篇的衔接 |
|------|--------------|
| [11 推理优化、融合与键值缓存](/posts/9729c03f/) | 在推理系统中用本篇的 Roofline/Occupancy 思路评估 Kernel Fusion、PagedAttention、Continuous Batching 的收益上界 |
| [12 标准库与工程实践](/posts/a1e20e80/) | 对比手写内核与标准库的性能时，直接用本篇提供的建模方法判断是否已接近硬件或库的Roofline |

---

## 顺序导航

- 上一篇：[CUDA实践-12-标准库与工程实践](/posts/a1e20e80/)
- 下一篇：[CUDA实践-14-模板矩阵乘与代数布局](/posts/f1b57921/)
