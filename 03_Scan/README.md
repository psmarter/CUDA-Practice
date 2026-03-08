# 03_Scan: 前缀和与数据依赖并行化

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

本章探讨并行的核心难题之一：扫描（Scan）/前缀和（Prefix Sum）。与简单的向量加法不同，前缀和具有强烈的元素前后依赖性（$k$ 位置的结果依赖于 $0$ 到 $k-1$ 位置的所有结果）。本章旨在学习如何在 GPU 这种高度并行的设备上，突破数据依赖瓶颈并将串行流程转换为并行平衡树形归纳结构。它在快速排序、数据规整（Stream Compaction）中有极其广泛的应用。

目录实现涵盖从标准扫描到分段变体：

- `01_prefix_sum/`：实现基于工作效率最优的 Blelloch 算法（上扫 Up-Sweep 与 下扫 Down-Sweep），解决标准一维数组的 exclusive / inclusive 前缀和。
- `02_segmented_scan/`：实现分段扫描，用来解决同一个数据流中存在多个独立分组子序列的并行累加问题。

## 2. 原理推导与数学表达 (Math & Logic)

对于 Exclusive Prefix Sum 命题：
给定输入序列 $x_0, x_1, x_2, \ldots, x_{n-1}$，
输出序列 $y_k = \sum_{i=0}^{k-1} x_i$，且 $y_0 = 0$。

为了满足工作数（Work Complexity）仍为 $O(N)$ 以比肩串行实现，我们利用了 Blelloch 算法。分为两步过程：

1. **构建树（Up-Sweep / Reduce）**：从叶子到根不断折叠相加，计算子树总和。
2. **分配树（Down-Sweep）**：
   - 将根节点强制置为 0。
   - 逐层下降，将父节点的旧值赋给左子节点，将父节点旧值加上原左子节点的值赋给右子节点。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

在处理大数组扫描（超出一个 Block 承载的宽度）时，需要借助辅助数组来传递不同 Block 之间的块级特征（Block Aggregate）。

```text
[Global Memory 映射全景]
Input  [ B0 (N 个元素) ] [ B1 (N 个元素) ] [ B2 (N 个元素) ]
              |                  |                 |
 (Kernel 1 内) Scan             Scan              Scan
              |                  |                 |
           产生总和 S0            S1                 S2
              \------------------------------------/
                           |
(Kernel 2 内)       对 [S0, S1, S2] 做 Scan 
                           |
                得到 [0, S0, S0+S1] 辅助数组 (Auxiliary)
                           |
(Kernel 3 内)    将对应的 Aux 置偏量（Offset）加回对应的块 B 中
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `01_prefix_sum/prefix_sum.cu` 解决共享内存 Bank Conflict 的经典技巧：

```cpp
// ⚠️ 宏定义：Bank Conflict 填充量
// 因为每一层步长的缩小都可能是2的幂次方，导致所有的线程都去访问同一 Bank （如 0, 32, 64）。
// 加入一个除以 NUM_BANKS 的偏移，能够恰好打乱内存对齐周期，使其错开。
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANK_BITS)

// 在写入共享内存时应用填充偏移：
int ai = offset * (2 * thid + 1) - 1;
int bi = offset * (2 * thid + 2) - 1;

// 计算真实的按位偏置后下标
ai += CONFLICT_FREE_OFFSET(ai);
bi += CONFLICT_FREE_OFFSET(bi);

// 无冲突读写
temp[bi] += temp[ai];
```

## 5. 性能基准与分析视角 (Performance & Profiling)

- **基准**：对比简单的串行 Prefix Sum（CPU 端 $O(N)$ 循环）和原生的未加内存冲突修正前的并行版本。
- **典型分析**：这充分反映了在 GPU 计算强依赖序列结构时的带宽妥协。使用了 `CONFLICT_FREE_OFFSET` 的版本能够彻底抹平由于 $2^N$ 步幅导致的 32-路 Shared Memory Bank Conflict，内存合并度（Memory Coalescing）极佳。

## 6. 编译指引与参考资料 (Compile & References)

```bash
nvcc -O3 -arch=sm_89 prefix_sum.cu -o run_scan
# 使用 ncu 检查 bank conflict 是否已被完全平息
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./run_scan
```

- 参考资料: GPU Gems 3, Chapter 39. "Parallel Prefix Sum (Scan) with CUDA".
