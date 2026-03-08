# 01_Basics: CUDA 基础编程与执行模型

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

本章是 CUDA 学习的起点，旨在让开发者熟悉 CUDA 的异构编程模型、线程层级结构（Grid, Block, Thread）以及最基础的全局内存（Global Memory）访问。它涵盖了如何从 CPU 侧向 GPU 发起计算请求，以及 GPU 计算内核（Kernel）的基本编写范式。这些是最基础也是最核心的 CUDA 概念。

目录下的实现逐步加深对计算与内存瓶颈的理解：

- `01_vector_add/`：演示最基本的单维度网格与线程块划分，以及内存的 Allocate、Copy 与 Free 流程。
- `02_matrix_mul_naive/`：演示二维线程网格映射，实现基础的矩阵乘法，暴露出未优化的全局内存的冗余访存问题。
- `03_matrix_mul_tiled/`：引入基于共享内存（Shared Memory）的分块（Tiling）策略，大幅减少全局内存宽带的浪费，为后续更高级的优化打下基础。

## 2. 原理推导与数学表达 (Math & Logic)

向量加法是最纯粹的 Element-wise 操作：
$$ C_i = A_i + B_i $$

对于矩阵乘法 $C = A \times B$（假设矩阵大小为 $M \times K$ 和 $K \times N$），目标元素计算式为：
$$ C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} \cdot B_{k, j} $$
在朴素实现中，这需要执行 $O(M \cdot N \cdot K)$ 次全局内存访问。而在 Tiling 优化中，我们将大矩阵分解为大小为 $B_s \times B_s$ 的小块，此时访存量缩减至原先的 $1/B_s$（理论上）。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

以 2D Tiling 矩阵乘法为例，使用 Shared Memory 降低全局内存访存带宽（假定 Block 大小为 $16 \times 16$）：

```text
[Global Memory] 矩阵 A 与 B
---------------------------------------------------
|                                                 |
|  [Shared Mem 块 `sA` 16x16]                      |
|  +--------------------+                         |
|  | Thread 映射区(0,0)  | <-- Block内的(tx, ty)   |
|  | 到(15, 15)协作加载   |                         |
|  +--------------------+                         |
---------------------------------------------------
           ||
           \/  (寄存器做内积累加)
           
计算局部 $C_{sub}$ 积累至对应的 Global Memory / Register
```

每个 Block 合作将数据从 Global Memory 读入 Shared Memory 后，需要调用 `__syncthreads()` 确保数据就绪，然后再执行乘累加。

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `03_matrix_mul_tiled/matrix_mul_tiled.cu` 的共享内存同步读取：

```cpp
// 声明 2D 共享内存，用于存储 A 和 B 的子块，利用高带宽和极低延迟
__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

// 协作加载：当前 Thread (tx, ty) 负责把全局内存中属于它的那个元素搬到共享内存
s_A[ty][tx] = A[Row * k + (ph * TILE_WIDTH + tx)];
s_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * n + Col];

// ⚠️ 极其关键的屏障：必须等该 Block 内部所有 Thread 完成搬运，才能开始计算
__syncthreads();

// 在共享内存上计算局部的乘积累加
for (int j = 0; j < TILE_WIDTH; ++j) {
    Cvalue += s_A[ty][j] * s_B[j][tx];
}

// ⚠️ 释放屏障：防止迭代过快导致下一轮的加载覆盖了当前还在计算的数据
__syncthreads();
```

## 5. 性能基准与分析视角 (Performance & Profiling)

- **基准**：对比 CPU 上的顺序循环矩阵乘法，以及简单的朴素版（Naïve）CUDA 矩阵乘法。
- **典型分析**：使用 Tiled 分块后，全局显存带宽的使用率大幅度降低。在 NCU 中观察 `sm__throughput` 和 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` 的比值，明显改善了 Memory Workload，使得计算受限（Compute Bound）的比重增加。

## 6. 编译指引与参考资料 (Compile & References)

```bash
# 通用编译指令
nvcc -O3 -arch=sm_89 matrix_mul_tiled.cu -o run_tiled
# NCU 性能分析指令（提取内存带宽和SM利用率）
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./run_tiled
```

- 参考资料: [CUDA C++ Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
