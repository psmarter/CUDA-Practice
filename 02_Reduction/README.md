# 02_Reduction: 并行规约与内存树优化

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

本章深入探讨“并行规约（Parallel Reduction）”问题。这是一种从 N 个元素中提取单个输出（如求和、求最大值）的基础算法。在 GPU 的并行架构上，规约直接关系到如何合理分配线程、避免线程空转（Warp Divergence）以及最大限度减少非连续地址访存冲突。掌握规约是深入学习诸如 Softmax 等复杂算子的前置技能。

目录下的实现从简入深，逐步消除架构瓶颈：

- `01_reduce_sum/`：朴素的树状层次规约实现，采用交错寻址的折叠思想，但存在严重的分支发散。
- `02_reduce_optimized/`：采用了连续寻址、展开最后一个 Warp（Unrolling the Last Warp）等优化手段，消除了 Bank Conflict 和不必要的 `__syncthreads()` 同步。
- `03_dot_product/`：演示如何在规约的同时结合按向量元素的乘法，实现两个向量的内积（Dot Product）。

## 2. 原理推导与数学表达 (Math & Logic)

求和规约的基础算式：
$$ S = \sum_{i=0}^{N-1} x_i $$
在并行树状计算中，如果以相邻元素两两折叠的策略（Stride 取一半逐渐缩小），其递推过程如下（设初始数据在 shared memory 数组 $s$ 中，当前宽度 $w$）：
$$ s[t] = s[t] + s[t + w] \quad (\text{其中 } t < w, \; \text{第一步 } w = B/2) $$
复杂度从串行的 $O(N)$ 下降到了 $O(\log N)$，然而如何分配线程使其物理执行最高效是工程重点。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

以连续寻址（Sequential Addressing）的优化规约过程为例。这种设计可以保证激活的 Thread 一定是连续的，从而完美避免了 Warp Divergence。

```text
[Shared Memory 长度为 8 (简化版)]
Thread ID:    T0  T1  T2  T3
              |   |   |   |
步骤 1: stride=4
s[0]+=s[4] <--+---+---+---+ 
s[1]+=s[5]        |       |
                 ...
步骤 2: stride=2  
s[0]+=s[2] <--+---+     
s[1]+=s[3] <--+   

   (最终保留在 s[0])
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `02_reduce_optimized/reduce_optimized.cu` 中对最后一条 Warp 进行展开（Warp Unrolling）的核心代码：

```cpp
// 步幅按每次减半进行普通规约，需要同步屏障
for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}

// ⚠️ Warp 级别展开极其关键：当步长缩小到 32 及以内时，它们都在同一个 Warp 内。
// 因为同一 Warp 的指令是 SIMT 锁步执行的，不再需要显式 __syncthreads() 也可以保证顺序。
// 这里使用 volatile 强制每次访存都能从 shared memory 获取最新值（而非寄存器旧值）。
if (tid < 32) {
    volatile float* vsmem = sdata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
}
```

## 5. 性能基准与分析视角 (Performance & Profiling)

- **基准**：对比朴素交错规约（Interleaved Addressing）。
- **典型分析**：使用 `ncu` 对比分析。通过展开最后的 Warp，不仅消除了一堆不必要的同步开销语句计算量，并且在连续寻址方式下彻底消灭了 Shared Memory 的 Bank Conflict 问题。整体的吞吐量指标对比无优化版本可提升 2 - 3 倍。

## 6. 编译指引与参考资料 (Compile & References)

```bash
nvcc -O3 -arch=sm_89 reduce_optimized.cu -o run_reduce
# 性能测试：捕捉同步指令消耗与分支情况
ncu --metrics sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active ./run_reduce
```

- 参考资料: Mark Harris 的经典演说《Optimizing Parallel Reduction in CUDA》。
