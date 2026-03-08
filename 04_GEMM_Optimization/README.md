# 04_GEMM_Optimization: 通用矩阵乘法的极致压榨

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

通用矩阵乘法（GEMM, General Matrix Multiply）是深度学习的核心支柱，无论是全连接层还是卷积运算底层均依赖它。本章的目标是如何通过对内存层级、指令并发和寄存器分布的深度理解，将一张显卡的算力彻底“压榨”出极限。它是从初学者迈向高性能架构师的一道门槛。

目录下的实现展示了经典的优化阶梯演进：

- `01_tiled_gemm/`：基于 2D Shared Memory 的基础分派策略（与基础篇的 Tiling 类似），为后续打下根基。
- `02_advanced_gemm/`：引入数据预取（Prefetching）、矩阵展平展开与内存合并（Coalesced Memory Access）机制。
- `03_register_tiling/`：最终级的常规 CUDA Core 优化阶段。将数据块进一步缓存在每个 Thread 私有的超高速寄存器中（Register Tiling），实现真正的算术逻辑单元（ALU）重算优化。

## 2. 原理推导与数学表达 (Math & Logic)

标准的矩阵乘法目标函数：
$$ C_{m, n} = \alpha \sum_{k=0}^{K-1} A_{m, k} B_{k, n} + \beta C_{m, n} $$
在 `Register Tiling` 时，我们要求一个线程不仅负责 1 个 $C$ 元素的写入，而是负责 $T_M \times T_N$ 块元素的计算。这样每一次从 Shared Memory 中读取 $T_M$ 个 $a$ 和 $T_N$ 个 $b$，它能复用于计算 $T_M \times T_N$ 个乘加指令：
数据读取量：$T_M + T_N$
计算量：$T_M \times T_N$
算术强度比（复用率）：$\frac{T_M \times T_N}{T_M + T_N}$  （随着 Tile 增大，复用率越高，显著降低对缓存的带宽需求）。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

展示 Register Tiling 优化架构的极致阶梯分配。一个线程持有 $8 \times 8$ 个寄存器进行计算：

```text
[Shared Mem 块 `sA` 128x16] 和 [`sB` 16x128] <-- (Block级协作)
             |                      |
             v                      v
        [读取 a 寄存器]        [读取 b 寄存器]
        (1x8 缓存)             (8x1 缓存)
             |                      |
             +--------\ /-----------+
                       v
         [完全驻留在 Thread 寄存器的计算结果 C_ij]
         [大小可能为 8x8，占用 64 个浮点寄存器] 
         [不需要中间读写任何内存，纯计算]
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `03_register_tiling/register_tiling.cu` 中对大矩阵内层乘并使用寄存器的逻辑：

```cpp
// 声明完全驻留在寄存器中的暂存数组（由编译器自动分配为 Register）
float frag_a[TM];
float frag_b[TN];
float accum[TM][TN] = {0.0f}; // 线程持有的 C 的子切片，比如 8x8 = 64 个浮点寄存器

for (int bk = 0; bk < BK; ++bk) {
    // 1. 各个线程从共同的 Shared Memory 把这轮需要的数据独吞进私有寄存器
    for (int i = 0; i < TM; ++i) frag_a[i] = s_A[ty * TM + i][bk];
    for (int j = 0; j < TN; ++j) frag_b[j] = s_B[bk][tx * TN + j];

    // 2. ✨ 全负荷进行 8x8 次 FMA （乘积累加）指令计算
    // 此时数据全在极速寄存器中，纯 ALU 吞吐运转
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            accum[i][j] += frag_a[i] * frag_b[j];
        }
    }
}
```

## 5. 性能基准与分析视角 (Performance & Profiling)

- **基准**：对比原生的 `cuBLAS` 内部非 Tensor Core （即纯 CUDA Core 的 SGEMM）实现。
- **典型分析**：这种 `Register Tiling` 完全转移了矛盾方向，原先极大限制速度的是 Shared / VRAM 的读取速度（Memory Bound）。经过此法后，瓶颈会过渡到计算密集（Compute Bound）。`ncu` 中会观察到极高的 SM `Pipe_ALU_Cycles_Active`，且能达到硬件峰值算力的 70%~90%。

## 6. 编译指引与参考资料 (Compile & References)

```bash
# 开启最高级别优化，防止自动展开被挂起
nvcc -O3 -arch=sm_89 register_tiling.cu -o run_gemm
# NCU 抽取指令的吞吐量占比和浮点算力峰值百分比
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active ./run_gemm
```

- 参考资料: NVIDIA CUDA GEMM Tuning Guide / CUTLASS 开源库的设计理念。
