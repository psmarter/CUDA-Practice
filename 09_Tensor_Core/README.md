# 09_Tensor_Core: 混合精度与 WMMA 矩阵引擎算力极限

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

至此，原有的纯 CUDA Core 数值标量 ALU 已不再能满足现代深度学习对于海量矩阵吞吐的要求。NVIDIA 自 Volta 架构引入了 Tensor Core 面向特殊矩阵的硬件电路阵列，它在一个极短的周期内即可吞噬 $16 \times 16 \times 16$ 规模的半精度矩阵乘加。本章标着你将叩开底层硬件黑核魔法的门扉，学习如何显式调用 WMMA (Warp Matrix Multiply Accumulate) API 来启动这台野兽级引擎。

目录涵盖了开启纯张量硬件特供管线的基础：

- `01_wmma_gemm/`：讲解如何包含 WMMA 头文件，声明特殊的寄存器碎片（Fragment），并以 Warp 为兵团粒度进行指令级别的矩阵装载、计算与存储。
- `02_mixed_precision/`：重点解析混合精度推演，如何在计算高频段内使用 FP16 高速吞吐，而在累加段保持 FP32 以遏制梯度消失与精度跌落的情况。

## 2. 原理推导与数学表达 (Math & Logic)

Tensor Core 执行的是形如：
$$ D = A \times B + C $$
的融合矩阵乘加计算。

在 Mixed Precision 的物理等量推导中，A 和 B 必须是 FP16 （或更低的格式），而累加结果则可以是更加饱满的数据类型：
$$ C_{\text{FP32}} = A_{\text{FP16}} \times B_{\text{FP16}} + C_{\text{FP32}}^{\text{old}} $$
这意味着张量核心电路内部包含了一个全精度的累加器（Accumulator），能够在执行 $4 \times 4$ 小块高频点积时立刻对齐指数位并归入宽类型结果中，既避免了软件层面的类型转换耗时，又锁死了精度。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

Tensor Core 的使用是绝对的**以 Warp (32 个线程)为物理执行单位**的，个别线程无法独立调用它。我们需要看它如何将数据切入 Fragment：

```text
[Shared Memory 中有一个 16x16 的 Tile_A (FP16)]

WMMA 显式装载 (Load Matrix Sync)
          ||
          \/
[ 32 个线程的内部构成了特殊的物理载体 Fragment ]
  +-----------------------------------------+
  | Thread 0 并不是独立拿某个标量，而是和同 Warp |
  | 的弟兄共同维护并指向某些交错分布在内部隐藏总 |
  | 线上的一部分 FP16。此时内存对于开发者透明了 |
  +-----------------------------------------+
          ||
wmma::mma_sync(frag_c, frag_a, frag_b, frag_c)
(硬件级的 Tensor Core 矩阵乘列阵瞬间激活，只需很少次时钟即完成)
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `01_wmma_gemm/wmma_gemm.cu`，启动核心硬件的关键指令：

```cpp
#include <mma.h>
using namespace nvcuda;

// 1. 声明张量片段，每个 Warp 去吃下 16x16x16 的标准子块
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c; // 精度回升为 Float 防溢出

// 2. 将 C_fragment 初始清理为 0
wmma::fill_fragment(frag_c, 0.0f);

for (int i = 0; i < K; i += 16) {
    // 3. 将 Share Memory 的数据协力加载入这些碎梦网格中
    wmma::load_matrix_sync(frag_a, sA_ptr, ldm_a);
    wmma::load_matrix_sync(frag_b, sB_ptr, ldm_b);
    
    // 4. ✨ 触发神迹的指令：硬件接管，激活 Tensor Core 作矩阵操作
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
}

// 5. 再次把结果反向吐出到物理寻址的内存系统
wmma::store_matrix_sync(C_ptr, frag_c, ldm_c, wmma::mem_row_major);
```

## 5. 性能基准与分析视角 (Performance & Profiling)

- **基准**：极其震撼的对比——哪怕是写得极为精致的 Register Tiling CUDA Core FP32 GEMM，在启动了 Tensor Core 的 FP16 GEMM 面前也是不堪一击。
- **典型分析**：在 NCU 中会看到一个极为夸张的新指标 `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`。当此指标升高，才证明你的代码突破了普通计算核心束缚而真正被送上了量身定制的高速车道。在 A100 / RTX 4090 架构上，可获得较标量计算高出几个数量级的 TFLOPS。

## 6. 编译指引与参考资料 (Compile & References)

```bash
# 架构底线是 sm_70 (Volta 架构引入 TensorCore1.0)
nvcc -O3 -arch=sm_89 wmma_gemm.cu -o run_wmma
# 必查 Tensor 核心硬件利用率与管线状态
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./run_wmma
```

- 参考资料: NVIDIA Developer Blog: "Programming Tensor Cores in CUDA 9+".
