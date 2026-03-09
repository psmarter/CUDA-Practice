# 09_Tensor_Core: 混合精度与 WMMA 矩阵引擎算力极限

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

自 Volta 架构起，NVIDIA 引入了突破标量 ALU 算力瓶颈的硬件电路阵列——Tensor Core（张量核心），专门面向深度学习的海量矩阵吞吐需求。Tensor Core 在极短的周期内即可吞噬 $16 \times 16 \times 16$ 规模的半精度矩阵乘加。本章将叩开底层硬件黑核魔法的门扉，学习如何显式调用 WMMA (Warp Matrix Multiply Accumulate) API 启动这台野兽级引擎。

目录涵盖了开启纯张量硬件特供管线的基础：
- `01_wmma_gemm/`：讲解如何包含 `mma.h` 头文件，声明特殊的寄存器碎片（Fragment），并以 Warp 为兵团粒度进行指令级别的矩阵装载、计算与存储。
- `02_mixed_precision/`：重点解析混合精度推演，即在计算高频段内使用 FP16 高速吞吐，而在累加段保持 FP32 以遏制梯度消失与精度跌落的情况。

## 2. 原理推导与数学表达 (Math & Logic)

Tensor Core 执行的是形如：
$$ D = A \times B + C $$
的融合矩阵乘加（MMA）计算。

在 Mixed Precision 的物理等量推导中，输入矩阵 A 和 B 通常是 FP16，而累加结果则可以是更宽的数据类型：
$$ C_{\text{FP32}} = A_{\text{FP16}} \times B_{\text{FP16}} + C_{\text{FP32}}^{\text{old}} $$

这意味着张量核心电路内部包含了一个全精度的累加器（Accumulator），能够在执行小块高频点积时立刻对齐指数位并归入宽类型结果中，既利用了 FP16 的双倍数据带宽和算力，又保持了数值稳定性。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

Tensor Core 的使用是绝对的**以 Warp (32 个线程)为物理执行单位**的，个别线程无法独立调用它。我们需要看它如何将数据切入 Fragment：

```text
[Global/Shared Memory 中有一个 16x16 的 Tile_A (FP16)]
          ||
          || WMMA 显式装载 (wmma::load_matrix_sync)
          \/
[ 32 个线程的内部构成了特殊的物理载体 Fragment ]
  +---------------------------------------------------------+
  | Thread 0 并不是独立拿某个标量，而是和同 Warp 的弟兄     |
  | 共同维护并指向某些交错分布在内部隐藏总线上的一部分 FP16 |
  | 此时寄存器和内存的对应关系对于开发者透明了              |
  +---------------------------------------------------------+
          ||
          || wmma::mma_sync(frag_c, frag_a, frag_b, frag_c)
          \/
(硬件级的 Tensor Core 矩阵乘列阵瞬间激活，极大提升 TFLOPS)
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `01_wmma_gemm/wmma_gemm.cu`，启动核心硬件的关键指令：

```cpp
#include <mma.h>
using namespace nvcuda;

// 1. 声明张量片段，每个 Warp 负责 16x16x16 的子块
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c; // 精度回升为 Float 防溢出

// 2. 将 C_fragment 初始清理为 0
wmma::fill_fragment(frag_c, 0.0f);

for (int i = 0; i < K; i += 16) {
    // 3. 将 Memory 的数据协力加载入这些碎片中
    wmma::load_matrix_sync(frag_a, A_ptr, lda);
    wmma::load_matrix_sync(frag_b, B_ptr, ldb);
    
    // 4. ✨ 触发硬件指令接管，激活 Tensor Core 作矩阵操作
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
}

// 5. 再次把结果反向吐出到物理寻址的内存系统
wmma::store_matrix_sync(C_ptr, frag_c, ldc, wmma::mem_row_major);
```

## 5. 性能基准与分析视角 (Performance & Profiling)

在高端 GPU 上进行 $2048 \times 2048 \times 2048$ 矩阵乘法测速：

- **01_wmma_gemm**：
  - CPU Base: ~100000 ms (嵌套 O(N^3) 开销巨大)
  - **GPU WMMA**: **0.58 ms**
  - **加速比**: **169999.28x**，计算吞吐达到 **29.21 TFLOPS**！

- **02_mixed_precision**：
  - **GPU FP32 (纯 CUDA Core)**: 0.4098 ms (5.24 TFLOPS)
  - **GPU WMMA Mixed Precision**: **0.0569 ms**
  - **混合精度算力**: **37.77 TFLOPS**，比传统 FP32 纯 CUDA Core 内核提速 **7.21x**。

哪怕是写得极为精致的 Register Tiling CUDA Core FP32 GEMM，在启动了 Tensor Core 的 FP16 GEMM 面前也是不堪一击。当查阅 Profiler 的 `sm__pipe_tensor_op_hmma_cycles_active` 指标升高时，才证明代码真正突破了普通计算核心束缚，跑在了量身定制的高速车道上。

## 6. 编译指引与参考资料 (Compile & References)

```bash
# 架构底线是 sm_70 (Volta 架构引入 TensorCore 1.0)
cd 09_Tensor_Core/02_mixed_precision
mkdir build && cd build
cmake ..
make
./mixed_precision

# 查阅 Tensor 核心硬件管线状态
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active ./mixed_precision
```

**参考资料:**
- [NVIDIA Developer Blog: Programming Tensor Cores in CUDA 9+](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [CUDA Toolkit Documentation: Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
