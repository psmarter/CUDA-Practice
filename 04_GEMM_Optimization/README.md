# 04_GEMM_Optimization: 通用矩阵乘法极致优化

## 1. 全局定位与学习价值
通用矩阵乘法 (GEMM) 是深度学习和科学计算的地基。本章节从 `01_Basics` 的 Shared Memory Tiling 出发，全景展示了如何榨干 GPU 的极限性能。
- **阶段位置**：通过访存结构、指令级并行、以及层级缓存最大化利用，它是通向 CUTLASS 与 LLM 算子架构（如 FlashAttention）的分水岭。
- **硬件瓶颈突破**：揭示并解决各种内存墙问题。从 Shared Memory 利用 -> 寄存器复用 (Register Tiling) -> 内存向量化读取 (Vectorized Fetching) -> 异步机制的初步探索 (Double Buffering)。
- **后续承载**：为 `07_Quantization` 和 `14_CUTLASS` 打下坚实的底层微架构调度认知。

## 2. 子项目解析
### 2.1 逐级 Tiling (`01_tiled_gemm`)
回顾了基础的 Tiled GEMM，并扩展至 **Thread Coarsening**（1D与2D寄存器分块初步）。
- **优化点**：用更少的线程做更多的工作。1D粗化（横向展开）和2D粗化使得单个线程无需反复进入共享内存中索取数据，利用快速的寄存器暂存 A 或 B 的数据进行多次积加！
- **性能标尺** (RTX 4090, 1024x1024)：
  - 基础 Tiled：~14055 GFLOPS
  - Register Tiled (2D)：相比基础 Tiled 再加速 **2.14x** 达到 0.15ms。

### 2.2 高级访存技术 (`02_advanced_gemm`)
介绍如何更好地从 Global Memory 提取数据。
- **Vectorized Fetching**：使用 `float4` （128-bit）读写指令，利用最宽的数据通道增加带宽利用率，减少发出的加载指令数。
- **Double Buffering**：隐藏访存延迟（Latency Hiding）。在计算第一块数据时，预先将第二块数据加载到共享内存的另一半中，进一步打满流水线。

### 2.3 寄存器极致分块 (`03_register_tiling`)
这是手动编写高水平 GEMM 的终极战役（在没有使用 MMA/Tensor Core 前）。核心思想是通过**共享内存分块 (Block-level) + 寄存器分块 (Thread-level)** 将全矩阵分解到微型核（Micro-kernel）。
- **架构数据**：Block Tile为 `128x128`，Thread Tile为 `8x8`。每个线程独立负责输出区域中的 $8 \times 8 = 64$ 个值。
- **性能对齐** (RTX 4090, 2048x2048):
  - CPU 性能：0.74 GFLOPS / 耗时 ~23073 ms
  - 纯手写 Register Tiling：**28.86 TFLOPS** / 耗时 ~0.60 ms
  - cuBLAS (业界标杆)：**57.63 TFLOPS**
  - **成绩单**：不利用汇编与特殊的 Tensor Core，纯 CUDA C++ 实现了官方商用库一半以上的性能，计算提速达 **38760x**！

## 3. 编译与运行
```bash
cd build
cmake ..
make tiled_gemm advanced_gemm register_tiling
./04_GEMM_Optimization/01_tiled_gemm/tiled_gemm
./04_GEMM_Optimization/02_advanced_gemm/advanced_gemm
./04_GEMM_Optimization/03_register_tiling/register_tiling
```
