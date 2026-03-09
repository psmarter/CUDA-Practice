# 01_Basics: CUDA 编程基础与访存入门

## 1. 全局定位与学习价值
本目录作为 CUDA 学习的核心起手式，主要解决**执行模型认知**和**基础访存瓶颈**两大问题。
- **阶段位置**：通过 1D 向量加法和 2D 矩阵乘法，建立对 Grid、Block、Thread 层级的直观认知。
- **硬件瓶颈突破**：在朴素矩阵乘法（Naive GEMM）的基础上，引入共享内存（Shared Memory）与 Tiling 分块技术，大幅缓解全局内存（Global Memory）的访问延迟和带宽压力。
- **后续承载**：为后续的 `04_GEMM_Optimization`（算子优化）和 `10_Memory_Optimization`（访存极致优化）打下原理基础。

## 2. 子项目解析
### 2.1 Vector Add (`01_vector_add`)
- **核心概念**：1D 线程网格映射、Host-Device 数据搬运 (`cudaMemcpy`)。
- **验证数据** (RTX 4090, N=16777216)：
  - GPU/CPU 结果完全一致。

### 2.2 Naive Matrix Multiplication (`02_matrix_mul_naive`)
- **核心概念**：2D 线程网格映射。
- **性能缺陷**：每个线程独立读取矩阵的一行和一列，导致全局内存极度冗余访问。

### 2.3 Tiled Matrix Multiplication (`03_matrix_mul_tiled`)
- **核心概念**：共享内存 (Shared Memory) 与线程块内同步 (`__syncthreads()`)。
- **性能验证** (RTX 4090, 1024x1024 混合浮点)：
  - CPU 执行时间：~2105.81 ms （1.02 GFLOPS）
  - GPU Tiled 执行时间：~0.33 ms (6592.74 GFLOPS)
  - **加速比**：~6464x (Kernel纯计算)

## 3. 编译与运行
```bash
mkdir build && cd build
cmake ..
make -j
./01_Basics/03_matrix_mul_tiled/matrix_mul_tiled
```
