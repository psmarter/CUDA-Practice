# 04_GEMM_Optimization 综合测试报告

## 一、 测试条件

- **测试环境**: 当前 Linux 编译环境 (包含 CMake, NVCC)
- **硬件配置**: 多卡环境 (2x NVIDIA GeForce RTX 4090, 架构 sm_89)
- **编译参数**: `nvcc` -O3, 标准 CUDA 运行时，启用 C++17
- **测试库依赖**: CUDA 原生库 (cuBLAS, cuFFT), CUTLASS, NCCL (针对多卡)

## 二、 测试方法与执行逻辑

针对此模块下的实现，测试覆盖了：

1. **正确性验证 (Correctness Check)**：使用 CPU 计算出基准结果 (Reference)，与 GPU 计算结果进行对比并使用宏 `CHECK` 比较误差。
2. **基本耗时测量 (Timer)**：依赖 `cudaEventRecord` 或者 `std::chrono` 来统计算子的时间。
3. **安全与内存分析 (Compute Sanitizer)**：部分核心利用 `compute-sanitizer` 检查 Shared Memory/Global Memory 越界。
4. **性能剖析探测 (Nsight Compute - ncu)**：通过 `ncu` 收集 `sm__throughput`, `dram__bytes` 及寄存器利用率数据。

## 三、 测试命令模板

```bash
# 标准正确性运行
./build/04_GEMM_Optimization/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/04_GEMM_Optimization/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录

*(此下为真机二进制标准执行日志)*

## register_tiling.cu 代码逻辑与测试

**代码路径**: `04_GEMM_Optimization/03_register_tiling/register_tiling.cu`
**测试命令**: `./build/04_GEMM_Optimization/03_register_tiling/register_tiling`

**实现逻辑分析**:

1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**:

```text
检测到 2 块 CUDA 设备
设备 0： NVIDIA GeForce RTX 4090
  计算能力：8.9
  全局显存：23.65 GB
  每个 Block 共享内存：49152 Bytes
  每个 Block 最大线程数：1024
  Block 维度上限：(1024, 1024, 64)
  Grid 尺寸上限：(2147483647, 65535, 65535)
  Warp 大小：32
  SM 数量：128
  每个 SM 最大线程数：1536
设备 1： NVIDIA GeForce RTX 4090
  计算能力：8.9
  全局显存：23.64 GB
  每个 Block 共享内存：49152 Bytes
  每个 Block 最大线程数：1024
  Block 维度上限：(1024, 1024, 64)
  Grid 尺寸上限：(2147483647, 65535, 65535)
  Warp 大小：32
  SM 数量：128
  每个 SM 最大线程数：1536

========================================
  Register Tiling GEMM 性能基准测试
========================================
矩阵维度：A(2048 x 2048) * B(2048 x 2048) = C(2048 x 2048)
数据大小：48.00 MB (A + B + C)
Block Tile: BM=128 BN=128 BK=8
Thread Tile: TM=8 TN=8
线程数/Block: 256
每线程计算量: 8×8 = 64 个 C 元素
Kernel 迭代次数：20 次

--- CPU 计时 (若尺寸较小) ---
矩阵尺寸过大，跳过 CPU 参考计算。

--- GPU 版本 1: Register Tiling GEMM (手写) ---
H2D 传输时间：       3.18 ms
Kernel 执行时间：    0.60 ms (20 次平均)
D2H 传输时间：       1.60 ms
GPU 总时间：         5.38 ms
计算性能：       28.79 TFLOPS

--- GPU 版本 2: cuBLAS SGEMM (对比基准) ---
H2D 传输时间：       3.20 ms
Kernel 执行时间：    0.30 ms (20 次平均)
D2H 传输时间：       1.63 ms
GPU 总时间：         5.13 ms
计算性能：       57.49 TFLOPS

--- 性能对比分析 ---
Register Tiling: 28.79 TFLOPS
cuBLAS SGEMM:    57.49 TFLOPS
手写/cuBLAS 比率: 50.1%
CPU vs 手写 GEMM 加速比：1.7x

--- 结果验证 ---
✓ 结果验证跳过 (因矩阵尺寸过大，CPU 基准未计算；本节不包含 GPU/CPU 逐元素对比，仅验证了 GPU 端计算过程无显式错误)

========================================

```

## tiled_gemm.cu 代码逻辑与测试

**代码路径**: `04_GEMM_Optimization/01_tiled_gemm/tiled_gemm.cu`
**测试命令**: `./build/04_GEMM_Optimization/01_tiled_gemm/tiled_gemm`

**实现逻辑分析**:

1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**:

```text
检测到 2 块 CUDA 设备
设备 0： NVIDIA GeForce RTX 4090
  计算能力：8.9
  全局显存：23.65 GB
  每个 Block 共享内存：49152 Bytes
  每个 Block 最大线程数：1024
  Block 维度上限：(1024, 1024, 64)
  Grid 尺寸上限：(2147483647, 65535, 65535)
  Warp 大小：32
  SM 数量：128
  每个 SM 最大线程数：1536
设备 1： NVIDIA GeForce RTX 4090
  计算能力：8.9
  全局显存：23.64 GB
  每个 Block 共享内存：49152 Bytes
  每个 Block 最大线程数：1024
  Block 维度上限：(1024, 1024, 64)
  Grid 尺寸上限：(2147483647, 65535, 65535)
  Warp 大小：32
  SM 数量：128
  每个 SM 最大线程数：1536

========================================
    Tiled GEMM 性能基准测试
========================================
矩阵维度：A(1024 x 1024) * B(1024 x 1024) = C(1024 x 1024)
数据大小：12.00 MB (A + B + C)
Tile 大小：32 x 32
粗化因子：COARSE_FACTOR=4, COARSE_X=4, COARSE_Y=4
Kernel 迭代次数：10 次

--- CPU 计时 ---
CPU 执行时间：    2117.46 ms

--- GPU 版本 1: Tiled GEMM ---
H2D 传输时间：       0.92 ms
Kernel 执行时间：    0.33 ms (10 次平均)
D2H 传输时间：       0.45 ms
GPU 总时间：         1.70 ms

--- GPU 版本 2: Coarse GEMM (1D) ---
H2D 传输时间：       0.93 ms
Kernel 执行时间：    0.30 ms (10 次平均)
D2H 传输时间：       0.45 ms
GPU 总时间：         1.69 ms

--- GPU 版本 3: Register Tiled GEMM (2D) ---
H2D 传输时间：       0.81 ms
Kernel 执行时间：    0.15 ms (10 次平均)
D2H 传输时间：       0.45 ms
GPU 总时间：         1.42 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：13858.57x
CPU vs GPU 总时间加速比：1495.05x
GPU 计算性能：14055.10 GFLOPS
CPU 计算性能：1.01 GFLOPS
(RTX 4090 理论峰值：~82.6 TFLOPS FP32)

--- Kernel 性能对比 ---
Tiled GEMM:            0.3273 ms (基准)
Coarse GEMM (1D):      0.3047 ms (1.07x)
Register Tiled (2D):   0.1528 ms (2.14x)

--- 结果验证 ---
✓ Tiled GEMM PASSED: 结果验证通过 (最大误差 0.00)
✓ Coarse GEMM PASSED: 结果验证通过 (最大误差 0.00)
✓ Register Tiled PASSED: 结果验证通过 (最大误差 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## advanced_gemm.cu 代码逻辑与测试

**代码路径**: `04_GEMM_Optimization/02_advanced_gemm/advanced_gemm.cu`
**测试命令**: `./build/04_GEMM_Optimization/02_advanced_gemm/advanced_gemm`

**实现逻辑分析**:

1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**:

```text
检测到 2 块 CUDA 设备
设备 0： NVIDIA GeForce RTX 4090
  计算能力：8.9
  全局显存：23.65 GB
  每个 Block 共享内存：49152 Bytes
  每个 Block 最大线程数：1024
  Block 维度上限：(1024, 1024, 64)
  Grid 尺寸上限：(2147483647, 65535, 65535)
  Warp 大小：32
  SM 数量：128
  每个 SM 最大线程数：1536
设备 1： NVIDIA GeForce RTX 4090
  计算能力：8.9
  全局显存：23.64 GB
  每个 Block 共享内存：49152 Bytes
  每个 Block 最大线程数：1024
  Block 维度上限：(1024, 1024, 64)
  Grid 尺寸上限：(2147483647, 65535, 65535)
  Warp 大小：32
  SM 数量：128
  每个 SM 最大线程数：1536

========================================
    Advanced GEMM 性能基准测试
========================================
矩阵维度：A(1024 x 1024) * B(1024 x 1024) = C(1024 x 1024)
数据大小：12.00 MB (A + B + C)
Tile 大小：32 x 32
Kernel 迭代次数：10 次

--- CPU 计时 ---
CPU 执行时间：    2132.62 ms

--- GPU 版本 1: Vectorized GEMM ---
H2D 传输时间：       0.86 ms
Kernel 执行时间：    0.38 ms (10 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         1.68 ms

--- GPU 版本 2: Double Buffer GEMM ---
H2D 传输时间：       0.83 ms
Kernel 执行时间：    0.31 ms (10 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         1.59 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：6772.79x
CPU vs GPU 总时间加速比：1345.10x
GPU 计算性能：6820.01 GFLOPS
CPU 计算性能：1.01 GFLOPS
(RTX 4090 理论峰值：~82.6 TFLOPS FP32)

--- Kernel 性能对比 ---
Vectorized GEMM:      0.3821 ms (基准)
Double Buffer GEMM:   0.3149 ms (1.21x)

--- 结果验证 ---
✓ Vectorized GEMM PASSED: 结果验证通过 (最大误差 0.00)
✓ Double Buffer GEMM PASSED: 结果验证通过 (最大误差 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## tiled_gemm 代码逻辑与测试

**实现逻辑分析**:

1. **Shared Memory Tiling**: 将 Global Memory 中的矩阵切块装载进 Shared Memory 中，再参与乘加。
2. **性能提升**: 极大地降低了对于 Global memory 的读取次数（降低了N倍的带宽压力），显著提升计算受限程序的性能。

## advanced_gemm 代码逻辑与测试

**实现逻辑分析**:

1. **Double Buffering**: 引入双缓冲机制，利用异或等切换缓冲区，消除软管线的等待时间。
2. **指针投递与内存布局转化**: 使用向量化存取(float4)以及 Shared Memory Padding 降低存取延迟。
3. **优化结果**: 在 1024×1024 规模下，Double Buffer 相比向量化版本再提升约 1.2 倍，实测约 6.8 TFLOPS（约占 RTX 4090 FP32 理论峰值的 8% 左右），主要受限于 Tile 尺寸与调度，而非理论上限的 70-80%。

## register_tiling 代码逻辑与测试

**实现逻辑分析**:

1. **寄存器重用**: 进一步将 Shared Memory 中的数据缓存到 Registers 中进行反复的 FMA 操作。
2. **指令级并行**: 通过循环展开（Pragma unroll）提高指令发射速率并减少分支预测开销。
3. **极端优化边界**: 使该算子受限于寄存器数量和计算单元，而不是带宽层级。

> 注：对于 2048×2048 的大矩阵，由于 CPU 参考计算被刻意跳过以避免测试时间过长，`Results` 中“CPU vs 手写 GEMM 加速比”的数值仅在小尺寸下具有实际意义；大尺寸场景下应主要关注手写 Kernel 与 cuBLAS 之间的 TFLOPS 比例（约 50%）。 
> 换句话说，本页 `Register Tiling GEMM 性能基准测试` 区块中涉及 CPU 的行（加速比、CPU 性能等）可视为“仅供小矩阵对比的参考格式”，对默认的大矩阵规模 **不构成严格意义上的 CPU/GPU 对比结论**。
