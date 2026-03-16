# 12_Standard_Libraries 综合测试报告

## 一、 测试条件
- **测试环境**: 当前 Linux 编译环境 (包含 CMake, NVCC)
- **硬件配置**: 多卡环境 (2x NVIDIA GeForce RTX 4090, 架构 sm_89)
- **编译参数**: `nvcc` -O3, 标准 CUDA 运行时，启用 C++17
- **测试库依赖**: CUDA 原生库 (cuBLAS, cuFFT), CUTLASS, NCCL (针对多卡)

## 二、 测试方法与执行逻辑
本模块按以下维度整理，但需注意：`cublas_gemm.cu` 在 `M=N=K=1024` 时默认跳过 CPU 参考计算，因此其 `0.00 ms / 0.00x` 仅代表“未测占位”，不应视作真实 CPU/GPU 对比。
1. **正确性验证 (Correctness Check)**：在可承受规模下使用 CPU 计算出基准结果 (Reference) 与 GPU 结果进行对比；大矩阵样例可能显式跳过。
2. **基本耗时测量 (Timer)**：依赖 `cudaEventRecord` 或者 `std::chrono` 来统计算子的时间。
3. **安全与内存分析 (Compute Sanitizer)**：是否执行以具体子样例的日志记录为准。
4. **性能剖析探测 (Nsight Compute - ncu)**：是否执行以具体子样例的日志记录为准。

## 三、 测试命令模板
```bash
# 标准正确性运行
./build/12_Standard_Libraries/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/12_Standard_Libraries/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为真机二进制标准执行日志)*

## cublas_gemm.cu 代码逻辑与测试
**代码路径**: `12_Standard_Libraries/01_cublas_gemm/cublas_gemm.cu`
**测试命令**: `./build/12_Standard_Libraries/01_cublas_gemm/cublas_gemm`

**实现逻辑分析**:
1. **cuBLAS**: 使用 NVIDIA 官方高度优化的 cuBLAS 库。
2. **参考基准**: 为所有 GEMM 手写提供对比的上限天花板基准。

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
      cuBLAS GEMM 官方标准库性能测试
========================================
矩阵形状：M=1024, N=1024, K=1024
单算例数据量：12.00 MB
Batch 规模  ：8 个矩阵组合 (96.00 MB)
Kernel 迭代次数：50 次

--- CPU 计时 (M=1024, N=1024, K=1024) ---
CPU 单算例执行时间：       0.00 ms

--- GPU 版本 1: 基础 cublasSgemm ---
H2D 传输时间：       0.95 ms
Kernel 执行时间：    0.04 ms (50 次平均)
D2H 传输时间：       0.47 ms
GPU 总时间：         1.46 ms

--- GPU 版本 2: 启发式 cublasLtMatmul ---
H2D 传输时间：       0.91 ms
Kernel 执行时间：    0.04 ms (50 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         1.40 ms

--- GPU 版本 3: Strided Batched SGEMM (Batch=8) ---
H2D 传输时间：       6.41 ms
Kernel 执行时间：    0.45 ms (50 次平均)
D2H 传输时间：       3.30 ms
GPU 总时间：        10.17 ms

--- 性能分析 (TFLOPS) ---
CPU vs GPU (基础) 加速比：0.00x

cublasSgemm       算力：   49.91 TFLOPS
cublasLtMatmul    算力：   50.10 TFLOPS
StridedBatched    算力：   37.88 TFLOPS (相比单算例通常隐藏了 Kernel 启动开销)
(RTX 4090 FP32 理论峰值：~82.58 TFLOPS)

说明：M=N=K=1024 时 CPU 计算被视为过慢而跳过，因此 `CPU 单算例执行时间：0.00 ms` 与
`CPU vs GPU (基础) 加速比：0.00x` **仅为占位输出（表示“未测 / SKIPPED”）**，并非真实 CPU/GPU 对比。阅读或引用本节数据时，应只使用上方 cuBLAS 三行 TFLOPS 数值，忽略所有带有 0.00 ms / 0.00x 的占位字段。

--- 结果验证 ---
  [Skip] cublasSgemm	 validation for large matrices.
  [Skip] cublasLtMatmul	 validation for large matrices.
  [Skip] StridedBatched	 validation for large matrices.
✓ 本节大矩阵样例已显式跳过 GPU/CPU 逐元素验证；可确认的是 Row-Major 转置处理逻辑已在代码路径中实现，但本次日志不构成 CPU 基线验证通过的证据。

========================================

```

## cufft_example.cu 代码逻辑与测试
**代码路径**: `12_Standard_Libraries/02_cufft/cufft_example.cu`
**测试命令**: `./build/12_Standard_Libraries/02_cufft/cufft_example`

**实现逻辑分析**:
1. **cuFFT**: 快速傅里叶变换的官方实现。
2. **信号处理**: 对高吞吐的数字信号做快速处理。

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
      cuFFT 官方标准库 频谱计算测试
========================================
校验波长规模：4096 采样点 (0.0312 MB)
验证算法  ：GPU cuFFT vs CPU $O(N^2)$ 自研基准
Kernel 迭代次数：100 次

--- CPU 计时 (DFT $O(N^2)$) ---
CPU DFT 耗时：   395.0780 ms

--- GPU 版本 1: cuFFT 1D (Forward) ---
H2D 传输时间：     0.0150 ms
Kernel 执行时间：  0.0035 ms (100 次平均)
D2H 传输时间：     0.0123 ms
GPU 总时间：       0.0035 ms (纯算力段)

--- GPU 版本 2: cuFFT 1D (Inverse) ---
Kernel 执行时间：  0.0052 ms (100 次平均)

--- 性能分析与拓展基准 ---
CPU(Naive) vs GPU(cuFFT) 加速比（4096 维度下）：112156.50x
>> 注：这并不公平，因为 CPU 为 O(N^2) 而 GPU 基于 O(N log N) 库。仅仅用于功能对比和感受差距。

===> 测试大数据吞吐量 (Batch=65536, N=1024) <===
巨量 Batch Kernel 执行时间：    1.17 ms (10 次平均)
GPU 最小理论访存有效带宽：457.46 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- 结果验证 ---
✓ FFT (Forward) PASSED: 结果 (-9.83477e-07,0) (期望 (4.47035e-07,0))
✓ IFFT(Inverse) PASSED: 结果 (0.5,-1.49012e-07) (期望 (0.5,0))
✓ GPU/CPU 一致性验证通过！ (包含数学上的完美逆向重构功能验证)

========================================

```

## thrust_algorithms.cu 代码逻辑与测试
**代码路径**: `12_Standard_Libraries/03_thrust/thrust_algorithms.cu`
**测试命令**: `./build/12_Standard_Libraries/03_thrust/thrust_algorithms`

**实现逻辑分析**:
1. **Thrust**: 类似 C++ STL 的 CUDA 并行库。
2. **高效使用**: 提供了高速的 sort, reduce, scan 常规操作封包。

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
      Thrust 核心算法性能基准测试
========================================
数组大小：10000000 元素
数据大小：38.15 MB
Kernel 迭代次数：100 次

---------- [1] Sort (排序) ----------
CPU std::sort 时间： 2124.06 ms
GPU thrust::sort 时间：    1.30 ms (5次平均)
GPU Sort 加速比：1634.06x
✓ thrust::sort	 PASSED: 结果 0.00 (期望 0.00)

---------- [2] Reduce (归约 sum) ----------
CPU std::accumulate 时间：   28.35 ms
GPU thrust::reduce 时间：    0.08 ms (100 次平均)
GPU Reduce 加速比：371.31x
Reduce 有效带宽：487.88 GB/s
(RTX 4090 理论峰值：~1008 GB/s)
✓ thrust::reduce	 PASSED: 结果 500073024.00 (期望 500073024.00)

---------- [3] Transform (SAXPY元素级) ----------
CPU for-loop SAXPY 时间：   29.20 ms
GPU thrust::transform 时间：    0.13 ms (100 次平均)
GPU Transform 加速比：222.01x
Transform 有效带宽：849.73 GB/s
(RTX 4090 理论峰值：~1008 GB/s)
✓ thrust::transform PASSED: 结果 53.27 (期望 53.27)

========================================

```
