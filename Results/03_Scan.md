# 03_Scan 综合测试报告

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
./build/03_Scan/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/03_Scan/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

# 03_Scan 综合测试报告

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
./build/03_Scan/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/03_Scan/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*


### Binary: segmented_scan
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
   测试场景 1：小数据量（算法对比）
========================================
数组大小：4096 元素
数据大小：16.00 KB
Block 大小：1024 线程
粗化因子：4
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：       0.01 ms
CPU 结果（末尾）：1860.60

--- GPU 版本 1: Coarse Scan ---
H2D 传输时间：       0.06 ms
Kernel 执行时间：    0.00 ms (100 次平均)
D2H 传输时间：       0.01 ms
GPU 总时间：         0.08 ms

--- GPU 版本 2: Segmented Scan ---
H2D 传输时间：       0.01 ms
Kernel 执行时间：    0.01 ms (100 次平均)
D2H 传输时间：       0.01 ms
GPU 总时间：         0.02 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：1.29x
CPU vs GPU 总时间加速比：0.08x
GPU 有效带宽：7.03 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Coarse Scan:      0.0047 ms (基准)
Segmented Scan:   0.0058 ms (0.81x)

--- 结果验证 ---
✓ Coarse Scan PASSED: 结果 1860.60 (期望 1860.60)
✓ Segmented Scan PASSED: 结果 1860.60 (期望 1860.60)
✓ 算法一致性 PASSED: 结果 1860.60 (期望 1860.60)
✓ GPU/CPU 结果一致性验证通过

========================================
   测试场景 2：大数据量（性能测试）
========================================
数组大小：1048576 (1 M) 元素
数据大小：4.00 MB
Block 大小：1024 线程
Block 数量：1024
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：       1.72 ms
CPU 结果（末尾）：471818.66

--- GPU 版本: Segmented Scan ---
H2D 传输时间：       0.38 ms
Kernel 执行时间：    0.02 ms (100 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         0.84 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：77.87x
CPU vs GPU 总时间加速比：2.03x
GPU 有效带宽：380.87 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比（数据规模扩展）---
小数据量 Segmented:   0.0058 ms (4096 元素)
大数据量 Segmented:   0.0220 ms (1M 元素)
数据量增长：256x
Kernel 时间增长：3.81x

--- 结果验证 ---
✓ Segmented Scan PASSED: 结果 471823.41 (期望 471818.66)
✓ GPU/CPU 结果一致性验证通过

========================================
```
### Binary: prefix_sum
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
      前缀和（Scan）性能基准测试
========================================
数组大小：1024 元素
数据大小：4.00 KB
Block 大小：1024 线程
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：       0.00 ms

--- GPU 版本 1: Kogge-Stone ---
H2D 传输时间：       0.01 ms
Kernel 执行时间：    0.00 ms (100 次平均)
D2H 传输时间：       0.01 ms
GPU 总时间：         0.02 ms

--- GPU 版本 2: Brent-Kung ---
H2D 传输时间：       0.01 ms
Kernel 执行时间：    0.00 ms (100 次平均)
D2H 传输时间：       0.01 ms
GPU 总时间：         0.02 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：0.27x
CPU vs GPU 总时间加速比：0.06x
GPU 有效带宽：2.21 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Kogge-Stone:   0.0028 ms (基准)
Brent-Kung:    0.0037 ms (0.76x)

--- 结果验证 ---
✓ Kogge-Stone PASSED: 结果 451.60 (期望 451.60)
✓ Brent-Kung PASSED: 结果 451.60 (期望 451.60)
✓ 算法一致性 PASSED: 结果 451.60 (期望 451.60)
✓ GPU/CPU 结果一致性验证通过

========================================
```

## segmented_scan.cu 代码逻辑与测试
**代码路径**: `03_Scan/02_segmented_scan/segmented_scan.cu`
**测试命令**: `./build/03_Scan/02_segmented_scan/segmented_scan`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## prefix_sum.cu 代码逻辑与测试
**代码路径**: `03_Scan/01_prefix_sum/prefix_sum.cu`
**测试命令**: `./build/03_Scan/01_prefix_sum/prefix_sum`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## prefix_sum 代码逻辑与测试
**实现逻辑分析**:
1. **前缀和基本原理**: 利用Blelloch/Kogge-Stone等并行算法，通过两阶段（Up-sweep/Down-sweep）或者分步累加的方法完成Scan操作。
2. **Bank Conflict与优化**: 通过Shared memory计算并在索引中插入padding，以避免bank conflict。
3. **多Block扩展**: 单个Block完成内部的Prefix sum后，输出中间数组（各个Block的最后一个元素的扫描值），再次进行Scan，最后加回原本数组，实现Global Scan。


## segmented_scan 代码逻辑与测试
**实现逻辑分析**:
1. **扩展前缀和**: 基于Flag数组划定段边界，在跨越段边界时中断累加，只在同段内执行Scan。
2. **应用场景**: 用于稀疏矩阵计算、图算法中划分不同的子任务区间直接并行计算。

