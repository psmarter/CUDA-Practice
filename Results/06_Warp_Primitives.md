# 06_Warp_Primitives 综合测试报告

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
./build/06_Warp_Primitives/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/06_Warp_Primitives/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

# 06_Warp_Primitives 综合测试报告

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
./build/06_Warp_Primitives/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/06_Warp_Primitives/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*


### Binary: warp_shuffle
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
      Warp Primitives 性能基准测试
========================================
数组大小：33554432 元素
数据大小：128.00 MB
Block 大小：256 线程
Kernel 迭代次数：100 次

--- CPU 计时 (计算所有版本) ---
CPU Broadcast 执行时间：   27.45 ms
CPU XOR       执行时间：   38.54 ms
CPU Up/Down   执行时间：   46.05 ms
CPU Reduce    执行时间：   41.22 ms

--- GPU 版本 1: Warp Broadcast ---
H2D 传输时间：      12.71 ms
Kernel 执行时间：    0.29 ms (100 次平均)
D2H 传输时间：      12.64 ms
GPU 总时间：        25.64 ms
✓ Warp Broadcast PASSED: 结果 0.66 (期望 0.66)

--- GPU 版本 2: XOR Shuffle ---
Kernel 执行时间：    0.29 ms (100 次平均)
✓ XOR Shuffle PASSED: 结果 0.92 (期望 0.92)

--- GPU 版本 3: Up/Down Shuffle ---
Kernel 执行时间：    0.29 ms (100 次平均)
✓ Up/Down Shuffle PASSED: 结果 0.40 (期望 0.40)

--- GPU 版本 4: Warp Reduce Sum ---
Kernel 执行时间：    0.15 ms (100 次平均)
✓ Warp Reduce Sum PASSED: 结果 15.27 (期望 15.27)

--- 性能分析 ---
CPU vs Broadcast Kernel 加速比：94.41x
CPU vs Reduce Sum Kernel 加速比：277.50x
Warp Broadcast 有效带宽：923.27 GB/s
Warp Reduce Sum 有效带宽：931.94 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Warp Broadcast:    0.2907 ms (基准)
XOR Shuffle:       0.2907 ms (1.00x)
Up/Down Shuffle:     0.29 ms (1.00x)
Warp Reduce Sum:     0.15 ms (高度内存密集与运算)

✓ GPU/CPU 结果一致性验证通过

========================================
```
### Binary: warp_reduce
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
      Warp & Block Reduce 性能基准测试
========================================
数组大小：33554432 元素
数据大小：128.00 MB
Block 大小：256 线程
Grid 大小：131072 Blocks
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU Reduce Sum 执行时间：   48.70 ms
CPU Reduce Max 执行时间：   50.89 ms

--- GPU 版本 1: Block Reduce Sum ---
H2D 传输时间：      12.89 ms
Kernel 执行时间：    0.14 ms (100 次平均)
D2H 传输时间：       0.08 ms
GPU 总时间：        13.11 ms
✓ Block Reduce Sum PASSED: 结果 129.86 (期望 129.86)

--- GPU 版本 2: Block Reduce Max ---
H2D 传输时间：      13.01 ms
Kernel 执行时间：    0.14 ms (100 次平均)
D2H 传输时间：       0.08 ms
GPU 总时间：        13.23 ms
✓ Block Reduce Max PASSED: 结果 0.98 (期望 0.98)

--- 性能分析 ---
CPU vs Reduce Sum 加速比：338.89x
CPU vs Reduce Max 加速比：354.22x
Reduce Sum 有效带宽：937.68 GB/s
Reduce Max 有效带宽：937.81 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

✓ GPU/CPU 结果一致性验证通过

========================================
```
### Binary: warp_scan
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
        Warp & Block Scan 性能基准测试
========================================
数组大小：33554432 元素
数据大小：128.00 MB
Block 大小：1024 线程 (支持最大 32 Warps 协作)
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU Inclusive Scan 执行时间：   51.61 ms
CPU Exclusive Scan 执行时间：   51.65 ms

--- GPU 版本 1: Block Inclusive Scan ---
H2D 传输时间：      12.74 ms
Kernel 执行时间：    0.30 ms (100 次平均)
D2H 传输时间：      12.40 ms
GPU 总时间：        25.44 ms
✓ Block Inclusive Scan PASSED: 结果 51.55 (期望 51.55)

--- GPU 版本 2: Block Exclusive Scan ---
H2D 传输时间：      12.68 ms
Kernel 执行时间：    0.30 ms (100 次平均)
D2H 传输时间：      12.36 ms
GPU 总时间：        25.35 ms
✓ Block Exclusive Scan PASSED: 结果 51.54 (期望 51.54)

--- 性能分析 ---
CPU vs Inclusive Scan 加速比：170.07x
CPU vs Exclusive Scan 加速比：170.29x
Inclusive Scan 有效带宽：884.57 GB/s
Exclusive Scan 有效带宽：884.96 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

✓ GPU/CPU 结果一致性验证通过

========================================
```

## warp_shuffle.cu 代码逻辑与测试
**代码路径**: `06_Warp_Primitives/01_warp_shuffle/warp_shuffle.cu`
**测试命令**: `./build/06_Warp_Primitives/01_warp_shuffle/warp_shuffle`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## warp_reduce.cu 代码逻辑与测试
**代码路径**: `06_Warp_Primitives/02_warp_reduce/warp_reduce.cu`
**测试命令**: `./build/06_Warp_Primitives/02_warp_reduce/warp_reduce`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## warp_scan.cu 代码逻辑与测试
**代码路径**: `06_Warp_Primitives/03_warp_scan/warp_scan.cu`
**测试命令**: `./build/06_Warp_Primitives/03_warp_scan/warp_scan`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```
