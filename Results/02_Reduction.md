# 02_Reduction 综合测试报告

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
./build/02_Reduction/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/02_Reduction/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

# 02_Reduction 综合测试报告

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
./build/02_Reduction/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/02_Reduction/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*


### Binary: reduce_sum
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
      Reduce Sum 性能基准测试
========================================
数组大小：2048 元素
数据大小：0.0078 MB
Block 大小：1024 线程
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：     0.0030 ms

--- GPU 版本 1: Simple Reduce ---
H2D 传输时间：     0.0082 ms
Kernel 执行时间：  0.0052 ms (100 次平均)
D2H 传输时间：     0.0084 ms
GPU 总时间：       0.0218 ms

--- GPU 版本 2: Convergent Reduce ---
H2D 传输时间：     0.0061 ms
Kernel 执行时间：  0.0037 ms (100 次平均)
D2H 传输时间：     0.0058 ms
GPU 总时间：       0.0157 ms

--- GPU 版本 3: Shared Memory Reduce ---
H2D 传输时间：     0.0062 ms
Kernel 执行时间：  0.0038 ms (100 次平均)
D2H 传输时间：     0.0045 ms
GPU 总时间：       0.0146 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：0.78x
CPU vs GPU 总时间加速比：0.21x
GPU 有效带宽：2.13 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Simple:        0.0052 ms (基准)
Convergent:    0.0037 ms (1.40x)
Shared Mem:      0.00 ms (1.35x)

--- 结果验证 ---
✓ Simple Reduce PASSED: 结果 2048.00 (期望 2048.00)
✓ Convergent Reduce PASSED: 结果 2048.00 (期望 2048.00)
✓ Shared Mem Reduce PASSED: 结果 2048.00 (期望 2048.00)
✓ GPU/CPU 结果一致性验证通过

========================================
```
### Binary: reduce_optimized
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
   Reduce Optimized 性能基准测试
========================================
数组大小：1048576 (1 M) 元素
数据大小：4.00 MB
Block 大小：1024 线程
粗化因子：4
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：       4.64 ms
CPU 求和结果：   519036.00
CPU 最大值结果： 0.99

--- GPU 版本 1: Segmented Reduce Sum ---
H2D 传输时间：       0.42 ms
Kernel 执行时间：    0.01 ms (100 次平均)
D2H 传输时间：       0.01 ms
GPU 总时间：         0.44 ms

--- GPU 版本 2: Coarsened Reduce Sum ---
H2D 传输时间：       0.41 ms
Kernel 执行时间：    0.00 ms (100 次平均)
D2H 传输时间：       0.01 ms
GPU 总时间：         0.42 ms

--- GPU 版本 3: Coarsened Reduce Max ---
H2D 传输时间：       0.39 ms
Kernel 执行时间：    0.00 ms (100 次平均)
D2H 传输时间：       0.00 ms
GPU 总时间：         0.40 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：983.47x
CPU vs GPU 总时间加速比：10.95x
GPU 有效带宽：889.77 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Segmented:     0.0083 ms (基准)
Coarsened:     0.0047 ms (1.77x)

--- 结果验证 ---
✓ Segmented Sum PASSED: 结果 519035.97 (期望 519036.00)
✓ Coarsened Sum PASSED: 结果 519035.97 (期望 519036.00)
✓ Coarsened Max PASSED: 结果 0.99 (期望 0.99)
✓ GPU/CPU 结果一致性验证通过

========================================
```
### Binary: dot_product
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
      Dot Product 性能基准测试
========================================
数组大小：1048576 (1 M) 元素
数据大小：8.00 MB (两向量)
Block 大小：1024 线程
粗化因子：4
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：       1.74 ms
CPU 结果：       1048576.00

--- GPU 版本 1: Simple Dot Product ---
H2D 传输时间：       0.82 ms
Kernel 执行时间：  0.0092 ms (100 次平均)
D2H 传输时间：     0.0081 ms
GPU 总时间：       0.8412 ms

--- GPU 版本 2: Coarsened Dot Product ---
H2D 传输时间：     0.8390 ms
Kernel 执行时间：  0.0055 ms (100 次平均)
D2H 传输时间：     0.0059 ms
GPU 总时间：       0.8504 ms

--- GPU 版本 3: FMA Dot Product ---
H2D 传输时间：     0.7697 ms
Kernel 执行时间：  0.0055 ms (100 次平均)
D2H 传输时间：     0.0049 ms
GPU 总时间：       0.7800 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：319.13x
CPU vs GPU 总时间加速比：2.23x
GPU 有效带宽：1535.88 GB/s
(RTX 4090 DRAM 理论峰值：~1008 GB/s，L2 峰值更高)

--- Kernel 性能对比 ---
Simple:      0.0092 ms (基准)
Coarsened:   0.0055 ms (1.68x)
FMA:         0.0055 ms (1.68x)

--- 结果验证 ---
✓ Simple PASSED: 结果 1048576.00 (期望 1048576.00)
✓ Coarsened PASSED: 结果 1048576.00 (期望 1048576.00)
✓ FMA PASSED: 结果 1048576.00 (期望 1048576.00)
✓ GPU/CPU 结果一致性验证通过

========================================
```

## reduce_sum.cu 代码逻辑与测试
**代码路径**: `02_Reduction/01_reduce_sum/reduce_sum.cu`
**测试命令**: `./build/02_Reduction/01_reduce_sum/reduce_sum`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## reduce_optimized.cu 代码逻辑与测试
**代码路径**: `02_Reduction/02_reduce_optimized/reduce_optimized.cu`
**测试命令**: `./build/02_Reduction/02_reduce_optimized/reduce_optimized`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## dot_product.cu 代码逻辑与测试
**代码路径**: `02_Reduction/03_dot_product/dot_product.cu`
**测试命令**: `./build/02_Reduction/03_dot_product/dot_product`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```
