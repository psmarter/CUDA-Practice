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
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

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
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*


### Binary: register_tiling
#### Standard Execution & CUDA Timer
```text
```
### Binary: tiled_gemm
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
    Tiled GEMM 性能基准测试
========================================
矩阵维度：A(1024 x 1024) * B(1024 x 1024) = C(1024 x 1024)
数据大小：12.00 MB (A + B + C)
Tile 大小：32 x 32
粗化因子：COARSE_FACTOR=4, COARSE_X=4, COARSE_Y=4
Kernel 迭代次数：10 次

--- CPU 计时 ---
CPU 执行时间：    2112.32 ms

--- GPU 版本 1: Tiled GEMM ---
H2D 传输时间：       0.85 ms
Kernel 执行时间：    0.33 ms (10 次平均)
D2H 传输时间：       0.45 ms
GPU 总时间：         1.63 ms

--- GPU 版本 2: Coarse GEMM (1D) ---
H2D 传输时间：       0.82 ms
Kernel 执行时间：    0.30 ms (10 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         1.57 ms

--- GPU 版本 3: Register Tiled GEMM (2D) ---
H2D 传输时间：       0.79 ms
Kernel 执行时间：    0.15 ms (10 次平均)
D2H 传输时间：       0.45 ms
GPU 总时间：         1.39 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：13832.22x
CPU vs GPU 总时间加速比：1514.58x
GPU 计算性能：14062.46 GFLOPS
CPU 计算性能：1.02 GFLOPS
(RTX 4090 理论峰值：~82.6 TFLOPS FP32)

--- Kernel 性能对比 ---
Tiled GEMM:            0.3273 ms (基准)
Coarse GEMM (1D):      0.3046 ms (1.07x)
Register Tiled (2D):   0.1527 ms (2.14x)

--- 结果验证 ---
✓ Tiled GEMM PASSED: 结果验证通过 (最大误差 0.00)
✓ Coarse GEMM PASSED: 结果验证通过 (最大误差 0.00)
✓ Register Tiled PASSED: 结果验证通过 (最大误差 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================
```
### Binary: advanced_gemm
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
    Advanced GEMM 性能基准测试
========================================
矩阵维度：A(1024 x 1024) * B(1024 x 1024) = C(1024 x 1024)
数据大小：12.00 MB (A + B + C)
Tile 大小：32 x 32
Kernel 迭代次数：10 次

--- CPU 计时 ---
CPU 执行时间：    2077.47 ms

--- GPU 版本 1: Vectorized GEMM ---
H2D 传输时间：       0.84 ms
Kernel 执行时间：    0.38 ms (10 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         1.67 ms

--- GPU 版本 2: Double Buffer GEMM ---
H2D 传输时间：       0.83 ms
Kernel 执行时间：    0.32 ms (10 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         1.59 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：6584.80x
CPU vs GPU 总时间加速比：1310.63x
GPU 计算性能：6806.73 GFLOPS
CPU 计算性能：1.03 GFLOPS
(RTX 4090 理论峰值：~82.6 TFLOPS FP32)

--- Kernel 性能对比 ---
Vectorized GEMM:      0.3826 ms (基准)
Double Buffer GEMM:   0.3155 ms (1.21x)

--- 结果验证 ---
✓ Vectorized GEMM PASSED: 结果验证通过 (最大误差 0.00)
✓ Double Buffer GEMM PASSED: 结果验证通过 (最大误差 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================
```

## register_tiling.cu 代码逻辑与测试
**代码路径**: `04_GEMM_Optimization/03_register_tiling/register_tiling.cu`
**测试命令**: `./build/04_GEMM_Optimization/03_register_tiling/register_tiling`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

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
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

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
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```
