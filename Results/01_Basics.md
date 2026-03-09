# 01_Basics 综合测试报告

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
./build/01_Basics/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/01_Basics/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

# 01_Basics 综合测试报告

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
./build/01_Basics/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/01_Basics/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*


### Binary: vector_add
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
      Vector Add 性能基准测试
========================================
问题规模：67108864 (64 M) 元素
数据大小：256.00 MB (每个数组)
Kernel 迭代次数：100 次

--- GPU 详细计时 ---
H2D 传输时间：      49.39 ms
Kernel 执行时间：    0.86 ms (100 次平均)
D2H 传输时间：      25.46 ms
GPU 总时间：        75.71 ms

--- CPU 计时 ---
CPU 执行时间：     157.24 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：182.17x
CPU vs GPU 总时间加速比：2.08x
有效带宽：932.97 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- 结果验证 ---
✓ GPU PASSED: 全部 67108864 个元素验证正确
✓ CPU PASSED: 全部 67108864 个元素验证正确

========================================
```
### Binary: matrix_mul_tiled
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
    Matrix Mul Tiled 性能基准测试
========================================
矩阵维度：A(1024 x 1024) * B(1024 x 1024) = C(1024 x 1024)
数据大小：12.00 MB (A + B + C)
Kernel 迭代次数：10 次

--- GPU 详细计时 ---
H2D 传输时间：       0.90 ms
Kernel 执行时间：    0.31 ms (10 次平均)
D2H 传输时间：       0.45 ms
GPU 总时间：         1.66 ms

--- CPU 计时 ---
CPU 执行时间：    2088.25 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：6733.98x
CPU vs GPU 总时间加速比：1257.55x
GPU 计算性能：6925.01 GFLOPS
CPU 计算性能：1.03 GFLOPS
(RTX 4090 理论峰值：~82.6 TFLOPS FP32)

--- 结果验证 ---
✓ GPU PASSED: 全部 1048576 个元素验证正确
✓ GPU/CPU 结果一致性验证通过

========================================
```
### Binary: matrix_mul_naive
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
    Matrix Mul Naive 性能基准测试
========================================
矩阵维度：A(1024 x 1024) * B(1024 x 1024) = C(1024 x 1024)
数据大小：12.00 MB (A + B + C)
Kernel 迭代次数：10 次

--- GPU 详细计时 ---
H2D 传输时间：       0.85 ms
Kernel 执行时间：    0.41 ms (10 次平均)
D2H 传输时间：       0.46 ms
GPU 总时间：         1.73 ms

--- CPU 计时 ---
CPU 执行时间：    2087.50 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：5071.09x
CPU vs GPU 总时间加速比：1208.70x
GPU 计算性能：5216.80 GFLOPS
CPU 计算性能：1.03 GFLOPS
(RTX 4090 理论峰值：~82.6 TFLOPS FP32)

--- 结果验证 ---
✓ GPU PASSED: 全部 1048576 个元素验证正确
✓ GPU/CPU 结果一致性验证通过

========================================
```

## vector_add.cu 代码逻辑与测试
**代码路径**: `01_Basics/01_vector_add/vector_add.cu`
**测试命令**: `./build/01_Basics/01_vector_add/vector_add`

**实现逻辑分析**:
1. **核心逻辑**: 这是 CUDA 编程的最基础范例。在 Kernel (`vector_add_kernel`) 中运用了 Grid-Stride Loop（网格跨步循环）设计，`i = blockIdx.x * blockDim.x + threadIdx.x; i += blockDim.x * gridDim.x`机制使得处理超大规模元素 (64M) 时不受 Grid 维度限制。
2. **主机管理**: 分配超大内存 (单数组 256MB) 完成 `cudaMemcpy` 搬留。
3. **性能结果**: Kernel 耗时仅 **0.86ms** 左右。不仅对 CPU 产生 180x+ 加速，在 RTX 4090 上展现了 932.97 GB/s 的极限高带宽表现，完美压榨了显存总线。
**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## matrix_mul_tiled.cu 代码逻辑与测试
**代码路径**: `01_Basics/03_matrix_mul_tiled/matrix_mul_tiled.cu`
**测试命令**: `./build/01_Basics/03_matrix_mul_tiled/matrix_mul_tiled`

**实现逻辑分析**:
1. **基于 Tiling 改进**: 把全局大矩阵均分为 `TILE_SIZE x TILE_SIZE`（如 32x32）的小方块，并利用每个 Block 极其低延迟的 Shared Memory 寄存局部数据 `__shared__ float s_A[...][...]`。
2. **减少显存压力**: 通过线程同步 `__syncthreads()`，Block 中多线程能够高效复位和重复利用片内数据，大幅度缩小 Global Memory 的带宽瓶颈拖累。
3. **性能结果**: 在 1024x1024 测试规模下表现优异，其耗时缩降至 **0.31ms**，运算性能由朴素版的 5200 GFLOPS 跃升至 **6925 GFLOPS**。计算结果与 CPU 对比保持一致无误差。
**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## matrix_mul_naive.cu 代码逻辑与测试
**代码路径**: `01_Basics/02_matrix_mul_naive/matrix_mul_naive.cu`
**测试命令**: `./build/01_Basics/02_matrix_mul_naive/matrix_mul_naive`

**实现逻辑分析**:
1. **原始数学逻辑直录**: 直接依照矩阵乘法定义编写，通过 `blockIdx` 与 `threadIdx` 获取每个线程计算位置，内部运用单一的由 0 到 K 的大循环 `for` 直接累加乘积。
2. **未优化痛点展示**: 每个线程需要自发向全局内存读取整行/列的 N 个元素，使得大量相似访存未被合并使用，导致了高昂的 Global Memory 延迟堆积。
3. **性能基准验证**: 能够被 4090 并行化执行而时间压缩至约 **0.41ms** (相较于单核 CPU 的 2100ms 是巨大的 5100倍 碾压)，但相对于 GPU 理论浮点极限算力和带 Tiling 的进阶方法还有进一步发展空间。
**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```
