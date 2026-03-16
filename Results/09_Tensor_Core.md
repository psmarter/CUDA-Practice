# 09_Tensor_Core 综合测试报告

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
./build/09_Tensor_Core/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/09_Tensor_Core/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 说明与约定

- **wmma_gemm / mixed_precision**：当矩阵规模超过 512 时，代码跳过 CPU 参考计算，并将 `cpu_time_ms` 置为 1.0 以避免除零，故日志中的「CPU vs GPU Kernel 加速比」「CPU vs GPU 总时间加速比」为占位数值，**非真实 CPU/GPU 对比**。
- 实机有效算力、Kernel 时间、FP32 vs WMMA 加速比等以日志为准。

## 五、 本地自动脚本基础运行记录

*(此下为真机二进制标准执行日志)*

## wmma_gemm.cu 代码逻辑与测试

**代码路径**: `09_Tensor_Core/01_wmma_gemm/wmma_gemm.cu`  
**测试命令**: `./build/09_Tensor_Core/01_wmma_gemm/wmma_gemm`  
**Kernel**: `wmma_gemm_naive`

**实现逻辑分析**:

1. **硬件级矩阵指令映射**: 利用 `<mma.h>` 调用 Volta 及之后架构的 WMMA 能力，通过 `wmma::fragment` 将数据装入 Tensor Core 寄存器。
2. **执行粒度**: 核心在 `wmma::mma_sync`，每个 Warp（32 线程）协作完成 16×16×16 的 FP16 $D = A \times B + C$；大规模时按 Tile 切分、Grid-stride 喂满算力。
3. **测试结果**: 矩阵 2048 时跳过 CPU 参考；实机日志中 Kernel 时间约 **0.56 ms**，有效算力 **30.50 TFLOPS**（占位加速比见上文说明）。

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
      WMMA Tensor Core 性能基准测试
========================================
数组大小：M=2048 N=2048 K=2048
数据大小：32.00 MB
Block 大小：256 线程 (32x8 Warps)
Kernel 迭代次数：100 次

--- CPU 验证 (若尺寸较小) ---
矩阵尺寸过大，跳过 CPU 参考计算。

--- GPU 版本 1: Naive WMMA Tensor Core ---
H2D 传输时间：       1.65 ms
Kernel 执行时间：    0.56 ms (100 次平均)
D2H 传输时间：       1.65 ms
GPU 总时间：         3.86 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：1.78x
CPU vs GPU 总时间加速比：0.26x
GPU 有效算力 (TFLOPS)：30.50 TFLOPS
(RTX 4090 FP16 TC 理论算力峰值：~ 165 TFLOPS (无稀疏))
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Naive WMMA:   0.5633 ms (基准)

--- 结果验证 ---
✓ 结果验证跳过 (因矩阵尺寸过大，CPU 基准未计算)

========================================

```

## mixed_precision.cu 代码逻辑与测试

**代码路径**: `09_Tensor_Core/02_mixed_precision/mixed_precision.cu`  
**测试命令**: `./build/09_Tensor_Core/02_mixed_precision/mixed_precision`  
**Kernel**: `gemm_fp32_kernel`, `wmma_mixed_gemm_kernel`

**实现逻辑分析**:

1. **混合精度设计**: 输入 A、B 为 `half`，累加器 `wmma::accumulator<float>` 为 FP32，防止 FP16 累加溢出。
2. **基准对比**: 传统 FP32 GEMM（`gemm_fp32_kernel`）与 WMMA 混合精度（`wmma_mixed_gemm_kernel`）同规模 1024 对比；1024 时跳过 CPU 参考，占位加速比见上文说明。
3. **实测结果**: 实机日志中 FP32 Kernel 约 0.39 ms，WMMA Kernel 约 **0.05 ms**，有效算力 **39.36 TFLOPS**，相对 FP32 约 **7.21×**；结果验证在跳过 CPU 时仅输出跳过提示。

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
      WMMA 混合精度性能基准测试
========================================
矩阵尺寸：1024 x 1024 x 1024
数据大小：12.00 MB
WMMA 切块：16x16x16
Kernel 迭代次数：100 次

--- CPU 计时 (若尺寸较小) ---
矩阵尺寸过大，跳过 CPU 参考计算。

--- GPU 版本 1: 传统 FP32 GEMM ---
H2D 传输时间：       0.85 ms
Kernel 执行时间：    0.39 ms (100 次平均)
D2H 传输时间：       0.45 ms
GPU 总时间：         1.70 ms

--- GPU 版本 2: WMMA 混合精度 (FP16乘加FP32) ---
H2D(含转换)时间：    0.88 ms
Kernel 执行时间：    0.05 ms (100 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         1.37 ms

--- 性能分析 ---
CPU vs WMMA Kernel 加速比：18.33x
CPU vs WMMA 总时间加速比： 0.73x
FP32 有效算力：5.45 TFLOPS
WMMA 有效算力：39.36 TFLOPS
FP32 有效访存带宽：31.96 GB/s
WMMA 有效访存带宽：153.73 GB/s
(RTX 4090 理论峰值：~1008 GB/s)
(附：RTX 4090 Tensor Core 算力理论峰值 ~330 TFLOPS)

--- Kernel 性能对比 ---
Naive FP32 GEMM:         0.3937 ms (基准)
WMMA Mixed Precision:    0.0546 ms (7.21x)

--- 结果验证 ---
✓ 结果验证跳过 (因矩阵尺寸过大，CPU 基准未计算)

========================================

```
