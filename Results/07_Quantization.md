# 07_Quantization 综合测试报告

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
./build/07_Quantization/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/07_Quantization/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 说明与约定

- **fp16_gemm / int8_gemm**：当矩阵规模较大（如 M=1024）时，代码中 CPU 参考计算被跳过（`if (M <= 512)` 才执行），故日志中「CPU 执行时间：0.00 ms」「CPU vs GPU 加速比：0.00x」表示未做 CPU 对比，并非 GPU 无加速。
- **quant_dequant**：Kernel 耗时极短（0.02–0.03 ms）、有效带宽超过 HBM 理论峰值（~1008 GB/s）是因为数据落在 L2 缓存，属正常现象。

## 五、 本地自动脚本基础运行记录

*(此下为真机二进制标准执行日志)*

## quant_dequant.cu 代码逻辑与测试

**代码路径**: `07_Quantization/03_quant_dequant/quant_dequant.cu`  
**测试命令**: `./build/07_Quantization/03_quant_dequant/quant_dequant`  
**Kernel**: `quantize_per_tensor`, `dequantize_per_tensor`, `quantize_per_channel`, `fp32_to_fp16`, `fp16_to_fp32`

实现：Per-Tensor / Per-Channel 量化与反量化、FP32↔FP16 类型转换；有效带宽超过 HBM 峰值因数据落在 L2 缓存。

**运行测试输出**:

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
      量化/反量化性能基准测试
========================================
数组大小：10485760 元素
数据大小：40.00 MB
Block 大小：256 线程
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：
 - Quantize Per-Tensor:      86.65 ms
 - Dequantize Per-Tensor:     8.34 ms
 - Quantize Per-Channel:     88.87 ms
 - FP32 to FP16:             95.77 ms
 - FP16 to FP32:             55.25 ms

--- GPU 版本 1: FP32 to INT8 Per-Tensor 量化 ---
H2D 传输时间：       4.00 ms
Kernel 执行时间：    0.02 ms (100 次平均)
D2H 传输时间：       1.03 ms
GPU 总时间：         5.05 ms

--- GPU 版本 2: INT8 to FP32 Per-Tensor 反量化 ---
H2D 传输时间：       1.04 ms
Kernel 执行时间：    0.02 ms (100 次平均)
D2H 传输时间：       3.98 ms
GPU 总时间：         5.04 ms

--- GPU 版本 3: FP32 to INT8 Per-Channel 量化 ---
H2D 传输时间：       3.98 ms
Kernel 执行时间：    0.03 ms (100 次平均)
D2H 传输时间：       1.02 ms
GPU 总时间：         5.03 ms

--- GPU 版本 4: FP32 to FP16 直接转换 ---
H2D 传输时间：       4.00 ms
Kernel 执行时间：    0.02 ms (100 次平均)
D2H 传输时间：       2.01 ms
GPU 总时间：         6.03 ms

--- GPU 版本 5: FP16 to FP32 直接转换 ---
H2D 传输时间：       2.02 ms
Kernel 执行时间：    0.02 ms (100 次平均)
D2H 传输时间：       3.99 ms
GPU 总时间：         6.03 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：
 - Quantize Per-Tensor:   3580.82x
 - Dequantize Per-Tensor: 388.25x
 - Quantize Per-Channel:  2985.81x
 - FP32 to FP16:          4432.87x
 - FP16 to FP32:          2567.49x

GPU 有效带宽：
 - Quantize Per-Tensor:   2166.62 GB/s
 - Dequantize Per-Tensor: 2440.42 GB/s
 - Quantize Per-Channel:  1762.77 GB/s
 - FP32 to FP16:          2911.98 GB/s
 - FP16 to FP32:          2923.45 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- 结果验证 ---
✓ Quantize Per-Tensor PASSED: 结果 -7.00 (期望 -7.00)
✓ Dequantize Per-Tensor PASSED: 结果 -1.40 (期望 -1.40)
✓ Quantize Per-Channel PASSED: 结果 -113.00 (期望 -113.00)
✓ FP32 to FP16 Cast PASSED: 结果 -0.34 (期望 -0.34)
✓ FP16 to FP32 Cast PASSED: 结果 -0.34 (期望 -0.34)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## fp16_gemm.cu 代码逻辑与测试

**代码路径**: `07_Quantization/01_fp16_gemm/fp16_gemm.cu`  
**测试命令**: `./build/07_Quantization/01_fp16_gemm/fp16_gemm`  
**Kernel**: `kernel_naive_fp16_gemm`, `kernel_tiled_fp16_gemm`, `kernel_vectorized_fp16_gemm`

实现：Naive / Tiled (Shared Memory) / Vectorized (`half2`) 三种 FP16 GEMM；Host 负责分配与拷贝，与 CPU 结果容差验证。大矩阵 (如 1024×1024) 时跳过 CPU 参考，故日志中「CPU 执行时间：0.00 ms」「加速比：0.00x」表示未做 CPU 对比。

**运行测试输出**:

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
      FP16 GEMM 性能基准测试
========================================
矩阵维度：A(1024 x 1024) * B(1024 x 1024) = C(1024 x 1024)
数据大小：6.00 MB
Block 大小：32 x 32 线程
Kernel 迭代次数：10 次

--- CPU 计时 ---
CPU 执行时间：       0.00 ms

--- GPU 版本 1: Naive FP16 GEMM ---
H2D 传输时间：       0.45 ms
Kernel 执行时间：    0.42 ms (10 次平均)
D2H 传输时间：       0.25 ms
GPU 总时间：         1.13 ms

--- GPU 版本 2: Tiled FP16 GEMM ---
H2D 传输时间：       0.45 ms
Kernel 执行时间：    0.33 ms (10 次平均)
D2H 传输时间：       0.25 ms
GPU 总时间：         1.03 ms

--- GPU 版本 3: Vectorized FP16 GEMM ---
H2D 传输时间：       0.40 ms
Kernel 执行时间：    0.22 ms (10 次平均)
D2H 传输时间：       0.24 ms
GPU 总时间：         0.86 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：0.00x
CPU vs GPU 总时间加速比：0.00x
GPU 计算性能：9697.25 GFLOPS
GPU 有效带宽：28.41 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Naive:         0.4239 ms (基准)
Tiled:         0.3314 ms (1.28x)
Vectorized:      0.22 ms (1.91x)

--- 结果验证 ---
  [Skip] Naive FP16 GEMM validation for large matrices.
  [Skip] Tiled FP16 GEMM validation for large matrices.
  [Skip] Vectorized FP16 GEMM validation for large matrices.
✓ GPU/CPU 结果一致性验证通过

========================================

```

## int8_gemm.cu 代码逻辑与测试

**代码路径**: `07_Quantization/02_int8_gemm/int8_gemm.cu`  
**测试命令**: `./build/07_Quantization/02_int8_gemm/int8_gemm`  
**Kernel**: `naive_int8_gemm`, `dp4a_int8_gemm`, `vectorized_int8_gemm`

实现：Naive / dp4a（4 个 INT8 打包乘加）/ Vectorized（每线程 4 列、`int4` 写回）三种 INT8 GEMM；大矩阵时同样跳过 CPU 参考，加速比 0.00x 表示未做 CPU 对比。

**运行测试输出**:

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
      INT8 GEMM 性能基准测试
========================================
矩阵维度：A(1024 x 1024) * B(1024 x 1024) = C(1024 x 1024)
数据大小：6.00 MB
Block 大小：32 x 32 线程
Kernel 迭代次数：10 次

--- CPU 计时 ---
CPU 执行时间：       0.00 ms

--- GPU 版本 1: Naive INT8 GEMM ---
H2D 传输时间：       0.30 ms
Kernel 执行时间：    0.41 ms (10 次平均)
D2H 传输时间：       0.45 ms
GPU 总时间：         1.16 ms

--- GPU 版本 2: dp4a INT8 GEMM ---
H2D 传输时间：       0.27 ms
Kernel 执行时间：    0.28 ms (10 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         0.99 ms

--- GPU 版本 3: Vectorized dp4a INT8 GEMM ---
H2D 传输时间：       0.23 ms
Kernel 执行时间：    0.19 ms (10 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         0.86 ms

--- 性能分析 ---
CPU vs GPU Kernel 加速比：0.00x
CPU vs GPU 总时间加速比：0.00x
GPU 有效带宽：33.14 GB/s
(RTX 4090 理论峰值：~1008 GB/s)
GPU 计算性能：11.31 TOPS

--- Kernel 性能对比 ---
Naive:              0.4070 ms (基准)
dp4a:               0.2758 ms (1.48x)
Vectorized dp4a:      0.19 ms (2.14x)

--- 结果验证 ---
  [Skip] Naive INT8 GEMM validation for large matrices.
  [Skip] dp4a INT8 GEMM validation for large matrices.
  [Skip] Vectorized dp4a INT8 GEMM validation for large matrices.
✓ GPU/CPU 结果一致性验证通过

========================================

```
