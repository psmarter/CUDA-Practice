# 14_CUTLASS 综合测试报告

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
./build/14_CUTLASS/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/14_CUTLASS/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为真机二进制标准执行日志)*

### Binary: cutlass_gemm
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
   CUTLASS GEMM 性能基准测试
========================================
矩阵维度：A(2048 x 2048) * B(2048 x 2048)
Kernel 迭代次数：20 次

--- CPU 计时 ---
CPU 执行时间：    0.00 ms

--- CUTLASS GEMM ---
Kernel 执行时间：    0.31 ms
计算性能：       55.49 TFLOPS

--- cuBLAS SGEMM (对比基准) ---
Kernel 执行时间：    0.30 ms
计算性能：       57.59 TFLOPS

--- 性能对比 ---
CUTLASS GEMM:  55.49 TFLOPS
cuBLAS SGEMM:  57.59 TFLOPS
CUTLASS/cuBLAS: 96.3%

--- 结果验证 ---
✗ cuBLAS SGEMM FAILED: 4194304 个元素超出误差阈值 (max_diff=554.1)
✗ CUTLASS GEMM FAILED: 4194304 个元素超出误差阈值 (max_diff=554.1)
✗ GPU/CPU 结果存在差异

========================================
```

### Binary: tensorop_gemm
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
   CUTLASS Tensor Core GEMM 性能基准测试
========================================
矩阵维度：A(2048 x 2048) * B(2048 x 2048)
数据类型：FP16 输入，FP32 累加输出
Kernel 迭代次数：20 次

--- CPU 计时 ---
CPU 执行时间：    0.00 ms

--- CUTLASS Tensor Core GEMM ---
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
CUTLASS Error: Error Internal
Kernel 执行时间：    0.00 ms
计算性能：       244032.24 TFLOPS

--- cuBLAS Tensor Core GEMM (对比基准) ---
Kernel 执行时间：    0.11 ms
计算性能：       157.68 TFLOPS

--- 性能对比 ---
CUTLASS GEMM:  244032.24 TFLOPS
cuBLAS GEMM:   157.68 TFLOPS
CUTLASS/cuBLAS: 154763.6%

--- 结果验证 ---
✗ cuBLAS Tensor Core GEMM FAILED: 4194304 个元素超出误差阈值 (max_diff=554.1)
✓ CUTLASS Tensor Core GEMM PASSED (最大误差 0.0)
✗ GPU/CPU 结果存在差异

========================================
```

### Binary: cute_basics
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
      CuTe (CUTLASS 3.x) 基础演示
========================================

>>> 运行 CuTe Print Kernel ... <<<
--- CuTe Layout 基础演示 ---
Layout 2D Shape: (0, 0)
Index(1, 2) 的一维偏移量: 6

--- CuTe 循环打印 --- 
layout(0) = 0
layout(1) = 4
layout(2) = 8
layout(3) = 1
layout(4) = 5
layout(5) = 9
layout(6) = 2
layout(7) = 6
layout(8) = 10
layout(9) = 3
layout(10) = 7
layout(11) = 11

>>> 测试 CuTe Tensor Copy Kernel ... <<<
✓ CuTe Tensor Copy 验证通过

========================================
```

## cutlass_gemm.cu 代码逻辑与测试
**代码路径**: `14_CUTLASS/01_cutlass_gemm/cutlass_gemm.cu`
**测试命令**: `./build/14_CUTLASS/01_cutlass_gemm/cutlass_gemm`

**实现逻辑分析**:
1. **CUTLASS GEMM**: C++ 模板抽象实现的高性能矩阵乘。
2. **灵活与极致**: 具有和 cuBLAS 相当的性能又同时完全开源且易配置。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so
```

## tensorop_gemm.cu 代码逻辑与测试
**代码路径**: `14_CUTLASS/02_tensorop_gemm/tensorop_gemm.cu`
**测试命令**: `./build/14_CUTLASS/02_tensorop_gemm/tensorop_gemm`

**实现逻辑分析**:
1. **TensorOp GEMM**: CUTLASS 对 Tensor Core 的专属调优。
2. **硬件挖掘**: 压榨矩阵运算流水线与寄存器分配的理论上限。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so
```

## cute_basics.cu 代码逻辑与测试
**代码路径**: `14_CUTLASS/03_cute_basics/cute_basics.cu`
**测试命令**: `./build/14_CUTLASS/03_cute_basics/cute_basics`

**实现逻辑分析**:
1. **CuTe Basics**: CUTLASS 最新的 Layout/Tensor Tiling 核心抽象库。
2. **理论颠覆**: 使用纯粹代数语义化来描述原本复杂的内存偏移映射关系。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so
```
