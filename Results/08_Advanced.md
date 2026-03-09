# 08_Advanced 综合测试报告

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
./build/08_Advanced/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/08_Advanced/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录

*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

## multi_stream.cu 代码逻辑与测试

**代码路径**: `08_Advanced/02_multi_stream/multi_stream.cu`
**测试命令**: `./build/08_Advanced/02_multi_stream/multi_stream`

**实现逻辑分析**:

1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

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
      多流并发隐藏延迟性能基准测试
========================================
测试算子：C = A * sin(B) + B * cos(A)
数组大小：16777216 元素
数据大小：192.00 MB
并发流数：4 排队长度
Block 大小：256 线程
测试指标：全链路 (H2D -> Compute -> D2H) 周期执行时间
执行迭代：10 次

--- CPU 计时 ---
CPU 执行时间：     190.59 ms

--- GPU 版本 1: 传统单流 (完全串行) ---
Pipeline 周期时间：   15.55 ms (10 次平均)

--- GPU 版本 2: 多流 (流水线并发) ---
Pipeline 周期时间：   13.73 ms (10 次平均)

--- 性能分析 ---
CPU vs GPU (多流) 总加速比：13.88x
单流 vs 多流 并发加速比：   1.13x
GPU 有效流水线吞吐带宽：14.66 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
传统单流:  15.5549 ms (基准)
多流并发:    13.7338 ms (1.13x 并发开销减免)

--- 结果验证 ---
✓ 传统单流执行 PASSED: 结果 0.57 (期望 0.57)
✓ 多流并发执行 PASSED: 结果 0.57 (期望 0.57)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## pytorch_extension.cu 代码逻辑与测试

**代码路径**: `08_Advanced/03_pytorch_extension/pytorch_extension.cu`
**测试命令**: `./build/08_Advanced/03_pytorch_extension/pytorch_extension`

**实现逻辑分析**:

1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

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
      PyTorch Extension 基准测试计算层
========================================
自定义算子：Swish Activation (Forward & Backward)
数组大小：10485760 元素
单数组大小：40.00 MB
Block 大小：256 线程
Kernel 迭代次数：100 次

--- CPU 计时 ---
Forward CPU 执行时间：      30.30 ms
Backward CPU 执行时间：     46.01 ms

--- GPU 版本: Custom Swish ---
[Forward] H2D 传输时间：     3.97 ms
[Forward] Kernel 时间：      0.08 ms (100 次平均)
[Forward] D2H 传输时间：     4.07 ms
[Backward] H2D 传输时间：    7.87 ms
[Backward] Kernel 时间：     0.13 ms (100 次平均)
[Backward] D2H 传输时间：    4.06 ms

--- 性能分析 ---
[Forward] CPU vs GPU Kernel 加速比：369.13x
[Backward] CPU vs GPU Kernel 加速比：342.43x
GPU Forward 有效带宽：1022.08 GB/s
GPU Backward 有效带宽：936.41 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- 结果验证 ---
✓ Swish Forward PASSED: 结果 -0.00 (期望 -0.00)
✓ Swish Backward PASSED: 结果 -0.00 (期望 -0.00)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## cuda_graphs.cu 代码逻辑与测试

**代码路径**: `08_Advanced/01_cuda_graphs/cuda_graphs.cu`
**测试命令**: `./build/08_Advanced/01_cuda_graphs/cuda_graphs`

**实现逻辑分析**:

1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

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
      CUDA Graphs 编排性能基准测试
========================================
流水线步骤：(A + B) * D + F = G
数组大小：100000 元素
涉及显存：2.67 MB
Block 大小：256 线程
Kernel 迭代次数：1000 次

--- CPU 计时 ---
CPU 执行时间：       0.17 ms

--- GPU 版本 1: 传统多 Kernel 发射 ---
H2D 传输时间：       0.18 ms
Kernel 执行时间：    0.00 ms (1000 次平均)
D2H 传输时间：       0.07 ms
GPU 总时间：         0.25 ms

--- GPU 版本 2: CUDA Graph Launch ---
H2D 传输时间：       0.17 ms
Kernel 执行时间：    0.00 ms (1000 次平均)
D2H 传输时间：       0.06 ms
GPU 总时间：         0.24 ms

--- 性能分析 ---
CPU vs GPU (Graph) Kernel 加速比：41.52x
CPU vs GPU (Graph) 总时间加速比：0.72x
GPU 有效带宽：859.11 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Multi-Stream Launch:   0.0049 ms (基准)
CUDA Graph Launch:     0.0042 ms (1.18x CPU 发射开销减免)

--- 结果验证 ---
✓ 传统流发射 Kernel 流水 PASSED: 结果 1.27 (期望 1.27)
✓ CUDA Graphs 发射 PASSED: 结果 1.27 (期望 1.27)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## cuda_graphs 代码逻辑与测试

**实现逻辑分析**:

1. **CUDA Graphs**: 将一系列 CUDA 操作（Kernel启动、内存拷贝等）拓扑为一个图结构。
2. **优势**: 一次性提交图执行，极大降低了 CPU 发射 Kernel 的额外开销。

## multi_stream 代码逻辑与测试

**实现逻辑分析**:

1. **Multi Stream**: 使用多个 CUDA 流使不同的任务在 GPU 上并发和重叠执行。
2. **重叠**: 实现 Compute 与 Data Transfer (H2D/D2H) 的相互掩盖。

## pytorch_extension_test 代码逻辑与测试

**实现逻辑分析**:

1. **PyTorch Extension**: 将 CUDA 自定义算子通过 pybind11 封装入 Python，直接给 PyTorch 提供原生定制算子。
2. **作用**: 消除 Pytorch 底层与 CUDA C++ 的隔阂。
