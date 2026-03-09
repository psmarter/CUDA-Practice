# 13_Performance_Analysis 综合测试报告

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
./build/13_Performance_Analysis/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/13_Performance_Analysis/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

# 13_Performance_Analysis 综合测试报告

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
./build/13_Performance_Analysis/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/13_Performance_Analysis/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*


### Binary: nsight_profiling
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
      Nsight Profiling 基准测试与诱捕目标
========================================
数组大小：10000000 元素
数据大小：38.15 MB
Block 大小：256 线程
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：      19.40 ms

--- GPU 版本 1: Bad Kernel (非合并访存 Stride=32) ---
H2D 传输时间：       3.83 ms
Kernel 执行时间：    0.29 ms (100 次平均)
D2H 传输时间：       3.76 ms
GPU 总时间：         7.88 ms

--- GPU 版本 2: Good Kernel (规范合并访存) ---
H2D 传输时间：       3.76 ms
Kernel 执行时间：    0.07 ms (100 次平均)
D2H 传输时间：       3.80 ms
GPU 总时间：         7.63 ms

--- 性能分析 ---
合并访存 vs 非合并访存 加速比：4.46x

Bad  Kernel 有效带宽：273.36 GB/s
Good Kernel 有效带宽：1220.36 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- 结果验证 ---
✓ Bad Kernel	 PASSED: 结果 8.75 (期望 8.75)
✓ Good Kernel	 PASSED: 结果 8.75 (期望 8.75)
✓ GPU/CPU 结果一致性验证通过

========================================
完整性能分析步骤：

Step 1: 已完成 - 基准数据在上方

Step 2: Nsight Systems (系统级 Timeline)
  >> nsys profile --trace=cuda,nvtx -o nsight_timeline ./nsight_profiling
  >> nsys stats nsight_timeline.nsys-rep
  >> nsys-ui nsight_timeline.nsys-rep    # 需要显示器

Step 3: Nsight Compute (Kernel 深度分析)
  (1) 安装: sudo apt install nsight-compute
  (2) 链接: sudo ln -sf /opt/nvidia/nsight-compute/*/ncu /usr/local/bin/ncu
            sudo ln -sf /opt/nvidia/nsight-compute/*/ncu-ui /usr/local/bin/ncu-ui
  (3) 生成: sudo ncu --set full -o nsight_metrics ./nsight_profiling
  (4) 查看: ncu-ui nsight_metrics.ncu-rep       # 需要显示器
  或者纯命令行 (无需 GUI):
  >> sudo ncu --kernel-name profile_example --launch-skip 2 --launch-count 2 ./nsight_profiling
========================================
```
### Binary: occupancy
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
      Occupancy 分析基准测试
========================================
数组大小：10000000 元素
数据大小：38.15 MB
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：      16.29 ms

--- GPU 结构 1: 追求满 Occupancy (<256, 1>) ---
  >>> [High-Occ, Low-ILP] 理论信息 <<<
  Block配置        : < 256 线程, 1 数据/线程 >
  活跃 Block 数量  : 6 / SM
  活跃 Thread 数量 : 1536 / SM (最大 1536)
  理论 Occupancy   : 100.00 %
Kernel 执行时间：    0.07 ms

--- GPU 结构 2: 中等 Occupancy + ILP (<256, 4>) ---
  >>> [Mid-Occ, High-ILP] 理论信息 <<<
  Block配置        : < 256 线程, 4 数据/线程 >
  活跃 Block 数量  : 6 / SM
  活跃 Thread 数量 : 1536 / SM (最大 1536)
  理论 Occupancy   : 100.00 %
Kernel 执行时间：    0.06 ms

--- GPU 结构 3: 低 Occupancy + 终极 ILP (<64, 16>) ---
  >>> [Low-Occ, Max-ILP] 理论信息 <<<
  Block配置        : < 64 线程, 16 数据/线程 >
  活跃 Block 数量  : 24 / SM
  活跃 Thread 数量 : 1536 / SM (最大 1536)
  理论 Occupancy   : 100.00 %
Kernel 执行时间：    0.06 ms

--- GPU 结构 4: Shared Memory 挤占测试 (<256, 1> + 32KB Shared) ---
  >>> [32KB Shared Occ] 理论信息 <<<
  Block配置        : < 256 线程, 1 数据/线程 >
  活跃 Block 数量  : 3 / SM
  活跃 Thread 数量 : 768 / SM (最大 1536)
  理论 Occupancy   : 50.00 %
Kernel 执行时间：    0.08 ms

--- GPU 结构 5: __launch_bounds__ (<256, 1>) ---
  >>> [Launch Bounds] 理论信息 <<<
  Block配置        : < 256 线程, 1 数据/线程 >
  活跃 Block 数量  : 6 / SM
  活跃 Thread 数量 : 1536 / SM (最大 1536)
  理论 Occupancy   : 100.00 %
Kernel 执行时间：    0.07 ms

--- 性能总结 (读 + 写带宽) ---
配置 1 (满 Occupancy)   有效带宽： 1224.34 GB/s
配置 2 (ILP 均衡)       有效带宽： 1323.70 GB/s
配置 3 (低 Occupancy)   有效带宽： 1363.44 GB/s
配置 4 (被挤占的 Occ)   有效带宽： 1020.66 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

核心结论：Occupancy (占用率) 本质上是为了隐藏延迟。
当寄存器足够、通过使用 ILP (指令级并行) 也能极好地隐藏延迟时，即使 Occupancy 仅仅只有 25% 甚至 12%，其实际物理带宽吞吐依然能够逼近硬件极限，甚至超越高内存开销强制满 Occupancy 的情况。

--- 结果验证 ---
✓ 高 Occupancy	 PASSED: 结果 3.00 (期望 3.00)
✓ 中等 Occupancy	 PASSED: 结果 3.00 (期望 3.00)
✓ 低 Occupancy	 PASSED: 结果 3.00 (期望 3.00)
✓ Shared限制	 PASSED: 结果 3.00 (期望 3.00)
✓ Bounds限制	 PASSED: 结果 3.00 (期望 3.00)
✓ GPU/CPU 结果一致性验证通过

========================================
```
### Binary: roofline
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
    Roofline Model 硬件平台画像
========================================
单精度理论峰值算力 : 86.02 TFLOPS
理论峰值物理显存带宽: 1008.10 GB/s
拐点算术强度 (Ridge): 85.33 FLOPS/Byte
========================================

--- 测试项 1：Memory Bound Kernel (Vector Add N=10M) ---
CPU 执行时间：      23.39 ms
  [内存受限 Kernel (Vector Add)] Roofline 分析结果：
  算术强度 (I) : 0.083 FLOPS/Byte
  瓶颈受限     : Memory Bound (内存带宽受限)
  理论峰值速度 : 84.01 GFLOPS
  实际运行速度 : 78.67 GFLOPS
  计算侧效率   : 93.65 %

Kernel 执行时间：    0.13 ms (100 次平均)
✓ Memory Bound (VecAdd) PASSED: 结果 3.00 (期望 3.00)

--- 测试项 2：Compute Bound Kernel (GEMM N=1024) ---
  [计算受限 Kernel (Naive GEMM)] Roofline 分析结果：
  算术强度 (I) : 170.667 FLOPS/Byte
  瓶颈受限     : Compute Bound (计算核心受限)
  理论峰值速度 : 86016.00 GFLOPS
  实际运行速度 : 5235.03 GFLOPS
  计算侧效率   : 6.09 %

Kernel 执行时间：    0.41 ms (10 次平均)
✓ Compute Bound (GEMM) PASSED: 结果 2048.00 (期望 2048.00)
✓ GPU/CPU 结果一致性验证通过

========================================
```

## nsight_profiling.cu 代码逻辑与测试
**代码路径**: `13_Performance_Analysis/03_nsight_profiling/nsight_profiling.cu`
**测试命令**: `./build/13_Performance_Analysis/03_nsight_profiling/nsight_profiling`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## occupancy.cu 代码逻辑与测试
**代码路径**: `13_Performance_Analysis/01_occupancy/occupancy.cu`
**测试命令**: `./build/13_Performance_Analysis/01_occupancy/occupancy`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## roofline.cu 代码逻辑与测试
**代码路径**: `13_Performance_Analysis/02_roofline/roofline.cu`
**测试命令**: `./build/13_Performance_Analysis/02_roofline/roofline`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## occupancy 代码逻辑与测试
**实现逻辑分析**:
1. **占用率分析**: 研究 SM 上驻留的 Active Warps 比例。
2. **制约**: 寻找受限于寄存器还是 Shared Memory 的瓶颈。


## roofline 代码逻辑与测试
**实现逻辑分析**:
1. **Roofline 模型**: 评估当前算是 Calculate Bound 还是 Memory Bound。
2. **调优方向**: 给出目前离硬件带宽和算力天花板的相对位置。


## nsight_profiling 代码逻辑与测试
**实现逻辑分析**:
1. **Nsight Profiling**: 使用 `ncu` 或 `nsys` 具体捕捉执行时的事件流。
2. **细节捕捉**: 查看 Stall 因素如 MIO, L1 TEX, 等等。

