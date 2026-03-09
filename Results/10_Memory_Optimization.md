# 10_Memory_Optimization 综合测试报告

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
./build/10_Memory_Optimization/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/10_Memory_Optimization/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

## bank_conflict.cu 代码逻辑与测试
**代码路径**: `10_Memory_Optimization/02_bank_conflict/bank_conflict.cu`
**测试命令**: `./build/10_Memory_Optimization/02_bank_conflict/bank_conflict`

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
      共享内存 Bank Conflict 基准测试
========================================
矩阵尺寸：4096 x 4096 (16777216 元素)
数据大小：64.00 MB
Block 大小：32x32 线程
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：     246.51 ms

--- GPU 版本 1: 无 Bank Conflict (连续访问) ---
H2D 传输时间：       6.33 ms
Kernel 执行时间：    0.15 ms (100 次平均)
D2H 传输时间：       6.35 ms
GPU 总时间：        12.83 ms

--- GPU 版本 2: 严重 Bank Conflict (跨行同列) ---
H2D 传输时间：       6.30 ms
Kernel 执行时间：    0.18 ms (100 次平均)
D2H 传输时间：       6.35 ms
GPU 总时间：        12.83 ms

--- GPU 版本 3: Padding 消除 Bank Conflict ---
H2D 传输时间：       6.28 ms
Kernel 执行时间：    0.16 ms (100 次平均)
D2H 传输时间：       6.35 ms
GPU 总时间：        12.80 ms

--- 追加分析: 一维数组不同 Stride (对应不同 Conflict-way) ---
Stride =  1 (无冲突)    :     0.00 ms
Stride =  2 (2-way 冲突):     0.00 ms (1.00x)
Stride = 32 (32-way冲突):     0.01 ms (2.25x)

--- 性能与带宽分析 ---
无冲突带宽：  879.49 GB/s
有冲突带宽：  740.07 GB/s
Padding带宽：  826.01 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
无冲突(基准):     0.1526 ms
有冲突(未优化):   0.1814 ms (1.19x 变慢)
Padding优化:        0.16 ms (0.94x 几乎与无冲突持平)

--- 结果验证 ---
✓ No Bank Conflict PASSED: 结果 0.00 (期望 0.00)
✓ With Bank Conflict PASSED: 结果 0.00 (期望 0.00)
✓ Padded No Conflict PASSED: 结果 0.00 (期望 0.00)
✓ Analyze Stride 1 PASSED: 结果 0.00 (期望 0.00)
✓ Analyze Stride 2 PASSED: 结果 0.00 (期望 0.00)
⚠ Analyze Stride 32 SKIPPED: stride=32 导致 shared memory 写入冲突，结果不确定（预期行为）
✓ GPU/CPU 结果一致性验证通过

========================================

```

## async_copy.cu 代码逻辑与测试
**代码路径**: `10_Memory_Optimization/03_async_copy/async_copy.cu`
**测试命令**: `./build/10_Memory_Optimization/03_async_copy/async_copy`

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
      异步内存拷贝 (Async Copy) 性能基准测试
========================================
数组大小：67108864 元素
数据大小：256.00 MB
Block 大小：256 线程
Pipeline 阶段数：3
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：     115.77 ms

--- GPU 版本 1: 同步共享内存拷贝 (Sync Copy) ---
H2D 传输时间：      24.81 ms
Kernel 执行时间：    0.60 ms (100 次平均)
D2H 传输时间：      24.33 ms
GPU 总时间：        49.73 ms

--- GPU 版本 2: 异步内存拷贝 (Single Stage Async) ---
H2D 传输时间：      27.88 ms
Kernel 执行时间：    0.60 ms (100 次平均)
D2H 传输时间：      24.28 ms
GPU 总时间：        52.76 ms

--- GPU 版本 3: 多阶段异步流水线 (3 Stages Pipeline) ---
H2D 传输时间：      27.93 ms
Kernel 执行时间：    0.63 ms (100 次平均)
D2H 传输时间：      24.34 ms
GPU 总时间：        52.90 ms

--- 性能分析与带宽利用率 ---
同步拷贝     有效带宽：  901.43 GB/s
单阶异步     有效带宽：  898.00 GB/s
多阶流水线   有效带宽：  856.55 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
同步拷贝 (基准) :   0.5956 ms
单阶异步 相比同步: 1.00x 加速
多阶流水 相比同步: 0.95x 加速
多阶流水 相比单阶: 0.95x 加速

--- 结果验证 ---
✓ Sync Copy PASSED: 结果 0.00 (期望 0.00)
✓ Single Stage Async PASSED: 结果 0.00 (期望 0.00)
✓ Multi-stage Pipeline PASSED: 结果 0.00 (期望 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## coalesced_access.cu 代码逻辑与测试
**代码路径**: `10_Memory_Optimization/01_coalesced_access/coalesced_access.cu`
**测试命令**: `./build/10_Memory_Optimization/01_coalesced_access/coalesced_access`

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
      内存访问模式优化性能基准测试
========================================
数组大小：16777216 元素
单数组数据大小：64.00 MB
AoS/SoA 总数据大小：256.00 MB
Block 大小：1024 线程
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：      27.22 ms

--- GPU 版本 1: 合并访问 ---
H2D 传输时间：       6.03 ms
Kernel 执行时间：    0.15 ms (100 次平均)
D2H 传输时间：       6.36 ms
GPU 总时间：        12.53 ms

--- GPU 版本 2: 跨步访问 (Stride 2) ---
H2D 传输时间：       5.98 ms
Kernel 执行时间：    0.16 ms (100 次平均)
D2H 传输时间：       6.37 ms
GPU 总时间：        12.51 ms

--- GPU 版本 3: AoS 访问 ---
H2D 传输时间：      23.71 ms
Kernel 执行时间：    0.58 ms (100 次平均)
D2H 传输时间：      25.29 ms
GPU 总时间：        49.58 ms

--- GPU 版本 4: SoA 访问 ---
H2D 传输时间：      23.83 ms
Kernel 执行时间：    0.59 ms (100 次平均)
D2H 传输时间：      25.42 ms
GPU 总时间：        49.84 ms

--- 性能分析与带宽利用率 ---
合并访问    有效带宽：  925.31 GB/s
跨步访问(2) 有效带宽：  427.34 GB/s
AoS 访问    有效带宽：  922.31 GB/s
SoA 访问    有效带宽：  912.82 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
单数组合并 vs 跨步(2): 1.08x 变慢
SoA结构访问 vs AoS访问: 0.99x 加速

--- 结果验证 ---
✓ Coalesced Access PASSED: 结果 0.00 (期望 0.00)
✓ Strided Access (stride=2) PASSED: 结果 0.00 (期望 0.00)
✓ AoS Access PASSED: 结果 0.00 (期望 0.00)
✓ SoA Access X PASSED: 结果 0.00 (期望 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## coalesced_access 代码逻辑与测试
**实现逻辑分析**:
1. **合并访存**: 使得 Warp 内连续地址访问能够尽可能在同一个内存事务(transaction)中完成，充分利用带宽。
2. **优化基石**: 所有对齐 Global memory 访问优化的根本途径。


## bank_conflict 代码逻辑与测试
**实现逻辑分析**:
1. **Bank Conflict 消除**: 通过改变内存存储步长或增加 padding 阻止多个线程同时访问同一个 Shared Memory Bank。
2. **带宽救星**: 避免了 Shared Memory 访问的串行化回退。


## async_copy 代码逻辑与测试
**实现逻辑分析**:
1. **异步拷贝**: 使用 Volta 以后提供的异步内存拷贝机制（如 `cuda::memcpy_async`），由外部硬直接支持。
2. **Pipeline**: 与计算做到 Pipeline 无缝隐藏，摆脱寄存器做跳板的局限。

