# 15_Multi_GPU 综合测试报告

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
./build/15_Multi_GPU/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/15_Multi_GPU/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录

*(此下为真机二进制标准执行日志)*

## nccl_allreduce.cu 代码逻辑与测试

**代码路径**: `15_Multi_GPU/01_nccl_allreduce/nccl_allreduce.cu`
**测试命令**: `./build/15_Multi_GPU/01_nccl_allreduce/nccl_allreduce`

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
      NCCL AllReduce 多卡规约演示
========================================

检测到 2 张 GPU，正在初始化 NCCL 环境...
NCCL 初始化成功，开始执行 AllReduce ...
AllReduce 跨设备执行耗时: 28.20 ms
--- 结果验证 ---
✓ 全局 AllReduce 同步验证通过。所有设备归约到的结果都是 1.00

========================================

```

## nccl_allreduce 代码逻辑与测试

**实现逻辑分析**:

1. **NCCL AllReduce**: 分布式多卡间进行节点求和同步的霸主。
2. **Ring/Tree 算法**: 进行全量参数梯度同步的核心底层接口支持，完全绕过 CPU。
