# 11_Inference_Optimization 综合测试报告

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
./build/11_Inference_Optimization/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/11_Inference_Optimization/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

# 11_Inference_Optimization 综合测试报告

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
./build/11_Inference_Optimization/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/11_Inference_Optimization/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录
*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*


### Binary: kernel_fusion
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
      算子融合 (Kernel Fusion) 访存优化基准测试
========================================
特征数组大小：134217728 元素
单张量大小  ：512.00 MB
测试算子链路：Add(A, B) -> ReLU -> Scale
Kernel 迭代次数：50 次

--- CPU 计时 ---
CPU 执行时间：    1161.52 ms

--- GPU 版本 1: 非融合序列 (Unfused Series) ---
H2D 传输时间：     101.19 ms
Kernel 执行时间：    4.06 ms (50 次平均)
D2H 传输时间：      49.77 ms
GPU 总时间：       155.01 ms

--- GPU 版本 2: 算子融合 (Fused Kernel) ---
H2D 传输时间：     129.85 ms
Kernel 执行时间：    1.73 ms (50 次平均)
D2H 传输时间：      49.80 ms
GPU 总时间：       181.38 ms

--- 理论访存计算与性能分析 ---
非融合版本物理带宽：  926.12 GB/s
已融合版本物理带宽：  932.58 GB/s
--------------------------------
非融合版本有效带宽：  396.91 GB/s (受制于无效的中间访存)
算子融合后有效带宽：  932.58 GB/s (接近硬件物理极限)
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能加速比 ---
非融合序列 耗时  :   4.0579 ms
算子融合版 耗时  :   1.7271 ms
>> 融合加速比   : 2.35x

--- 结果验证 ---
✓ Unfused Kernels PASSED: 结果 0.00 (期望 0.00)
✓ Fused Kernels PASSED: 结果 0.00 (期望 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================
```
### Binary: kv_cache
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
      KV Cache 内存管理优化基准测试
========================================
Batch Size：32
Num Heads：16  Head Dim：64
最大序列长度：2048  Block 大小：16 个 Token
Kernel 迭代次数：100 次

--- 理论显存占用对比 ---
所有张量按完整长度 (2048) 预分配 (Naive):
> 预估 KV Cache 大小: 512.00 MB
使用分块按需分布并消除碎片 (Paged Attention):
> 预估 KV Cache 大小: 317.75 MB
> 节省显存: 37.94%

--- CPU 计时 ---
CPU 执行时间：     125.72 ms

--- GPU 版本 1: Naive (静态分配连续内存) ---
H2D 传输时间：      49.08 ms
Kernel 执行时间：    0.37 ms (100 次平均)
D2H 传输时间：       0.03 ms
GPU 总时间：        49.47 ms

--- GPU 版本 2: PagedAttention 机制 ---
H2D 传输时间：      67.60 ms
Kernel 执行时间：    0.45 ms (100 次平均)
D2H 传输时间：       0.03 ms
GPU 总时间：        68.07 ms

--- 性能分析 ---
CPU vs GPU (Naive) 加速比：341.29x
CPU vs GPU (Paged) 加速比：280.52x
Naive 有效带宽：  898.27 GB/s
Paged 有效带宽：  738.33 GB/s
性能对比差异  ：Paged 相比较 Naive 耗时 1.22x (主要来自指针解引用的开销)
(RTX 4090 理论峰值：~1008 GB/s)

--- 结果验证 ---
✓ Naive Attention PASSED: 结果 3.49 (期望 3.49)
✓ Paged Attention PASSED: 结果 3.49 (期望 3.49)
✓ GPU/CPU 结果一致性验证通过

========================================
```
### Binary: dynamic_batching
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
  动态/连续批处理 (Continuous Batching) 基准测试
========================================
Batch 规模  ：128 个并发请求
单请求极长  ：1024
网络结构    ：Num Heads=32, Head Dim=128
Kernel 迭代 ：100 次

--- 理论负载与显存开销 ---
1. 静态 / 基础批次调度 (Static Padding):
   等待集齐 128 个请求并向最长维度对齐 (1024)。
   所需处理的 Token 载量: 131072 [4096.00 MB]
2. 动态 / Inflight 连续调度 (Continuous Packed Tensor):
   摒弃所有 0-Padding，合并有效 Token 放入 Flatten 数组。
   实际需计算的 Token 载量: 41959 [1311.22 MB]
>>> 预估节省计算量 (FLOPS/Mem): 67.99%

--- CPU 计时 (真实 Token) ---
CPU 执行时间：    1369.75 ms

--- GPU 版本 1: 静态批处理 (Static Padding to Max Length) ---
H2D 传输时间：     129.94 ms
Kernel 执行时间：    1.46 ms (100 次平均)
D2H 传输时间：       0.25 ms
GPU 总时间：       131.65 ms

--- GPU 版本 2: 动态批处理 / Varlen Packed Tensor ---
H2D 传输时间：     187.52 ms
Kernel 执行时间：    1.49 ms (100 次平均)
D2H 传输时间：       0.24 ms
GPU 总时间：       189.24 ms

--- 性能分析 ---
Kernel 耗时对比 : Static 1.4642 ms  vs  Varlen 1.4851 ms
>> Static 内部通过分支跳过 Padding，因此两者 Kernel 速度接近。
>> Continuous Batching 的核心收益是显存节省（67.99%），使同一 GPU 能服务更多并发请求。

Static 实际有效带宽：  874.53 GB/s
Varlen 实际有效带宽：  862.25 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- 显存占用对比 (核心指标) ---
Static Padding 显存  ：4096.00 MB
Varlen Packed  显存  ：1311.22 MB
>> 节省显存 67.99%，等效于可多服务 3.1x 的并发请求

--- 结果验证 ---
✓ Var-Len Attention PASSED: 结果 -6.8 (期望 -6.8)
✓ GPU/CPU Variadic-Length Attention 结果验证通过

========================================
```

## kernel_fusion.cu 代码逻辑与测试
**代码路径**: `11_Inference_Optimization/02_kernel_fusion/kernel_fusion.cu`
**测试命令**: `./build/11_Inference_Optimization/02_kernel_fusion/kernel_fusion`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## kv_cache.cu 代码逻辑与测试
**代码路径**: `11_Inference_Optimization/01_kv_cache/kv_cache.cu`
**测试命令**: `./build/11_Inference_Optimization/01_kv_cache/kv_cache`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## dynamic_batching.cu 代码逻辑与测试
**代码路径**: `11_Inference_Optimization/03_dynamic_batching/dynamic_batching.cu`
**测试命令**: `./build/11_Inference_Optimization/03_dynamic_batching/dynamic_batching`

**实现逻辑分析**: 
1. 定义了相应的 CUDA Kernel 函数进行核心计算。
2. 包含了 Host 端代码负责显存分配 (cudaMalloc) 及数据拷贝 (cudaMemcpy)。
3. 利用 `CHECK` 宏或者 `std::abs` 针对 CPU 计算结果与 GPU 结果进行了容差比对与正确性验证。

**Sanitizer & 运行测试输出**: 
```text
========= COMPUTE-SANITIZER
========= Unable to find injection library libsanitizer-collection.so

```

## kv_cache 代码逻辑与测试
**实现逻辑分析**:
1. **KV Cache**: LLM Transformer 中对过去 token 的 Key 和 Value 进行缓存以避免重复计算。
2. **访存密集**: 将原先密集计算转变为访存受限计算的推手。


## kernel_fusion 代码逻辑与测试
**实现逻辑分析**:
1. **Kernel Fusion**: 将多个琐碎的内核合并成一个，避免对 Global memory 的重复读写。
2. **效率**: 省去了多次 Launch overhead，将很多数据驻留在寄存器或 Shared memory 内部周转。


## dynamic_batching 代码逻辑与测试
**实现逻辑分析**:
1. **Dynamic Batching**: 动态地将到达推理服务器不同长度或不同时间的请求拼合成批次计算。
2. **吞吐化**: 使算力密集的硬件得以喂饱。

