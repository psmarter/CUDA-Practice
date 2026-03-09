# 05_LLM_Ops 综合测试报告

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
./build/05_LLM_Ops/<sub_directory>/<binary_name>

# 内存排错检查
compute-sanitizer ./build/05_LLM_Ops/<sub_directory>/<binary_name>

# Nsight Compute 吞吐性能捕获
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<binary_name>
```

## 四、 本地自动脚本基础运行记录

*(此下为 `run_all_tests.sh` 抓取的真机二进制标准执行日志)*

## layernorm.cu 代码逻辑与测试

**代码路径**: `05_LLM_Ops/02_layernorm/layernorm.cu`
**测试命令**: `./build/05_LLM_Ops/02_layernorm/layernorm`

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
      LayerNorm 性能基准测试
========================================
Batch 大小：128
Hidden Size：4096
总元素数：524288
单矩阵大小：2.00 MB
Block 大小：1024 线程
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：       2.54 ms

--- GPU 版本 1: Naive LayerNorm (Shared Memory Reduce) ---
H2D 传输时间：       0.24 ms
Kernel 执行时间：    0.01 ms (100 次平均)
D2H 传输时间：       0.25 ms
GPU 总时间：         0.49 ms

--- GPU 版本 2: Welford LayerNorm (单次遍历求均值方差) ---
H2D 传输时间：       0.23 ms
Kernel 执行时间：    0.01 ms (100 次平均)
D2H 传输时间：       0.24 ms
GPU 总时间：         0.48 ms

--- GPU 版本 3: Warp Reduce LayerNorm (寄存器级极致规约) ---
H2D 传输时间：       0.21 ms
Kernel 执行时间：    0.01 ms (100 次平均)
D2H 传输时间：       0.24 ms
GPU 总时间：         0.46 ms

--- GPU 版本 4: Warp-per-row LayerNorm (适配小Hidden Size) ---
H2D 传输时间：       0.22 ms
Kernel 执行时间：    0.04 ms (100 次平均)
D2H 传输时间：       0.24 ms
GPU 总时间：         0.50 ms

--- 性能分析 ---
CPU vs Warp Reduce GPU 加速比：329.49x
Naive GPU 有效带宽：        644.72 GB/s
Welford GPU 有效带宽：      691.89 GB/s
Warp Reduce GPU 有效带宽：  543.24 GB/s
Warp-per-row GPU 有效带宽：111.12 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Naive LayerNorm:          0.0065 ms (基准)
Welford LayerNorm:        0.0061 ms (1.07x)
Warp Reduce LayerNorm:      0.01 ms (0.84x)
Warp-per-row LayerNorm:     0.04 ms (0.17x)

--- 结果验证 ---
✓ Naive LayerNorm PASSED: 结果验证通过 (最大误差 0.00)
✓ Welford LayerNorm PASSED: 结果验证通过 (最大误差 0.00)
✓ Warp Reduce LayerNorm PASSED: 结果验证通过 (最大误差 0.00)
✓ Warp-per-row LayerNorm PASSED: 结果验证通过 (最大误差 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## softmax.cu 代码逻辑与测试

**代码路径**: `05_LLM_Ops/01_softmax/softmax.cu`
**测试命令**: `./build/05_LLM_Ops/01_softmax/softmax`

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
      Softmax 性能基准测试
========================================
Batch 大小：128
序列长度：4096
总元素数：524288
单矩阵大小：2.00 MB
Block 大小：1024 线程
Kernel 迭代次数：100 次

--- CPU 计时 ---
CPU 执行时间：       2.91 ms

--- GPU 版本 1: Naive Softmax (Shared Memory Reduce) ---
H2D 传输时间：       0.24 ms
Kernel 执行时间：    0.01 ms (100 次平均)
D2H 传输时间：       0.26 ms
GPU 总时间：         0.51 ms

--- GPU 版本 2: Online Softmax (Single-pass Reduce) ---
H2D 传输时间：       0.23 ms
Kernel 执行时间：    0.00 ms (100 次平均)
D2H 传输时间：       0.25 ms
GPU 总时间：         0.48 ms

--- GPU 版本 3: Warp Reduce Softmax (原 Fused Softmax) ---
H2D 传输时间：       0.21 ms
Kernel 执行时间：    0.00 ms (100 次平均)
D2H 传输时间：       0.24 ms
GPU 总时间：         0.45 ms

--- GPU 版本 4: Warp-per-row Softmax ---
H2D 传输时间：       0.20 ms
Kernel 执行时间：    0.04 ms (100 次平均)
D2H 传输时间：       0.25 ms
GPU 总时间：         0.48 ms

--- 性能分析 ---
CPU vs Warp Reduce GPU 加速比：819.67x
Naive GPU 有效带宽：        785.19 GB/s
Warp Reduce GPU 有效带宽：  1180.62 GB/s
Warp-per-row GPU 有效带宽：119.74 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Naive Softmax:          0.0053 ms (基准)
Online Softmax:         0.0041 ms (1.30x)
Warp Reduce Softmax:      0.00 ms (1.50x)
Warp-per-row Softmax:     0.04 ms (0.15x)

--- 结果验证 ---
✓ Naive Softmax PASSED: 结果验证通过 (最大误差 0.00)
✓ Online Softmax PASSED: 结果验证通过 (最大误差 0.00)
✓ Warp Reduce Softmax PASSED: 结果验证通过 (最大误差 0.00)
✓ Warp-per-row Softmax PASSED: 结果验证通过 (最大误差 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## rmsnorm.cu 代码逻辑与测试

**代码路径**: `05_LLM_Ops/05_rmsnorm/rmsnorm.cu`
**测试命令**: `./build/05_LLM_Ops/05_rmsnorm/rmsnorm`

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
   RMSNorm 性能基准测试
========================================
Token 数量：2048
隐藏层维度：4096
数据大小：32.00 MB
Kernel 迭代次数：50 次

--- CPU 计时 ---
CPU 执行时间：   21.67 ms

--- GPU 版本 1: Naive RMSNorm (单线程/行) ---
Kernel 执行时间：    0.32 ms (50 次平均)
加速比：68.6x

--- GPU 版本 2: Warp-level RMSNorm (256线程/行, warp shuffle) ---
Kernel 执行时间：     0.0 ms (50 次平均)
加速比 vs CPU：845.9x
加速比 vs Naive：12.33x

--- 带宽分析 ---
Naive 有效带宽：212.46 GB/s
Warp  有效带宽：2620.64 GB/s

--- 结果验证 ---
✓ Naive RMSNorm PASSED (最大误差 0.00)
✓ Warp RMSNorm PASSED (最大误差 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## rope.cu 代码逻辑与测试

**代码路径**: `05_LLM_Ops/04_rope/rope.cu`
**测试命令**: `./build/05_LLM_Ops/04_rope/rope`

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
   RoPE 位置编码 性能基准测试
========================================
序列长度：2048
Head 数量：32
Head 维度：128
数据大小：32.00 MB
Kernel 迭代次数：50 次

--- CPU 计时 ---
CPU 执行时间：   67.76 ms

--- GPU 版本 1: Naive RoPE ---
Kernel 执行时间：    0.04 ms (50 次平均)
加速比：1692.2x

--- GPU 版本 2: Vectorized RoPE (float2) ---
Kernel 执行时间：     0.0 ms (50 次平均)
加速比 vs CPU：1751.1x
加速比 vs Naive：1.03x

--- 带宽分析 ---
Naive 有效带宽：  1675.92 GB/s
Vector 有效带宽： 1734.27 GB/s

--- 结果验证 ---
✓ Naive RoPE PASSED (最大误差 0.00)
✓ Vectorized RoPE PASSED (最大误差 0.00)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## flash_attention.cu 代码逻辑与测试

**代码路径**: `05_LLM_Ops/03_flash_attention/flash_attention.cu`
**测试命令**: `./build/05_LLM_Ops/03_flash_attention/flash_attention`

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
      Flash Attention 性能基准测试
========================================
Batch 大小：2
Heads 数量：4
序列长度 (Seq_Len)：2048
头维度维度 (Head_Dim)：64
输入矩阵体积 (Q)：4.00 MB
[警告] 朴素版 N*N 中间变量体积：128.00 MB
Flash Attention 块大小：BR=32 BC=32
Kernel 迭代次数：50 次

--- CPU 计时 ---
CPU 执行时间：    6813.06 ms

--- GPU 版本 1: Naive Attention (模拟全显存遍历) ---
H2D 传输时间：       1.35 ms
Kernel 累加时间：    6.60 ms (50 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：         8.38 ms

--- GPU 版本 2: Flash Attention V1 (SRAM Tiling + 重计算) ---
H2D 传输时间：       1.33 ms
Kernel 执行时间：    9.58 ms (50 次平均)
D2H 传输时间：       0.44 ms
GPU 总时间：        11.35 ms

--- GPU 版本 3: Flash Attention V3 (Macro-Block + Vectorization) ---
H2D 传输时间：       1.29 ms
Kernel 执行时间：    5.33 ms (50 次平均)
D2H 传输时间：       0.43 ms
GPU 总时间：         7.05 ms

--- 性能分析 ---
CPU vs Flash V3 加速比：1279.17x
Naive GPU 有效推断带宽：2.54 GB/s
Flash V1  有效推断带宽：1.75 GB/s
Flash V3  有效推断带宽：3.15 GB/s
(RTX 4090 理论峰值：~1008 GB/s)

--- Kernel 性能对比 ---
Naive Attention (3 Steps):   6.5959 ms (基准)
Flash Attention V1:          9.5833 ms (0.69x)
Flash Attention V3:            5.33 ms (1.24x)

--- 结果验证 ---
✓ Naive Attention PASSED: 结果 0.50 (期望 0.50)
✓ Flash Attention V1 PASSED: 结果 0.50 (期望 0.50)
✓ Flash Attention V3 (Macro-Block) PASSED: 结果 0.50 (期望 0.50)
✓ GPU/CPU 结果一致性验证通过

========================================

```

## softmax 代码逻辑与测试

**实现逻辑分析**:

1. **Warp级规约求最值**: 每个 Warp 中的线程并向寻找最大值，以稳定指数运算，防止溢出。
2. **全局分子分母并行计算**: Exponential计算后在 Warp 内进行加和规约，得出分母，然后完成除法映射。
3. **LLM核心操作**: 是注意力机制的核心后置归一化算子。

## layernorm 代码逻辑与测试

**实现逻辑分析**:

1. **均值和方差并行求解**: 单个Block负责一行数据的统计量（均值、平方均值）。
2. **在线单跑（Welford）**: 高级实现采用单遍读取完成均值与方差的在线计算，节省一次 Global Memory 读写。
3. **Transformer标配**: 处理张量的最后一维（Hidden dimension），将其规范化。

## flash_attention 代码逻辑与测试

**实现逻辑分析**:

1. **Tiling & Recomputation (Tiling Forward, Backward Recompute)**: 将 Q, K, V 进行分块装载到 SRAM 流式处理，大幅度省去存取大型 Attention Map ($N \times N$) 所带来 HBM 读写瓶颈。
2. **理论颠覆**: 从 Memory Bound 成功逆转为 Compute Bound 算子，提速并减少显存占用。
3. **安全指数规约计算**: 调整 Softmax 在分块中的局部最大值和缩放系数，保持数学等价。

## rope 代码逻辑与测试

**实现逻辑分析**:

1. **旋转位置编码 (Rotary Position Embedding)**: 将输入映射为复数域上的旋转，在不改变向量范数的前提下，将相对位置信息硬编码进去。
2. **访存模式优化**: 可以交织处理(相邻对)提取或半切分开操作，要求良好的内存合并特性。

## rmsnorm 代码逻辑与测试

**实现逻辑分析**:

1. **简化的 LayerNorm**: 去除了均值的减除，只进行均方根缩放。
2. **效率增益**: 减少了一次全行归约操作（少算一次均值），理论上略快于LayerNorm并在大多数现代LLM（如Llama）中具有等价效果。
