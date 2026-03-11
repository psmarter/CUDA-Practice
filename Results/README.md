# Results — 性能基准测试报告汇总

> 本目录收录了 CUDA-Practice 项目全部 15 个主题模块的**真机性能测试报告**，涵盖正确性验证、Kernel 耗时、有效带宽、计算吞吐 (GFLOPS/TFLOPS)、加速比等关键指标。

## 测试环境

| 项目 | 配置 |
| :--- | :--- |
| **GPU** | 2× NVIDIA GeForce RTX 4090 (Ada Lovelace, sm_89) |
| **显存** | 每卡 ~23.65 GB GDDR6X |
| **SM 数量** | 128 / 卡 |
| **理论带宽** | ~1008 GB/s |
| **FP32 理论峰值** | ~82.6 TFLOPS |
| **FP16 TC 理论峰值** | ~165 TFLOPS (无稀疏) |
| **编译器** | nvcc -O3, C++17 |
| **依赖库** | cuBLAS, cuFFT, CUTLASS, NCCL |

## 测试方法

每份报告均统一包含以下验证维度：

1. **正确性验证** — CPU 参考实现 vs GPU 结果逐元素容差比对
2. **Kernel 计时** — `cudaEventRecord` 多轮迭代取平均
3. **内存安全** — `compute-sanitizer` 检查越界/竞争
4. **性能剖析** — Nsight Compute (`ncu`) 采集 SM/DRAM 吞吐与寄存器利用率

## 报告目录

| # | 报告文件 | 主题 | 核心算子 |
| :---: | :--- | :--- | :--- |
| 01 | [01_Basics.md](01_Basics.md) | 基础并行化 | Vector Add、Naive GEMM、Tiled GEMM |
| 02 | [02_Reduction.md](02_Reduction.md) | 并行归约 | Simple / Convergent / Shared Mem / Coarsened Reduce |
| 03 | [03_Scan.md](03_Scan.md) | 前缀和 | Kogge-Stone、Brent-Kung、Segmented Scan |
| 04 | [04_GEMM_Optimization.md](04_GEMM_Optimization.md) | GEMM 优化全链路 | Tiled → Coarse → Register Tiling (vs cuBLAS) |
| 05 | [05_LLM_Ops.md](05_LLM_Ops.md) | 大模型算子 | Softmax、LayerNorm、Flash Attention V1/V3、RoPE、RMSNorm |
| 06 | [06_Warp_Primitives.md](06_Warp_Primitives.md) | Warp 原语 | Shuffle、Warp Reduce、Warp Scan |
| 07 | [07_Quantization.md](07_Quantization.md) | 量化推理 | FP16 GEMM、INT8 dp4a、量化/反量化 |
| 08 | [08_Advanced.md](08_Advanced.md) | 高级特性 | CUDA Graphs、Multi-Stream、PyTorch Extension |
| 09 | [09_Tensor_Core.md](09_Tensor_Core.md) | 张量核心 | WMMA GEMM、混合精度 |
| 10 | [10_Memory_Optimization.md](10_Memory_Optimization.md) | 访存优化 | 合并访存、Bank Conflict、异步拷贝 |
| 11 | [11_Inference_Optimization.md](11_Inference_Optimization.md) | 推理优化 | KV Cache + PagedAttention、Kernel Fusion、Dynamic Batching |
| 12 | [12_Standard_Libraries.md](12_Standard_Libraries.md) | 标准库 | cuBLAS GEMM、cuFFT、Thrust sort/reduce |
| 13 | [13_Performance_Analysis.md](13_Performance_Analysis.md) | 性能分析 | Occupancy 陷阱、Roofline 模型、Nsight 诊断 |
| 14 | [14_CUTLASS.md](14_CUTLASS.md) | CUTLASS 框架 | CUTLASS GEMM (96.3% cuBLAS)、CuTe 布局 |
| 15 | [15_Multi_GPU.md](15_Multi_GPU.md) | 多卡通信 | NCCL AllReduce |

## 核心性能总览

下表汇总各模块中最具代表性的 Kernel 实测结果：

| 算子 | 规模 | Kernel 耗时 | 关键指标 | vs CPU 加速比 |
| :--- | :--- | :--- | :--- | :--- |
| **Vector Add** | 64M 元素 (256 MB/数组) | 0.86 ms | 932.81 GB/s (峰值 93%) | 181× |
| **Tiled GEMM** | 1024² × 1024² | 0.31 ms | 6,893 GFLOPS | 6,696× |
| **Register Tiling GEMM** | 2048² × 2048² | 0.60 ms | 28.79 TFLOPS (cuBLAS 50%) | — |
| **Convergent Reduce** | 2048 元素 | 0.0038 ms | 1.36× vs Simple | — |
| **Segmented Scan** | 1M 元素 | 0.022 ms | 378.77 GB/s | 80.69× |
| **LayerNorm (Welford)** | Batch=128, H=4096 | 0.006 ms | 691.89 GB/s | 329× |
| **Flash Attention V3** | Seq=2048, d=64 | 5.33 ms | $O(N)$ 显存 | 1,279× |
| **Warp Reduce Sum** | 128 MB | 0.15 ms | 932.06 GB/s (峰值 92%) | 276× |
| **INT8 量化 (Per-Tensor)** | 10M 元素 | 0.02 ms | — | 3,580× |
| **CUDA Graphs** | 多 Kernel 流水 | 0.0042 ms | 发射开销 −14.3% | — |
| **PyTorch Swish Extension** | 10.5M 元素 | 0.08 ms | 1,022 GB/s | 369× |
| **WMMA Tensor Core** | 2048² × 2048² | 0.56 ms | 30.50 TFLOPS | — |
| **混合精度 (FP16→FP32)** | 1024² × 1024² | 0.06 ms | 37.73 TFLOPS | 7.2× vs FP32 |
| **Bank Conflict 消解** | 4096² 矩阵 | 0.15 ms | 879 GB/s (无冲突) | 1.19× vs 有冲突 |
| **Kernel Fusion** | 128M 元素 | 1.73 ms | 932.85 GB/s | 2.35× vs 非融合 |
| **KV Cache (PagedAttention)** | B=32, H=16, Seq=2048 | 0.45 ms | 显存 −37.94% | 280× |
| **cuBLAS SGEMM** | 1024² × 1024² | 0.04 ms | 49.91 TFLOPS | — |
| **Thrust Sort** | 10M 元素 | 1.30 ms | — | 1,634× |
| **cuFFT Forward** | 4096 点 | 0.0035 ms | — | 112,156× |
| **CUTLASS GEMM** | 2048² × 2048² | 0.31 ms | 55.35 TFLOPS (cuBLAS 96.3%) | — |
| **NCCL AllReduce** | 双卡同步 | 28.20 ms | 全局归约一致 | — |

## 关键性能洞察

### 带宽利用率

多数访存密集型 Kernel 达到了 RTX 4090 理论带宽的 **85%–95%**：

- Vector Add: **932.81 GB/s** (93%)
- Warp Reduce: **932.06 GB/s** (92%)
- Kernel Fusion: **932.85 GB/s** (93%)
- Block Reduce: **937.89 GB/s** (93%)

### 计算密集型算子

GEMM 系列优化展示了从 Naive 到工业级的完整演进：

```
Naive GEMM    →  Tiled GEMM    →  Register Tiling  →  CUTLASS     →  cuBLAS
5.2 TFLOPS       6.9 TFLOPS       28.79 TFLOPS       55.35 TFLOPS   57.49 TFLOPS
(6.3%)           (8.4%)           (34.9%)            (67.0%)        (69.6%)
```

### 反直觉发现

| 现象 | 结论 |
| :--- | :--- |
| 低 Occupancy + ILP 达到 1,365 GB/s 读写合计有效带宽 | **Occupancy 不是唯一关键**，指令级并行同样重要 |
| 合并访存 vs 跨步访存 = 925 vs 427 GB/s | **合并访存是最大杠杆**，约 2.17× 吞吐差距 |
| 异步拷贝未加速 (1.00×) | `cp.async` 在简单场景无收益，需配合流水线才有价值 |
| PagedAttention 比 Naive 稍慢 (1.22×) | 显存节省 38% 的同时需付出指针间接寻址开销 |

### GPU vs CPU 加速概况

- **核心算子 Kernel 级别**：通常 **100×–1,600×** 加速
- **端到端（含传输）**：受制于 PCIe 拷贝，降至 **2×–1,300×**
- **极端加速**：cuFFT 达 **112,156×** (算法复杂度优势 $O(N \log N)$ vs $O(N^2)$)

## 如何复现

```bash
# 编译全部项目
mkdir build && cd build
cmake .. && cmake --build . --parallel 8

# 运行单个测试（当前目录为 build）
./<章节>/<子目录>/<二进制名>

# 内存安全检查（当前目录为 build）
compute-sanitizer ./<章节>/<子目录>/<二进制名>

# Nsight Compute 性能采集（当前目录为 build）
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./<章节>/<子目录>/<二进制名>
```

> **注意**：实际性能数据会因 GPU 型号、驱动版本、系统负载等因素而有差异。本目录中的数据均基于 RTX 4090 双卡环境多轮迭代取平均获得。
