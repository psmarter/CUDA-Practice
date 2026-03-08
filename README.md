# CUDA-Practice

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> 从零开始的 CUDA 编程学习与实践项目。从基础算子的实现到结合具体场景的高性能优化，提供完整的代码参考与性能记录。

## 📁 项目结构

| 目录 | 内容 | 难度 |
|------|------|------|
| `01_Basics` | 向量加法、朴素矩阵乘法、分块矩阵乘法 | ⭐ |
| `02_Reduction` | 基础归约、优化归约（收敛/共享内存/粗化）、点积 | ⭐⭐ |
| `03_Scan` | 前缀和、分段扫描 | ⭐⭐ |
| `04_GEMM_Optimization` | 分块 GEMM、向量化 GEMM、双缓冲 GEMM、**寄存器分块 GEMM** | ⭐⭐⭐ |
| `05_LLM_Ops` | Softmax、LayerNorm、Flash Attention (V1 + V3)、**RoPE**、**RMSNorm** | ⭐⭐⭐⭐ |
| `06_Warp_Primitives` | Warp Shuffle、Warp Reduce、Warp Scan | ⭐⭐ |
| `07_Quantization` | FP16 GEMM、INT8 GEMM、量化/反量化 | ⭐⭐⭐ |
| `08_Advanced` | CUDA Graphs、多流并发、PyTorch C++ Extension | ⭐⭐⭐ |
| `09_Tensor_Core` | WMMA GEMM、混合精度计算 | ⭐⭐⭐ |
| `10_Memory_Optimization` | 合并访存、Bank Conflict、异步拷贝 | ⭐⭐⭐ |
| `11_Inference_Optimization` | KV Cache + PagedAttention、Kernel Fusion、Dynamic Batching | ⭐⭐⭐⭐ |
| `12_Standard_Libraries` | cuBLAS GEMM、cuFFT、Thrust | ⭐⭐ |
| `13_Performance_Analysis` | Occupancy 分析、Roofline 模型、Nsight Profiling | ⭐⭐⭐ |
| `14_CUTLASS` | CUTLASS SIMT GEMM、**Tensor Core GEMM**、**CuTe 基础** | ⭐⭐⭐ |
| `15_Multi_GPU` | 多卡通信：**NCCL AllReduce** | ⭐⭐⭐⭐ |

## 🧠 技术路线 (Skill Tree)

本项目构建了四个由浅入深的 CUDA 进阶学习路线：

- **基础与访存调优 (L1)**：CUDA 执行模型、Memory Coalescing、Bank Conflict 消解、Double Buffering。
- **经典算子与并发 (L2)**：Warp Primitives、并行规约 (Reduction) 与前缀和 (Scan)。
- **大模型硬核算子 (L3)**：Flash Attention (SRAM Tiling + Online Softmax)、RMSNorm、RoPE。
- **微架构与系统级 (L4)**：Tensor Core (WMMA/MMA)、CUTLASS/CuTe、CUDA Graphs、KV Cache & PagedAttention、NCCL 全局通信。

## 📈 性能记录 (Performance Benchmarks)

各子目录中预留了理论极值与实测数据的追踪记录表，用于展示针对不同架构特性进行优化的实际加速效果。以下为部分典型优化案例的概览：

| 核心算子 | 优化突破口 | 核心目标与对比基准 |
| :--- | :--- | :--- |
| **GEMM (Register Tiling)** | Block/Thread 级划分 + 寄存器复用 | 减少全局/共享内存压力，对比基准：cuBLAS |
| **Flash Attention (V3)**   | 向量化（float4）宏块 + SRAM 分解 | 将显存复杂度降至 $O(N)$ 并提升长序列处理速度 |
| **INT8 GEMM (DP4A)**       | 底层 DP4A 四并发位运算操作 | 降低访存量及带宽要求，对比基准：FP32 原生计算 |

## 🔥 核心亮点

- **Flash Attention 完整实现**：720 行代码，涵盖 Naive 3-step → V1 (Tiling + Online Softmax) → V3 (Macro-Block + float4 向量化)
- **KV Cache + PagedAttention**：492 行，包含完整的 Block Table 映射逻辑和显存对比分析
- **GEMM 优化全链路**：Naive → Tiled → Vectorized → Double Buffer → **Register Tiling (cuBLAS 对比)** → Tensor Core
- **统一基准测试框架**：每个 Kernel 都包含 CPU 参考实现 → GPU 实现 → 正确性验证 → 性能分析（带宽/GFLOPS/加速比）。*(注：由于测试机器差异，各子项目的 README 中均预留了 Benchmark 性能对比占位符供运行后自行填写)*
- **LLM 标配算子**：Flash Attention、RoPE、RMSNorm、LayerNorm、Softmax
- **公共工具库**：`Common/include/` 包含 CUDA 错误检查、计时器、Warp 级原语、统一类型别名

## 🛠 编译与运行

### 环境要求

- CUDA Toolkit ≥ 11.0
- CMake ≥ 3.18
- C++17 编译器
- 推荐 GPU：具有较新架构（如 Ampere 等）的 NVIDIA 设备

### 编译

```bash
mkdir build && cd build
cmake ..
cmake --build . --parallel 8
```

### 运行

```bash
# 运行单个项目，例如：
./01_Basics/01_vector_add/vector_add
./05_LLM_Ops/03_flash_attention/flash_attention
```

## 🏗 技术栈

- **语言**：CUDA C/C++ (C++17)
- **构建系统**：CMake
- **GPU 库**：cuBLAS、cuFFT、Thrust、WMMA、CUTLASS
- **平台**：Linux / Windows (MSVC)

## 📊 代码统计

- **40+ 个 CUDA 源文件**，分布在 14 个主题目录
- **9,500+ 行** CUDA/C++ 代码
- 结构清晰且附有详尽中文注释。

## ✍️ 代码说明

本项目的核心实现主要通过手写编写（包含底层 Kernel、封装与 CPU 对照），而在结果验证、输出排版以及辅助验证工具代码上的生成借助了辅助工具以加速代码编写。

## 📚 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass)
- [Flash Attention 论文 (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)

## 📄 License

本项目采用 [MIT License](LICENSE) 开源。
