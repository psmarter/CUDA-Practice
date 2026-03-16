# CUDA-Practice

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

从零开始的 CUDA 编程学习与实践仓库：从基础算子实现到工业级优化（GEMM、Flash Attention、KV Cache 等），提供可运行代码与实机性能记录。配套技术博客见 **[Smarter's Blog — https://smarter.xin/](https://smarter.xin/)**，系列文章与仓库章节一一对应，便于系统学习。

---

## 项目结构

| 目录 | 内容 | 难度 |
|------|------|------|
| `01_Basics` | 向量加法、朴素/分块矩阵乘法 | ⭐ |
| `02_Reduction` | 归约（朴素/收敛/共享内存/粗化）、点积 | ⭐⭐ |
| `03_Scan` | 前缀和、分段扫描 | ⭐⭐ |
| `04_GEMM_Optimization` | 分块 / 向量化 / 双缓冲 / **寄存器分块** GEMM | ⭐⭐⭐ |
| `05_LLM_Ops` | Softmax、LayerNorm、**Flash Attention**、RoPE、RMSNorm | ⭐⭐⭐⭐ |
| `06_Warp_Primitives` | Warp Shuffle、Reduce、Scan | ⭐⭐ |
| `07_Quantization` | FP16/INT8 GEMM、量化与反量化 | ⭐⭐⭐ |
| `08_Advanced` | CUDA Graphs、多流、PyTorch C++ 扩展 | ⭐⭐⭐ |
| `09_Tensor_Core` | WMMA GEMM、混合精度 | ⭐⭐⭐ |
| `10_Memory_Optimization` | 合并访存、Bank Conflict、异步拷贝 | ⭐⭐⭐ |
| `11_Inference_Optimization` | KV Cache + PagedAttention、Kernel 融合、动态批处理 | ⭐⭐⭐⭐ |
| `12_Standard_Libraries` | cuBLAS、cuFFT、Thrust | ⭐⭐ |
| `13_Performance_Analysis` | 占用率、Roofline、Nsight 剖析 | ⭐⭐⭐ |
| `14_CUTLASS` | CUTLASS GEMM、Tensor Core、CuTe 基础 | ⭐⭐⭐ |
| `15_Multi_GPU` | NCCL AllReduce 多卡通信 | ⭐⭐⭐⭐ |

- **代码**：各章子目录内为独立可编译的 CUDA 工程（见各章 `README.md`）。
- **博客**：`Blogs/` 下为与章节对应的深度文章（数学推导、硬件原理、实现与调优）；在线版与更多笔记见 [https://smarter.xin/](https://smarter.xin/)。
- **性能结果**：`Results/` 下为真机基准测试报告（环境：2× RTX 4090，详见 `Results/README.md`）。

---

## 技术路线概览

- **L1 基础与访存**：执行模型、合并访存、Bank Conflict、双缓冲。
- **L2 经典算子**：Warp 原语、归约、前缀和。
- **L3 大模型算子**：Flash Attention、RMSNorm、RoPE 等。
- **L4 系统与微架构**：Tensor Core、CUTLASS/CuTe、CUDA Graphs、KV Cache、NCCL。

---

## 编译与运行

### 环境要求

- CUDA Toolkit ≥ 11.0  
- CMake ≥ 3.18  
- C++17  
- 推荐：NVIDIA GPU（如 Ampere 及以上）  
- `14_CUTLASS` 需设置 `CUTLASS_DIR`；`15_Multi_GPU` 需 NCCL 与多卡，Windows/无 NCCL 时可能无法全量编译。

### 编译

```bash
mkdir build && cd build
cmake ..
cmake --build . --parallel 8
```

### 运行

**请在 `build` 目录下执行**（即先 `cd build` 再运行）：

```bash
# 示例：运行基础向量加与 Flash Attention
./01_Basics/01_vector_add/vector_add
./05_LLM_Ops/03_flash_attention/flash_attention
```

更多用例与子目录说明见各章 `README.md`；复现与性能采集方法见 `Results/README.md`。

---

## 技术栈与统计

- **语言 / 构建**：CUDA C++（C++17）、CMake  
- **依赖**：cuBLAS、cuFFT、Thrust、WMMA、CUTLASS、NCCL（多卡）  
- **平台**：Linux / Windows (MSVC)  
- **规模**：40+ 个 `.cu` 文件，约 9,500+ 行，带中文注释

---

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)

---

## License

[MIT License](LICENSE)
