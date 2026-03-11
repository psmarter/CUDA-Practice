---
title: "07_Quantization：FP16 带宽翻倍、INT8 dp4a 指令与混合精度工程学"
date: 2026-03-12 14:00:00
tags: [CUDA, 高性能计算, 量化, FP16, INT8, dp4a, 混合精度, Vectorized, GEMM]
categories: 深度学习系统架构
---

## 本文目标

读完本文，你将能够：

- 解释量化技术缓解 Memory Bound 算子访存极限的物理机制
- 定量分析 Per-Tensor 与 Per-Channel 量化过程的代价损耗及其 L2 Cache 效应
- 运用 `half2` 复合类型及硬件指令 `__hfma2` 实现双路浮点并发
- 理解 `__dp4a` 指令的打包语意，并通过向量化重组打满 GPU 的 INT8 吞吐下限

## 对应代码路径

> **硬件环境**：NVIDIA RTX 4090 (Ada Lovelace, sm_89)
> 128 SMs | FP32 82.6 TFLOPS | HBM 1008 GB/s | L2 72 MB | Roofline 拐点 81.9 FLOP/Byte

| 源文件 | Kernel 名称 | 核心技术 | 测试规模 |
|--------|-------------|----------|----------|
| `07_Quantization/03_quant_dequant/quant_dequant.cu` | `quantize_per_tensor`<br>`quantize_per_channel`<br>`fp32_to_fp16_cast` | Absmax 量化缩放与类型硬转 | `N=10M` |
| `07_Quantization/01_fp16_gemm/fp16_gemm.cu` | `gemm_fp16_naive`<br>`gemm_fp16_tiled`<br>`gemm_fp16_vectorized` | `half2` 与 `__hfma2` 双路突袭 | `1024x1024` |
| `07_Quantization/02_int8_gemm/int8_gemm.cu` | `gemm_int8_dp4a`<br>`gemm_int8_vectorized_dp4a` | 4位 INT8 并置打包 / `__dp4a` | `1024x1024` |

> Kernel 名称与源码中 `__global__` 函数签名完全一致。

## Baseline

**问题陈述**：大模型执行阶段，极大多数计算（如 LayerNorm，Softmax_）被卡死在显存带宽上限（Memory Wall），而不是计算单元上。FP32 占据 4 字节，通过量化将其物理宽度强制斩半或截为四分之一，能够立竿见影地提升受制于 I/O 吞吐的总计算帧率。

| Baseline 类别 | 测试场景 | 指标 | 值 | 数据来源 |
|---------------|----------|------|----|----------|
| CPU FP32 转 FP16 | `N=10M` (40.00 MB) | 单核均时 | 95.77 ms | [实测] Results/07_Quantization.md |
| GPU NAIVE FP16 GEMM| `1024x1024` | 基础耗时 | 0.42 ms | [实测] Results/07_Quantization.md |
| GPU NAIVE INT8 GEMM| `1024x1024` | 基础耗时 | 0.41 ms | [实测] Results/07_Quantization.md |

## 瓶颈分析

试图纯粹通过改变数据类型提速，遭遇到的微观阻力如下：

1. **类型转置开销陷阱 (Quant/Dequant Overhead)**
   - Per-Tensor 量化（单标度全局锁定）由于共享内存寄存器命中极高，自身并无阻力。但在应对带强 Outlier 的层时必须切换至 Per-Channel（逐行标度）。此时硬件被迫拉取外部规模矩阵以匹配各行 `Scale`。繁杂的横越寻址导致访存被中断切割，显著降低有效利用率带宽（相比于前者直降 400 GB/s 以上）。
2. **缺乏向量化指令背书的 FP16 退化**
   - 简单的强制把参数变成 `half` 送进计算单元，若核心依旧按 32 位流水的拆解法则用软算进行拆卸装载，只会徒增 `__half2float` 的来回解释冗余，反而完全不能打满张量算术的周期极限（Naive FP16 GEMM 即是反面反例）。
3. **未被对齐打包的 INT8 点积（`dp4a` 堵塞）**
   - NVIDIA 硬件中一条 `__dp4a` 指令要求直接送入已被 32 字节打包压合好的 `4x INT8` 连续载荷。如果不按硬件内存对齐规律，任由软件强行用移位逻辑去动态拼装字节（Shift-mask），这几大步多余的操作指令反而会抵消掉该单周期汇编引擎本应拉出的优势。

## 优化思路

### 优化 1：利用 L2 Cache 进行极限转换 (Quantization Micro-kernel)

**解决的瓶颈**：显存高位宽被冗杂操作霸占引发降速。
**核心思想**：所有的转型铸造必须遵循底层 SIMT 原则，切除非必要的逻辑跳步。在 $N=10M$ 体量下依赖 4090 的极限 L2 容积结构（72MB）让其驻留阵列内存端口完成闪换。
**预期收益**：FP32 转 FP16/INT8 (Per-Tensor) 彻底杀进 **0.02 ms** 的微秒级底层，跑出逆天破物理限速的近 3000 GB/s 有效带宽。

### 优化 2：强制复合半字双发 (Vectorized FP16)

**解决的瓶颈**：FP16 未被彻底榨取双指令发射（Dual Instruction Issue）。
**核心思想**：彻底舍弃单个 `half` 流，通过 C++ `reinterpret_cast` 指针重塑拦截，一举拉走邻近两人组成专有类型 `half2`。利用汇编层级内建函数 `__hfma2` 在内部用一个周期实现对向量对 $(a, b)$ 并联独立的 $2 \times Fused Multiply-Add$ 指令发射。
**预期收益**：相较原生 FP16 版本提速接近 **1.91x**。

### 优化 3：预先组合与矩阵平铺解构 (`dp4a` Vectorized)

**解决的瓶颈**：INT8 单值加载拼连导致高频次寄存器换页（Register Spilling）以及打包指令周期税。
**核心思想**：将对 B 矩阵提取扩大到 4 行齐下，依靠 32 位的宽口令单次抽取 4 字节的数据组，利用寄存器端的位运算把散落在不同行的同一列数值压成单列 32-bit Pack，然后极简送进 `dp4a` 处理端口。
**预期收益**：彻底打满 11+ TOPS 的基础理论压迫级算力下限。

## 关键代码解释

### 向量化 FP16 的双发魔法

```cpp
// 来源：07_Quantization/01_fp16_gemm/fp16_gemm.cu : 局部片选简写
    half2 sum2 = __float2half2_rn(0.0f); // 预置累加基极
    
    for (int i = 0; i < N; ++i) {
        // [1] 取 A 的单个元素化为两具等额复制的分身：[A_val, A_val]
        half2 a_val2 = __halves2half2(A[row * N + i], A[row * N + i]);
        
        // [2] 非常关键：把 B 这里按照 16-bit 为步长的两数，靠内存造型看作 32-bit 封存一次拿干
        half2 b_val2 = *reinterpret_cast<const half2*>(&B[i * K + col]); 
        
        // [3] 调用底层一条原生汇编，上下对穿同时启动两管 FP16 加法
        sum2 = __hfma2(a_val2, b_val2, sum2); 
    }
```

**要点解读**：

- `[1]-[3]` 此处并非简单的并联语句。`__hfma2` 是物理存在于 NVIDIA SM 板块之上的硬质连排双发射单元入口，直接跳过了两重译码，节约出了将近一半的运算管线生命周期。

### INT8 dp4a 拼包艺术

```cpp
// 来源：07_Quantization/02_int8_gemm/int8_gemm.cu
    // [1] 从主内存一次提取出 32 位实体（包含连续四个 8 位子元素）
    int32_t b_row0_pack = *reinterpret_cast<const int32_t*>(&B[(i+0) * K + col]);
    // ...同样地提取 b_row1,2,3 ...

    // [2] 使用宏截取分离每行 0th 列独立碎元，并位切并压填满一个新的封包：
    int8_t r0_c0 = (int8_t)(b_row0_pack & 0xFF);
    int8_t r1_c0 = (int8_t)(b_row1_pack & 0xFF);
...
    int32_t b_col0_pack = ((r3_c0 & 0xFF) << 24) | ((r2_c0 & 0xFF) << 16)
                        | ((r1_c0 & 0xFF) <<  8) |  (r0_c0 & 0xFF);

    // [3] 直接倾泻四线合并的单指令积和
    acc0 = __dp4a(*reinterpret_cast<const int32_t*>(&A[row * K + i]), 
                  b_col0_pack, acc0);
```

**要点解读**：

- 为什么要在局部写这么繁杂的 Pack 代码？因为这几个小型的 `shift` 和 `AND 0xff` 操作被锁定在了无延迟的寄存器内部域，远比你去开四次读条去向显存（甚至是 Shared Mem）请求要便宜几个数量级。

## 结果与边界

### 性能对比

> **测试条件**：双 RTX 4090 ($sm\_89$), nvcc -O3
> **数据来源**：`Results/07_Quantization.md` 原始实机日志，均以 10-100 次打脸求均值避开冷启动

**1. 极速量化类型转换**

对于数据体量 `N=10,485,760` (纯 40MB 区间)

| 实现类型转接策略 | Kernel 执行耗时 | 对位表观带宽界限 | 数据性质 |
|------------------|-----------------|------------------|----------|
| FP32 → FP16 Cast | **0.02 ms**     | **~2911 GB/s**   | [实测] |
| FP32 → INT8 Per-Tensor | **0.02 ms**  | **~2166 GB/s**   | [实测] |
| FP32 → INT8 Per-Channel| 0.03 ms         | ~1762 GB/s   | [实测] |

此处于显存理论 1008 GB/s 之上暴切 2900 GB/s 的带宽极值现象，纯因测试时此 40MB 块恰逢全面缩入 RTX 4090 **72MB 特大容量 L2 全缓存区间**内。这也证明只要剔除内存读写羁绊，全片纯硬件类型换接的时间成本只有区区 10 几微秒水平，极其适合插空在极快层归一化后被强行算子融合掉。

**2. GEMM 算武核心战力（极小矩阵 1024x1024 验证层）**

| 算术实现模型 | 总 Kernel 时长 | 对准基线加速倍率 | 实测绝对算力 | 数据性质 |
|--------------|----------------|------------------|------------|----------|
| FP16 Naive   | 0.42 ms        | 1.00x            | - | [实测] |
| FP16 `half2` Vec | **0.22 ms** | **1.91x**        | 9.69 TFLOPS| [实测] |
| INT8 `dp4a` Vec  | **0.19 ms** | **2.14x** (对INT8)| **11.31 TOPS**| [实测] |

受限评测体量以及本节纯手写底层 CUDA 探针尚未挂载顶层的 `mma.sync` Tensor Core 特技（在下章节详解），在纯算术发车层面我们稳稳榨取翻倍以上。

### 边界条件与局限

- **INT8 之非无限扩展性**：在 $1024 \times 1024$ 型规模下，INT8 的 0.19 相比于 FP16 的 0.22 并未彻底拔开身距（并未达到期望的成倍差异），其成算就在于，此等极微体量下的数据量早就彻底缩卷入 L2 Cache 池引发 Memory Bound 的强退潮失效。只有到巨型矩阵（如 $K \ge 8192$）越过 L2 总线护墙后，位块占地极宽窄的 INT8 才会一举撕裂 FP16 而绝尘。

## 常见误区

1. **误区**：一旦大模型使用量化策略，整个推演都会变快。
   **实际**：对于极短长度输入或是小 batch 批设下，由于系统被**计算延拓（Compute Bound）**或者极高调度频率挟制，多一次解封（Dequant）都是沉没开销。仅有被 Memory Bound 阻滞的情境下大幅度量化缩减方能在物理端线上反攻。
2. **误区**：所有的硬件芯片都必须去按块或者层面执行 `dp4a`。
   **实际**：在 Pascal (sm_61) 以上它是底单基础配器。但在 Turing 及往后的高端算力芯片中，专门用于 8位 压排连推并发的 **INT8 Tensor Core** 早就成为远超普通的 `dp4a` 的几倍数十倍存在。本章仅用于打磨其深算装载的原理体系验证。

## 系列导航

### 前置阅读

| 文章 | 关系 |
|------|------|
| [04_GEMM_Optimization_Register_Tiling.md](04_GEMM_Optimization_Register_Tiling.md) | 在掌握量化切分之前，必须吃透该文关于寄存器瓦片是如何在 32 位体系里封锁共享内存通道的深源理念 |

### 推荐后续

| 文章 | 关系 |
|------|------|
| [09_Tensor_Core_WMMA_Mixed_Precision.md](09_Tensor_Core_WMMA_Mixed_Precision.md) | 在学会利用 `half2` 在底端汇编压榨双发后，去阅读此文知晓怎样用一排原生函数直接唤醒整块光固化硅晶列阵去吞并算量 |
