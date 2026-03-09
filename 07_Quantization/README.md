# 07_Quantization: 混合精度、量化与指令级内积优化

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

在大规模神经网络（例如 LLM 推理）领域，显存带宽容量的匮乏往往比算力更早成为系统瓶颈。量化（Quantization）技术应运而生：通过缩减浮点数的位宽，不仅能成倍地降低全局访存压力，更能极大程度榨取低精度专用算术管线（如 `__dp4a` 及 Tensor Core）的海量吞吐潜能。本章的初衷是在脱离繁重的 cuBLAS 闭源算子库的情境下，从零到一手动在 CUDA 中实现精度缩减类型的核函数映射。

子模块实践内容：
- `01_fp16_gemm/`：引入 `half` 和 `half2` 数据类型，展示如何在不丢失过多精度的情况下使用原生 `__hfma2` 进行向量化打包的半精度 GEMM；
- `02_int8_gemm/`：突破定点类型的计算障壁，引入了专门针对深度学习推理核心开发的 8 位带符号点积累加指令 `__dp4a` 进行极限性能释放；
- `03_quant_dequant/`：深入详解网络上下游中进行 Per-tensor（全局）与 Per-channel（沿特定轴分布通道级排布）的对称线性量化和反量化机制。

## 2. 原理推导与数学表达 (Math & Logic)

对称性线性量化（Symmetric Linear Quantization）的基础公式是将一个浮点范围 $[ -|max|, +|max| ]$ 映射到 $[ -127, 127 ]$（对于 INT8）：

**放缩因子（Scale）：**
$$ S = \frac{|max(X_{fp32})|}{127.0} $$

**量化过程 (Quantization)：**
$$ X_{int8} = \text{round}\left( \frac{X_{fp32}}{S} \right) $$
*(越界的值进行裁切 / saturate)*

**反量化过程 (Dequantization)：**
$$ X_{fp32\_out} = X_{int8\_out} \times S $$
*(若计算乘积结果 $C = A \times B$，最后反量化时则需要同时乘以二者的 Scale：$C_{fp32} = C_{int32\_accum} \times S_A \times S_B$)*

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

以 INT8 中的 `__dp4a` (Dot Product of 4 8-bit integers and Accumulate) 内建指令为例，其能够在一条指令中打包执行 `4个乘法 + 1个累加`。

```text
[普通的 32-bit 的物理寄存器 Reg_A]
+------+------+------+------+
|  8b  |  8b  |  8b  |  8b  |
| a[3] | a[2] | a[1] | a[0] |
+------+------+------+------+

[普通的 32-bit 的物理寄存器 Reg_B]
+------+------+------+------+
|  8b  |  8b  |  8b  |  8b  |
| b[3] | b[2] | b[1] | b[0] |
+------+------+------+------+
             || 
     __dp4a(Reg_A, Reg_B, C) 
    (交由 SM 内专属的整数算术单元运算)
             \/
[输出为 32-bit 的积] C = C + a[0]*b[0] + ... + a[3]*b[3]
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `02_int8_gemm/int8_gemm.cu` 向量化版本中的核心：

```cpp
// 在内存中一次取 4 个对应的 int8 进行内积计算
// 每次移动步长为 4 这个复合通道 
for (int k = 0; k < K; k += 4) {
    // 强制把地址上的四个 int8_t 当作一个 uint32_t 直接读取，最大化内存事务连贯性
    uint32_t valA = *((const uint32_t*)&A[row * K + k]);
    uint32_t valB = *((const uint32_t*)&B[k * N + col]);

    // 调用汇编内建函数
    // 用 int32 保存 sum 防止累加时溢出
    sum = __dp4a(valA, valB, sum);
}
```

## 5. 基准表现与评估剖析 (Performance Data)

在双卡 **RTX 4090** 下我们拿到了炸裂级的吞吐量结果（测试维度 $1024 \times 1024$）：

- **FP16 GEMM Vectorized**：
  - 加速比超五万！达到了惊人的 **`9723.10 GFLOPS`（合 9.72 TFLOPS）**，相较于纯标量的 `FP32`，这正是数据结构压缩以及 `half2` 并发打包取得阶段性翻倍的有力证据。耗时压低至 `0.22 ms`。
- **INT8 GEMM Vectorized (__dp4a)**：
  - 整机算力彻底被引爆！基于普通的纯算力调用（非 Tensor Core 直接矩阵乘），硬干出了 **`11.40 TOPS`** (Tera Operations Per Second) 的标量极限性能。测得加速比相较 CPU 快超 8000 倍。
- **Quantization & Dequantization 精度层交互**：
  - 基于显存控制器上限测试 `FP32 <-> FP16` / `INT8` 转置过程中，依靠向量化和 L2 Cache 全击穿策略，录得转换带宽最高跑出 **`2927.28 GB/s`**（由于数据集 4MB 落入 72MB 极限超大 L2 所致），彻底瓦解格式转换带来的传输开销负担。

## 6. 编译及参考 (Compile & References)

```bash
cd build && make -j4 fp16_gemm int8_gemm quant_dequant
./07_Quantization/02_int8_gemm/int8_gemm
```

深度资料指引：
- NVIDIA Developer Blog: [Fast INT8 Inference for Autonomous Vehicles with TensorRT](https://developer.nvidia.com)
- CUDA Math API - [Half Precision Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html)
