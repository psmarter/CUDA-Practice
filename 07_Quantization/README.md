# 07_Quantization 混合精度与量化计算

## 一、 全景导览与学习目标

该子项目处于 CUDA-Practice 学习体系的 **前沿级 (L4)** 阶段。在现代深度学习系统（尤其是 LLM 推理系统 vLLM / TensorRT-LLM）中，计算能力与带宽的天平已严重失衡，内存墙 (Memory Wall) 成为最大阻碍。采用低精度 (FP16/BF16) 以及定点整数量化 (INT8/INT4) 已成为压榨硬件极限的必经之路。

本模块聚焦于**数据位宽降维打击**技术，摆脱传统 FP32 的思维惯性，带领你直接操作低精度指令与并行打包向量：

- `01_fp16_gemm`：**半精度浮点启蒙**。演示从朴素 `half` 乘加，到 Shared Memory 优化，再到利用 `half2` 向量化读取并使用 SM5.3+ 原生 `__hfma2` 指令拉满吞吐的进阶演化。
- `02_int8_gemm`：**整形暴力美学 (dp4a)**。学习如何手动打包 (Pack) 字节列，并使用专为推理打造的 `__dp4a` (Dot Product Accumulate 4) 指令，一个时钟周期计算 4 个 INT8 乘加，使算力发生质的飞跃。
- `03_quant_dequant`：**量化基石设施**。实现工业界最常见的 Per-Tensor 和 Per-Channel 线性量化 (FP32 ↔ INT8) 以及类型强转 (FP32 ↔ FP16) 算子，探讨内存带宽极限。

---

## 二、 原理推导与数学表达

### 1. 线性量化与反量化 (Linear Quantization)

对称线性量化的核心是将高精度范围的数据映射到一个离散且范围极小的整型空间。

假设输入浮点张量为 $\mathbf{X}$，缩放因子为 $Scale$ (代表每个整数刻度对应的真实浮点间隔)：

- **FP32 至 INT8 量化方程**:
  $$ \mathbf{X}_{INT8} = \text{Clamp}\left( \text{Round}\left( \frac{\mathbf{X}_{FP32}}{Scale} \right), -128, 127 \right) $$
- **INT8 至 FP32 反量化重构方程**:
  $$ \mathbf{X}_{FP32}' = \mathbf{X}_{INT8} \times Scale $$

在 `Per-Channel` 设定下，$Scale$ 是一个向量 $S \in \mathbb{R}^{C}$，则元素 $x_{n,c}$ 的量化缩放由 $S_c$ 决定，这对权重的离群值 (Outliers) 包容性更强。

### 2. INT8 GEMM (dp4a) 向量点积法则

在 FP32 标准点乘中 $C_{m,n} = \sum_k A_{m,k} B_{k,n}$ 是一对一累加。
在 CUDA `dp4a` 指令中，数据被分块打包，以 $4$ 为粒度：
$$ C_{m,n} = \sum_{k=0}^{K/4 - 1} \text{dp4a} \left( [A_{m, 4k}, \ldots, A_{m, 4k+3}], [B_{4k, n}, \ldots, B_{4k+3, n}] \right) $$
一次运算直接消耗 $4 \text{ Bytes} \times 2$，这不仅对 ALU 是极速并行，还完美契合了全球内存 32-bit 对齐读取的最佳步长。

---

## 三、 硬核内存映射解析

在 `02_int8_gemm` 的终极向量化 (Vectorized dp4a) 优化中，极其反直觉的一步是处理矩阵 `B`（列优先访问带来的非连续恶梦）。我们要让每个线程**强行啃下 4 个列**，以实现完美的向量内存重组。

### Vectorized INT8 手工装填 (Packing) 示意图

此时每个 Thread 负责生成输出矩阵 `C` 中的 $1 \times 4$ 行向量（对应 4 个由于并行计算产生的 int32_t）。因此，在主循环每次步进（行进 4 个 K）时，线程会读入 A 的连续行数据 和 B 的 $4 \times 4$ 小碎片。

```mermaid
graph TD
    classDef regA fill:#f9d0c4,stroke:#333;
    classDef regB fill:#c4e0f9,stroke:#333;
    classDef regPack fill:#f9f0c4,stroke:#333;

    subgraph "Thread (m, 4*n) 从 Global Mem 的粗粒度加载"
        A_read["读取 A 行上的 1 个 int32_t<br>包含: A[0], A[1], A[2], A[3]"]:::regA
        B_read["读取 B 的 4 个 int32_t (横向跨列读取)<br>Row0: B_00, B_01, B_02, B_03<br>Row1: B_10, B_11, B_12, B_13<br>Row2: B_20, B_21, B_22, B_23<br>Row3: B_30, B_31, B_32, B_33"]:::regB
    end

    subgraph "手工移位重组 (Transposing in Registers)"
        P0["Col 0 Pack 📦: [B_30, B_20, B_10, B_00]"]:::regPack
        P1["Col 1 Pack 📦: [B_31, B_21, B_11, B_01]"]:::regPack
        P2["Col 2 Pack 📦: ..." ]:::regPack
        P3["Col 3 Pack 📦: ..." ]:::regPack
    end

    A_read --> |直接共用| P0 & P1 & P2 & P3
    B_read --> |位运算 (Shift & Mask)| P0 & P1 & P2 & P3
    
    subgraph "并行 dp4a 发射"
        dp0["__dp4a(A_pack, Col0, sum0)"]
        dp1["__dp4a(A_pack, Col1, sum1)"]
        dp2["__dp4a(A_pack, Col2, sum2)"]
        dp3["__dp4a(A_pack, Col3, sum3)"]
    end
    
    P0 --> dp0
    P1 --> dp1
    P2 --> dp2
    P3 --> dp3
```

**💡 核心收益**：
通过 `reinterpret_cast<const int32_t*>` 横跨列域读取 B 的 4 字节，彻底消灭了分散取址带来的内存跳跃碎片化！随后只需要利用极快廉价的位运算（`>>` 和 `&`）在寄存器内将提取出来的元素进行手工转置对齐，再送入四管齐下的 `dp4a` 槽位，达到了算力与带宽的双重爆炸。

---

## 四、 关键源码逐行解剖

### 1. Vectorized FP16 中 half2 与 `__hfma2` 的绝杀配合

节选自 `01_fp16_gemm/fp16_gemm.cu`：

```cpp
// 🚀 使用 half2 处理一次迭代的末端对削
half2 sum2 = __float2half2_rn(0.0f); // 并发存储两个结果向量的槽

for (int i = 0; i < N; ++i) {
    // 技巧 1: 标量变向量广播. 将单一 A 复制到两个坑位 [A, A]
    half2 a_val2 = __halves2half2(A[row * N + i], A[row * N + i]); 
    
    // 技巧 2: 以 double 带宽的方式强行加载 B 矩阵内存中相邻的两个半精度点并封箱为 half2
    half2 b_val2 = *reinterpret_cast<const half2*>(&B[i * K + col]); 

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    // 技巧 3: 调用 Volta+ 原生硬件指令执行成对的点积: [a0*b0+c0, a1*b1+c1]！
    sum2 = __hfma2(a_val2, b_val2, sum2); 
#else
    // (向下兼容的回退补丁省略)
#endif
}

// 技巧 4: 将 [sum_left, sum_right] 一次性砸入 C 矩阵 Global Memory
*reinterpret_cast<half2*>(&C[row * K + col]) = sum2; 
```

**解剖结论**：
FP16 优化的极致在于尽可能地抹去它与 32bit 操作的边界隙缝，`half2` 的引入直接令 Global Read Request 数量砍半。

### 2. INT8 GEMM 中寄存器级的手工转置装填

节选自 `02_int8_gemm/int8_gemm.cu`：

```cpp
// 将 B 矩阵本应杂乱分布的一行横切面，通过 32-bit 野蛮加载
int32_t b_row0_pack = *reinterpret_cast<const int32_t*>(&B[(i + 0) * K + col]); 
int32_t b_row1_pack = *reinterpret_cast<const int32_t*>(&B[(i + 1) * K + col]);
int32_t b_row2_pack = *reinterpret_cast<const int32_t*>(&B[(i + 2) * K + col]);
int32_t b_row3_pack = *reinterpret_cast<const int32_t*>(&B[(i + 3) * K + col]);

// 痛点：我们要的是第 0 列的纵切面打包，必须通过掩码(Mask)进行转置提取！
int8_t r0_c0 = b_row0_pack & 0xFF;                  // 抽出行 0，列 0 
int8_t r1_c0 = b_row1_pack & 0xFF;                  // 抽出行 1，列 0 
int8_t r2_c0 = b_row2_pack & 0xFF;                  // 抽出行 2，列 0 
int8_t r3_c0 = b_row3_pack & 0xFF;                  // 抽出行 3，列 0 

// 逆天组装：把抽出的四个 byte 在小端序下重组成一个给 dp4a 食用的 INT32！
int32_t col0_val = ((r3_c0 & 0xFF) << 24) | ((r2_c0 & 0xFF) << 16) | ((r1_c0 & 0xFF) << 8) | (r0_c0 & 0xFF);

// 最后直接爆破计算
sum0 = compat_dp4a(a_val, col0_val, sum0);
```

---

## 五、 性能基准与分析

所有数据提取自 `Results/07_Quantization.md` 真实日志：

- **测试硬件**: NVIDIA GeForce RTX 4090 (sm_89) × 2, Linux 环境
- **测试条件**:
  - 量化独立算子规模: $10 \text{ M}$ (1千万) Elements ($40 \text{ MB}$ payload)
  - GEMM 规模: $A_{1024 \times 1024} \times B_{1024 \times 1024}$，执行 10 次取均值。

### 1. 独立量化管道测试（Bandwidth Bound）

低精度转换的核心是“跑满显存线”。

| 转换核心路线 | CPU(ms) | GPU(ms) | GPU 有效带宽评估 | 评价 |
| ------------ | ------- | ------- | ---------------- | ---- |
| FP32 → INT8 (全局缩放)| $86.65$ | $\mathbf{0.02}$ | $2166.62 \text{ GB/s}$ | L2 缓存命中溢满，达到神级瞬移！ |
| INT8 → FP32 (反量化)  | $8.34$  | $\mathbf{0.02}$ | $2440.42 \text{ GB/s}$ | 带宽评估已远超真实 DRAM，处于纯寄存器/L1命中热区 |
| FP32 → FP16 (类型降级)| $95.77$ | $0.02$  | $2911.98 \text{ GB/s}$ | 同上，纯缓存收缩率测试 |

### 2. 密集算力核弹：FP16 与 INT8 GEMM (Compute Bound)

这里展现了打破常规算力天花板的艺术。

| 数值类型 | 核武器代号 (优化版本) | GPU 执行耗时 (ms) | GPU 理论输出 | 加速比基准比对 |
| :------- | :-------------------- | :---------------- | :----------- | :------------- |
| **FP16** | Naive (单发点积)      | $0.424$ | - | 基准点 (1.00x) |
| **FP16** | Tiled (共享显存折叠)    | $0.331$ | $6436 \text{ GFLOPS}^*$ | 1.28x |
| **FP16** | Vectorized (`hfma2`)     | $\mathbf{0.220}$| $\mathbf{9697.25 \text{ GFLOPS}}$ | **1.91x** (接近10 TFLOPS极速)|
| **INT8** | Naive                 | $0.407$ | - | 基准点 (1.00x) |
| **INT8** | dp4a (硬件指令级打包)     | $0.276$ | $7.78 \text{ TOPS}^*$ | 1.48x |
| **INT8** | Vectorized dp4a         | $\mathbf{0.190}$| $\mathbf{11.31 \text{ TOPS}}$ | **2.14x** (算力碾压)|

*(注：带 `*` 标理论数据由代码框架中提供的基础耗时代入推算而得，真实表项源于官方统计的实际打表值)*

````mermaid
xychart-beta
  title "中等矩阵 (1Kx1K) 推理极速耗时压缩历程 (ms, 更低更好)"
  x-axis ["FP16 Naive", "FP16 Tiled", "FP16 Vector", "INT8 Naive", "INT8 dp4a", "INT8 Vector dp4a"]
  y-axis "执行耗时 (ms)" 0 --> 0.5
  bar [0.424, 0.331, 0.220, 0.407, 0.276, 0.190]
````

**💡 性能解析**:

1. **指令吞吐碾压**：从 FP16 到 INT8 最高级，耗时从 `0.22ms` 进一步压缩到了惊人的 `0.19ms`。这是在同等内存消耗水平线上，完全由指令集 (`__dp4a` 内建运算器) 提供的高密度四合一聚变带来的硬碰硬的红利 ($>11 \text{ TOPS}$)。
2. **超越物理带宽假象的真理**：由于独立量化算子的数组处于刚好能完全装挂并驻留在 RTX 4090 巨量二级缓存 ($72 \text{ MB}$ L2 Cache) 的规模热区内。我们所测得的夸张带宽 ($2440+\text{ GB/s}$) 反映了 Cache-Hit 下不经过外接针脚的 GPU 内部恐怖搬运速率，这在工业级实战里常常是模型层合并量化的重要基石。

---

## 六、 编译及参考资料

### 编译与标准运行指令

所有工程统一构建管线（建议基于 Linux 环境与 CMake）：

```bash
# 1. 跨目录综合配置 (生成至 build)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. 精确打击编译特定的微观算子
cmake --build build --target fp16_gemm -j8
cmake --build build --target int8_gemm -j8
cmake --build build --target quant_dequant -j8

# 3. 本地发车执行
./build/07_Quantization/01_fp16_gemm/fp16_gemm
./build/07_Quantization/02_int8_gemm/int8_gemm
./build/07_Quantization/03_quant_dequant/quant_dequant

# 4. 高级剖析器诊断 (Nsight Compute 查看指令级带宽受阻情况)
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./build/07_Quantization/02_int8_gemm/int8_gemm
```

### 推荐外延阅读技术栈

- [NVIDIA Tensor Core 核心解读 (PTX ISA)](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html) —— 解析为什么 `dp4a` 以及 `hfma2` 在硬件流水线上如此暴力。
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm) —— 大致了解 LLM 产业界真实推理由该模块里所写的 INT8/FP16 Kernel 拼凑形成的上游体现。
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) —— Google 关于量化乘加背后的基本原理圣经学术论文。
