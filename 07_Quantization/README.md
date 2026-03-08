# 07_Quantization: 混合精度与量化计算架构

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

在大规模神经网络乃至 LLM 推理领域，显存带宽往往远比算力先耗尽。量化（Quantization）技术应运而生：通过缩减浮点数的位宽，成倍地降低全局访存压力，甚至能利用特殊的低精度硬件算力（如 INT8 Tensor Core）。本章的目标是学习并掌握在 CUDA 侧如何手动安全地进行降维转换与半精度运算，这非常考验对硬件基础数据类型的掌控力。

目录下的实现涵盖了核心量化流：

- `01_fp16_gemm/`：引入 `half` 数据类型与专用的 `__half2` 向量化原语，展示如何正确地调度使用半精度代替常规单精度执行 GEMM。
- `02_int8_gemm/`：介绍 `int8_t` 的紧凑排布方式与点积指令 `__dp4a` 的使用，这是深度学习推理端的性能杀手锏。
- `03_quant_dequant/`：揭示对称/非对称量化的基础换算，提供将高精度张量按通道或层压缩为低精度再反解回来的安全实现。

## 2. 原理推导与数学表达 (Math & Logic)

对称性线性量化（Symmetric Linear Quantization）的基础公式是将一个浮点范围 $[ -|max|, +|max| ]$ 映射到 $[ -127, 127 ]$ 的 8 位带符号整数系：

放缩因子（Scale）：
$$ S = \frac{|max(X_{fp32})|}{127.0} $$

量化过程：
$$ X_{int8} = \text{round}( \frac{X_{fp32}}{S} ) $$

反量化过程（在核心算术输出后，用于流回标准误差空间）：
$$ X_{fp32\_out} = X_{int8\_out} \times S_A \times S_B $$

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

以 INT8 中的 `__dp4a` 内建指令（Dot Product of 4 8-bit integers and Accumulate）为例，硬件是如何在一个时钟周期内消化掉它的：

```text
[一个 32-bit 的物理寄存器 Reg_A]
+------+------+------+------+
|  8b  |  8b  |  8b  |  8b  |
| a[3] | a[2] | a[1] | a[0] |
+------+------+------+------+

[另一个 32-bit 的物理寄存器 Reg_B]
+------+------+------+------+
|  8b  |  8b  |  8b  |  8b  |
| b[3] | b[2] | b[1] | b[0] |
+------+------+------+------+
             || 
      __dp4a(Reg_A, Reg_B, C) 
     (触发专用的整数 ALU 算术引擎)
             \/
[输出为 32-bit 的整数] C = C + a[0]*b[0] + ... + a[3]*b[3]
```

内存读取上：以前一个 `float` 占用 4 bytes，现在能一次性利用 `int4` 这种复合结构读取 16 bytes 对应 16 个参数，极大地榨干了总线带宽。

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `02_int8_gemm/int8_gemm.cu` 中的 INT8 打包快速点积逻辑：

```cpp
// ⚠️ 这里强制将两个 32 bit 变量作为四个 int8 处理
// 依赖了 union 或者 char4 的内存 reinterpret
char4 val_a = ...; 
char4 val_b = ...;

// C 的累加器使用 int32 以防止中途数据溢出截断
int sum = 0; 
for(int k=0; k < K/4; k++) {
    // 提取两个 uint32_t 直接扔进内建的四位内积指令中
    // 这一行代码替代了原本的 4 此乘法 + 4 次加法循环
    sum = __dp4a(*((int*)&val_a), *((int*)&val_b), sum); 
}
```

## 5. 性能基准与分析视角 (Performance & Profiling)

- **基准**：对比同等矩阵大小下的 FP32 标准 GEMM 计算效率。
- **典型分析**：使用 NCU 可查看 DRAM 吞吐量。INT8 矩阵相较于 FP32 在理论上缩减了 75% 的数据搬运压力。并且由于 `__dp4a` 的一条指令完成了四对乘加，指令发射（Instruction Issue）的压力也只有原本的四分之一。

## 6. 编译指引与参考资料 (Compile & References)

```bash
# INT8 dp4a 要求架构至少为 sm_61
nvcc -O3 -arch=sm_89 int8_gemm.cu -o run_int8
# 使用 ncu 探测专用的 integer/FP16 指令占比
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./run_int8
```

- 参考资料: NVIDIA Developer Blog: "Fast INT8 Inference for Autonomous Vehicles with TensorRT".
