# 04_GEMM_Optimization 矩阵乘法极致优化

## 一、 全景导览与学习目标

该子项目在 CUDA-Practice 学习体系中属于 **工业级算子调优 (L3)** 阶段。**通用矩阵乘法 (GEMM, General Matrix Multiply)** 是深度学习、科学计算的绝对核心基石。在 `01_Basics` 我们仅实现了 Naive 与简单的 Shared Memory Tiling 矩阵乘法，而本章节则是一场**算力极限压榨**之旅，目标是突破内存带宽墙 (Memory Wall)，无限逼近 GPU 的理论计算峰值。

本项目层层递进地展示了如何从显存、L1/L2 Cache 一路优化到寄存器分配与指令级并行，其核心技术均被 NVIDIA cuBLAS 与 CUTLASS 等工业级算子库广泛采用。

包含以下核心演进体系：

- `01_tiled_gemm/tiled_gemm.cu`：**基础 Tiling 与 1D/2D 粗化**。对基础版分块引入 1D / 2D 方向的 Thread Coarsening (线程粗化)，有效提高了数据服用率。
- `02_advanced_gemm/advanced_gemm.cu`：**向量化访存与双缓冲流水线**。引入了 `float4` 强行将 4 个 float 捆绑读取，榨干 Global Memory 总线带宽；引入 `Double Buffering` 掩盖访存延迟。
- `03_register_tiling/register_tiling.cu`：**工业级 Register Tiling**。完全放弃以标量为单位的思维，每个线程在寄存器中维护一个 $TM \times TN$ 的二维网格，并利用 FMA (融合乘加) 展开外积迭代，最终与 cuBLAS 展开正面交锋。

---

## 二、 原理推导与数学表达

对于 $C = A \times B$ 的通用矩阵乘，其中 $A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N}, C \in \mathbb{R}^{M \times N}$：
标准定义为：
$$ C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \times B_{k,j} $$

### 1. 2D 分块 (Tiling) 的数学重写

为了在 GPU 上利用高速 Shared Memory 缓存数据，我们在三个维度上以步长 $BM, BN, BK$ 将问题分割。外层由 Block 遍历 $M$ 和 $N$，内层由 Block 在 $K$ 维度以 $BK$ 为单位步进：
$$ C_{\text{block\_row}, \text{block\_col}} += \sum_{step=0}^{K/BK-1} \left( A_{sub[BM \times BK]} \times B_{sub[BK \times BN]} \right) $$

### 2. Register Tiling：基于外积 (Outer Product) 的极致拆解

在 `register_tiling` 中，一个 Block 被分配了 $BM \times BN$ 大小的 $C$ 块，但为了极度降低寄存器访问 Shared Memory 的开销，每个 Thread 必须被赋予更重的计算任务——**每个线程负责计算 $C$ 的一个 $TM \times TN$ 微块**。
假设 $TM=8, TN=8$，在一个 $BK$ 的小迭代内，数学上相当于在做向量的外积：
对于线程负责的局部 $C_{reg}[8 \times 8]$：
$$
\mathbf{C}_{reg} += \sum_{k=0}^{BK-1} \mathbf{a}_{reg}^{(k)} \otimes \mathbf{b}_{reg}^{(k)}
$$
其中，$\mathbf{a}_{reg}^{(k)}$ 是长度为 $TM$ 的列向量，$\mathbf{b}_{reg}^{(k)}$ 是长度为 $TN$ 的行向量。通过这种方式，读取 $TM+TN = 16$ 个元素，就能完成 $TM \times TN = 64$ 次 FMA（融合乘加）计算，计算-访存比 (Compute-to-Memory Ratio) 达到了惊人的 **4:1**。

---

## 三、 硬核内存映射解析

本节以 `03_register_tiling` 中 $BM=128, BN=128, BK=8$, $TM=8, TN=8$ 为例，揭示最硬核的数据级联复用分发过程。

### Register Tiling 层次化内存分配时序

```mermaid
graph TD
    classDef global fill:#f9d0c4,stroke:#333,stroke-width:2px;
    classDef shared fill:#fcf1c8,stroke:#333,stroke-width:2px;
    classDef reg_mem fill:#bbf,stroke:#333,stroke-width:2px;
    classDef compute fill:#dfd,stroke:#333,stroke-width:2px,color:#000;

    subgraph "全局内存 (HBM)"
      A_global[Matrix A <br> Global Mem]:::global
      B_global[Matrix B <br> Global Mem]:::global
    end

    subgraph "块级缓存: K 维度每轮读取一次 (256线程协作提取)"
      sA[sA: 128 (BM) × 8 (BK) <br> Shared Mem]:::shared
      sB[sB: 8 (BK) × 128 (BN) <br> Shared Mem]:::shared
    end
    
    subgraph "线程级私有: 内层循环 dotIdx 从 0 到 7 (每个线程独立操作)"
      regA[regA 寄存器 <br> 长度 TM=8 存放列]:::reg_mem
      regB[regB 寄存器 <br> 长度 TN=8 存放行]:::reg_mem
      regC[regC 累加器 <br> TM×TN = 64 个浮点数]:::compute
    end

    A_global -- "加载至缓存" --> sA
    B_global -- "加载至缓存" --> sB
    
    sA -- "读取 1 列 (8 float)" --> regA
    sB -- "读取 1 行 (8 float)" --> regB
    
    regA -- "外积 (Outer Product)<br>64次 fmaf" --> regC
    regB -- "外积 (Outer Product)<br>64次 fmaf" --> regC
```

**📊 映射核心洞察**:
在这个三级缓存层递模型中：

- 线程数计算：整个 Block 的大小并不是 $128 \times 128 = 16384$ (远超 CUDA 1024 上限)，而是 $(128/8) \times (128/8) = 16 \times 16 = \mathbf{256}$ **个线程**。
- 这 `256` 个幽灵般的线程，在外力协作下从 Global Mem 拽出 $128\times8 + 8\times128 = 2048$ 个数据塞到 Shared Mem。随后，每个线程独立自主地切入 Shared Mem，各自吞噬属于自己的 $8$ 行和 $8$ 列，疯狂在自己的 $64$ 块腹肌（`regC`）上进行点乘！

---

## 四、 关键源码逐行解剖

### Register Tiling 的外积累加核心 (Outer Product FMA)

截取自 `03_register_tiling/register_tiling.cu` 最暴力的核心循环地带（发生在内存从 Shared Mem 抵达 Register 后）。

```cpp
// 此时已经准备好进入内层的 dotIdx = 0 倒 BK-1 (比如 8) 循环中
for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // 💡 物理硬件映射：从 L1 Cache (Shared Memory) 中，将 8 个数据扒入高速的 Thread File Register！
    for (int i = 0; i < TM; ++i) {
        regA[i] = sA[threadRow * TM + i][dotIdx]; 
    }
    // 同理，将 B 的一行扒拉到寄存器
    for (int j = 0; j < TN; ++j) {
        regB[j] = sB[dotIdx][threadCol * TN + j];
    }
    
    // 💥 算力核爆区：TM × TN 次 FMA (Fusion Multiply-Add)
    // 注意：这里的循环体不涉及任何 Global 或 Shared Memory 的访存指令！
    // 它是纯纯的 ALU (算术逻辑单元) 与 Register 之间的直接对话！
    // 编译器看到这里连续的、无逻辑依赖的循环，会自动进行 Loop Unrolling (循环展开)
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            regC[i][j] = fmaf(regA[i], regB[j], regC[i][j]);
        }
    }
}
```

**深入注解**：
为何这段代码如此致命（高性能）？因为 `fmaf` 指令可以在一个时钟周期内完成一乘一加，而这里由于没有任何外部依赖（全在 Register 当中），SM 内的 Warp Scheduler 可以无缝且流水线般地派发几十条 `fmaf`，直接将硬件算力填满，这也是将 TFLOPS 冲到几十级别不可或缺的核心底座。

---

## 五、 性能基准与分析

所有数据提取自 `Results/04_GEMM_Optimization.md` 真实日志：

- **测试硬件**: NVIDIA GeForce RTX 4090 (sm_89) × 2, Linux 环境
- **测试规模**:
  - `Tiled & Advanced`: Matrix $1024 \times 1024$
  - `Register Tiling vs cuBLAS`: Matrix $2048 \times 2048$ (超大规模算力交锋)

### 1. 初步级优化 ($1024 \times 1024$) 基准表现

| 实现版本 | Kernel 时间 | 计算性能 (GFLOPS) | vs CPU (2117 ms) 加速比 |
| -------- | ----------- | ---------------- | ------------- |
| CPU 参考 | 2117.46 ms | ~1 GFLOPS | 1x |
| GPU (Tiled GEMM 基础) | 0.33 ms | ~6500 GFLOPS | ~6400x |
| GPU (Vectorized float4) | 0.38 ms | 6820 GFLOPS | ~6700x |
| GPU (Double Buffer) | 0.31 ms | 6820 GFLOPS | ~6700x |
| **GPU (2D Register Tiled)** | **0.15 ms** | **14055.10 GFLOPS** | **~13858x** |

*(在 $1024 \times 1024$ 级别，Register Tiled 以 14 TFLOPS 一骑绝尘，是普通 Tiled 性能的 $2.14\text{x}$。)*

### 2. 算力之巅：手写 vs 官方 cuBLAS ($2048 \times 2048$ 大核碰撞)

| 实现版本 | Kernel 时间 | 浮点吞吐巅峰 | 评测对决 |
| -------- | ----------- | ---------------- | ------------- |
| GPU (Register Tiling 手写) | 0.60 ms | **28.79 TFLOPS** | 虽然惨败但也硬气（纯 CUDA C） |
| **cuBLAS (官方神级汇编)** | **0.30 ms** | **57.49 TFLOPS** | **行业巅峰标杆** |

*\*注：我们的手写 Register Tiling 跑到了惊人的 $28.79 \text{ TFLOPS}$。这相当于达到了 cuBLAS (`57.49 TFLOPS`) 约 **50.1%** 的性能表现。要知道，cuBLAS 底层使用了 SASS 汇编并解决了所有的 Bank Conflict 并且应用了极其精密的流水线调校。*

````mermaid
xychart-beta
  title "Kernel 算力巅峰对比：手写 Tiling 攀峰之旅 (TFLOPS, 越高越好)"
  x-axis ["Tiled 基础", "Register(手写)", "cuBLAS"]
  y-axis "性能 (TFLOPS)" 0 --> 60
  bar [6.5, 28.79, 57.49]
````

**📊 深入分析：**

1. **Double Buffering 与 Vectorized 的收益**：在 1024 尺度上，`float4` 向量化相较于基础 `Tiled` 有轻微性能下降反转（0.38ms），这是因为当 Tiled 已不再严重受制于内存带宽时，强行构造 `float4` 可能产生了一些冗余代价；但加入 `Double Buffering` 的异步思想流水线后，时间立刻回落到 $0.31\text{ms}$，展现了良好的延迟掩盖效应。
2. **Register Tiling 的量变引起质变**：它通过外积矩阵法将内存-算力要求转化为 **$O(V) \mapsto O(V^2)$**。直接让性能冲上了几十 TFLOPS。虽然未能击溃拥有底层汇编特权的 Tensor Core/cuBLAS，但达到了 50.1% 的压榨率，证明这已经是单纯使用 CUDA C 高级语言手写优化的理论极致上限区带。

---

## 六、 编译及参考资料

### 编译与标准运行指令

借助根目录的统一 `CMakeLists.txt` 构建目标：

```bash
# 1. 切换至项目根目录并执行整体配置（首次构建）
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. 独立编译对应的子项目 Target 
cmake --build build --target tiled_gemm -j8
cmake --build build --target advanced_gemm -j8
cmake --build build --target register_tiling -j8

# 3. 运行基础验证程序进行观测 (包含 cuBLAS 的基准对比挑战)
./build/04_GEMM_Optimization/01_tiled_gemm/tiled_gemm
./build/04_GEMM_Optimization/02_advanced_gemm/advanced_gemm
./build/04_GEMM_Optimization/03_register_tiling/register_tiling

# 4. 可选测试：利用 ncu 剖析 register_tiling 中疯狂吞咽的吞吐量
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./build/04_GEMM_Optimization/03_register_tiling/register_tiling
```

### 推荐阅读

- [NVIDIA Developer Blog: How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM) —— （**神级博文**：一步一步教你如何逼近 cuBLAS 性能的完整手札，本项目大量思想源生于此。）
- [Volta Architecture and Tensor Core Optimization Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html) —— 如果想了解进一步优化到 Tensor Core 应当阅读。
- [CUDA C++ Programming Guide: Shared Memory and Bank Conflicts](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) —— 理解阶段 3 加载内存为什么要警惕 Padding！
