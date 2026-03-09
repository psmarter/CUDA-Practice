# 05_LLM_Ops 大语言模型核心算子群

## 一、 全景导览与学习目标

该子项目在 CUDA-Practice 学习体系中处于 **前沿工业实践 (L3/L4)** 阶段。随着 LLaMA、GPT 等大规模语言模型 (LLM) 的爆发，Transformer 架构中的少数几个标准算子（Attention, Norm, RoPE, Softmax）成为了决定长文本推理与吞吐能力的生死判官。

本章节不再局限于通用优化，而是完全**面向 Transformer 解剖**，手撕目前工业界大模型前推中最具技术深度的 5 大核心算子：

- `01_softmax`/`softmax.cu`：**Softmax 极限推演**。从两次规约（Naive）到融合单趟（Online Softmax），再到暴力榨干寄存器的 Warp-Reduce 规约机制。
- `02_layernorm`/`layernorm.cu`：**层归一化**。引入 Welford 单趟在线算法与极致的 Warp-per-row 小维度处理模式。
- `03_flash_attention`/`flash_attention.cu`：**Flash Attention**。划时代的注意力机制优化！从 $O(N^2)$ 的显存噩梦（Naive Attention）重构为利用 SRAM Tiling 分块与重计算的 Compute-Bound 形式（Flash V1），并展示进一步的 Macro-Block + 向量化吞吐压榨（Flash V3）。
- `04_rope`/`rope.cu`：**扭转位置编码 (RoPE)**。LLM 的核心位置标定机制，展示了使用 `float2` 交织读写解决零散访存的最佳实践。
- `05_rmsnorm`/`rmsnorm.cu`：**RMSNorm**。LLaMA 系模型标配，直接抹去均值计算的极简归一化及其 Warp Shuffle 高速化实现。

---

## 二、 原理推导与数学表达

LLM 算子的极致优化往往来源于**底层数学公式的等价变换**。

### 1. Online Softmax 递推式 (Safe Softmax 的单趟进化)

传统 Safe Softmax 要求扫一遍求 $max$，再扫一遍求 $e^{x-max}$。通过数学等式变换，我们可以引入修正因子将其融合为一趟操作：
设局部最大值为 $m_i$，局部指数和为 $l_i$，当加入新元素 $x$ 时，新的最大值 $m_{new} = \max(m_i, x)$。则新的指数和可修正为：
$$ l_{new} = l_i \cdot e^{m_i - m_{new}} + e^{x - m_{new}} $$
这就彻底消除了两次访问大块 Global Memory 的带宽噩梦。

### 2. RMSNorm (Root Mean Square Norm)

RMSNorm 相较于标准 LayerNorm 砍掉了减去 $\mu$ (均值) 的步骤，数学定义如下，它的算力消耗降低了近 30%，但能提供相同的收敛能力：
$$ \bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot \gamma_i $$

### 3. Flash Attention 的 Tiling 重计算

在基础的 Attention $O = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$ 中，$S = QK^T$ 是一个 $N \times N$ 的巨型矩阵。
Flash Attention 的数学核心在于：**Softmax 的分母 $l_i$ 可以被分块（Tiling）逐步累加**！
对于第 $j$ 块的局部最大值 $m^{(j)}$ 和局部指数和 $l^{(j)}$，全局状态可通过类似 Online Softmax 的逻辑进行迭代折叠，使得计算过程中 $S$ 矩阵永远只保留在一个小的 SRAM（Shared Memory）块中，计算完即销毁。

---

## 三、 硬核内存映射解析

本节深度剖析 **Flash Attention V3** 的 Macro-Block 优化设计。传统的 Flash Attention V1 会让一个 Block(例如大小 $BR \times BC$) 负责一小块区域，但这会导致 K, V 矩阵在全局显存中被反复加载。

### Flash Attention V3 (Macro-Block) 块级装载与派发流

```mermaid
graph TD
    classDef global fill:#f9d0c4,stroke:#333,stroke-width:2px;
    classDef shared fill:#fcf1c8,stroke:#333,stroke-width:2px;
    classDef thread fill:#bbf,stroke:#333,stroke-width:2px,color:#000;

    subgraph "全局显存 (Global Memory) "
        Q[Matrix Q<br>按行分块加载]:::global
        K[Matrix K<br>1个大循环纵向下滚]:::global
        V[Matrix V<br>跟随 K 同步下滚]:::global
    end

    subgraph "Shared Memory (1块管 4 个 Warp = 128线程)"
        sQ[sQ: 扩容为 128(行) × D]:::shared
        sK[sK: 共享池 32(行) × D]:::shared
        sV[sV: 共享池 32(行) × D]:::shared
    end

    subgraph "Warp 工作流 (并发榨取)"
        W1[Warp 0: 处理 Q 第 0~31 行<br>复用 sK, sV]:::thread
        W2[Warp 1: 处理 Q 第 32~63 行<br>复用 sK, sV]:::thread
        W3[Warp 2: 处理 Q 第 64~95 行<br>复用 sK, sV]:::thread
        W4[Warp 3: 处理 Q 第 96~127 行<br>复用 sK, sV]:::thread
    end

    Q -. "128人搬128行" .-> sQ
    K -. "128人协作搬32行" .-> sK
    V -. "128人协作搬32行" .-> sV

    sQ --> W1 & W2 & W3 & W4
    sK --> W1 & W2 & W3 & W4
    sV --> W1 & W2 & W3 & W4
```

**📊 映射核心洞察**:
在 V3 迭代中，我们直接把加载 `Q` 矩阵的行数扩张了 4 倍 (`128`)，而保持被内侧循环遍历的 `K/V` 行数不变 (`32`)。在 Shared Memory 中，因为 4 个 Warp 同时并向复用同一组刚刚从 Global Mem 吸进来的 `sK` 和 `sV` 数据，这让 `K/V` 从显存中被读取的总次数猛降了 **4 倍**，彻底化解了注意力机制的显存流量危机！

---

## 四、 关键源码逐行解剖

### 1. Warp-per-row Softmax 的组内极速广播

节选自 `01_softmax/softmax.cu` 中对短序列 (Hidden size 较小) 极其有效的处理逻辑：

```cpp
// 🚀 当序列长度较短时，让整整一个 Warp (32人) 全局接管"一行"的 Softmax
float row_max = warp_reduce_max(local_max);

// ⚠️ 经典反直觉：warp_reduce 后，其实只有 0 号线程手里的变量是真实的全局 max！
// 如果就此继续往下算，其他 31 个线程拿的都是局部垃圾值！
// 因此必须使用 __shfl_sync 强制让 0 号线程把宝贝（row_max）喂到大家嘴里。
row_max = __shfl_sync(0xffffffff, row_max, 0);

// 以此类推求分母的和
float row_sum = warp_reduce_sum(local_sum);
row_sum = __shfl_sync(0xffffffff, row_sum, 0);
```

### 2. RoPE Float2 交织重组

节选自 `04_rope/rope.cu` 向量化版本：

```cpp
// RoPE 的公式需要同时动用到位置 2i 和 2i+1 的数据
// 朴素做法是： float x0 = x[2*i]; float x1 = x[2*i+1];
// 这对于访存合并极其不利。正确的硬件解法是强迫指针变构为 float2：
float2* x2 = reinterpret_cast<float2*>(&x[offset]);
float2 val = x2[half_idx]; // 👑 一次内存事务 (8 bytes)，稳准狠抠出两个分量！

float2 rotated;
// 完美的局域寄存器内部解包、旋转
rotated.x = val.x * cos_theta - val.y * sin_theta;
rotated.y = val.x * sin_theta + val.y * cos_theta;
x2[half_idx] = rotated; // 原地写回
```

---

## 五、 性能基准与分析

所有数据提取自 `Results/05_LLM_Ops.md` 真实日志：

- **测试硬件**: NVIDIA GeForce RTX 4090 (sm_89) × 2, Linux 环境
- **测试规模细节**: 严格标定在工业级上下文，Softmax $(128 \times 4096)$、RMSNorm/LayerNorm $(\approx 4000 \text{ hidden_size})$、Flash Attention $(2048 \text{ SeqLen})$.

### 1. 通用向量归一化系列 (Softmax / Norm) 基准

*(注：由于测试张量体积较小（$<35\text{MB}$），GPU 显存调度极快。以下截取部分核心战况)*

| 测试模块 | 实现版本 | Kernel 时间 | 带宽/加速特征 | 硬件简析 |
| -------- | -------- | ----------- | ------------- | -------- |
| **Softmax** | Naive | $0.0053 \text{ ms}$ | $785.19 \text{ GB/s}$ | |
| **Softmax** | **Warp Reduce** | **$0.00 \text{ ms}$** | **$1180.62 \text{ GB/s}$** | 破表级带宽，寄存器蝶步归约消灭了所有的 Shared Mem 读写！ |
| **RMSNorm** | Naive | $0.32 \text{ ms}$ | $212.46 \text{ GB/s}$ | 遇到短行循环产生严重停顿 |
| **RMSNorm** | **Warp RMSNorm** | **$0.00 \text{ ms}$** | **$2620.64 \text{ GB/s}$** | vs CPU 一骑绝尘 **845.9x**。因为极小耗时导致瞬态带宽测算超峰值 |
| **RoPE** | **Float2 Vectorized** | **$0.00 \text{ ms}$** | **$1734.27 \text{ GB/s}$** | 强制 64-bit 内存对齐存取，吃满显存总线引脚通道。 |

### 2. Flash Attention 算力逆袭战 ($Seq=2048, HeadDim=64$)

在处理注意力机制时，核心指标不是绝对时间，而是它能否阻断中间变量 $N \times N$ 所产生的无妄带宽。

| 实现版本 | Kernel 时间 | 中间显存占用 (S矩阵) | 分析结论 |
| -------- | ----------- | -------------------- | -------- |
| CPU 参考 | $6813.06 \text{ ms}$ | $128.00 \text{ MB}$ | 算力受限与灾难级换页 |
| **GPU V1 (Naive)** | $6.60 \text{ ms}$ | **$128.00 \text{ MB}$** | 分为 $QK^T \rightarrow Softmax \rightarrow PV$ 三次启动，不仅来回写入巨大的中间矩阵 $S$，而且大量缓存行处于闲置。 |
| GPU V2 (Flash V1) | $9.58 \text{ ms}$ | **$0 \text{ MB}$** (极简 SRAM) | **经典的反直觉现象**：在较小规模下，由于 Tiling 切块过多且重计算开销存在，绝对时间反而不如 Naive，但它成功消解了 $128\text{MB}$ 的显存毒瘤，使大 Batch 长序列成为可能！ |
| **GPU V3 (Flash Macro-Block)**| **$5.33 \text{ ms}$** | **$0 \text{ MB}$** | **工业之光：**相比于 V1 大幅扩增了单群落 Q 处理数量（复用 K/V），又消灭了显存爆炸，又夺回了性能王座！vs CPU 整体加速比 $1279.17\text{x}$。 |

````mermaid
xychart-beta
  title "Flash Attention 从妥协到压制的 Kernel 时间演变 (ms，越低越好)"
  x-axis ["Naive(含巨型SRAM)", "Flash V1(显存救星但是慢)", "Flash V3(完全体)"]
  y-axis "时间 (ms)" 0 --> 12
  bar [6.60, 9.58, 5.33]
````

---

## 六、 编译及参考资料

### 编译与标准运行指令

借助根目录的统一 `CMakeLists.txt` 构建目标：

```bash
# 1. 切换至项目根目录并执行整体配置（首次构建）
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. 独立编译对应的子项目 Target 
cmake --build build --target softmax -j8
cmake --build build --target layernorm -j8
cmake --build build --target flash_attention -j8
cmake --build build --target rope -j8
cmake --build build --target rmsnorm -j8

# 3. 运行基础验证程序进行观测
./build/05_LLM_Ops/03_flash_attention/flash_attention
./build/05_LLM_Ops/05_rmsnorm/rmsnorm

# 4. 显存排错（极为重要，因为 Flash V3 中有复杂的跨界判定）
compute-sanitizer ./build/05_LLM_Ops/03_flash_attention/flash_attention
```

### 推荐阅读

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) —— 作者 Tri Dao 的开山神作，彻底理解 Tiling 和 Recomputation。
- [One-pass Tensor/Warp Reduce Softmax / LayerNorm](https://arxiv.org/abs/1911.02085) —— Welford 算法在 LLM 中的规范化工业实践原理。
- [RoPE - Rotary Position Embedding (Su et al.)](https://arxiv.org/abs/2104.09864) —— 旋转位置编码的严格数学推演。
