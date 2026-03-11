---
title: "05_LLM_Ops：Transformer 核心算子——Softmax, Norm, RoPE 与 FlashAttention"
date: 2026-03-12 12:00:00
tags: [CUDA, 高性能计算, LLM, Softmax, LayerNorm, RMSNorm, RoPE, FlashAttention, Welford算法]
categories: 深度学习系统架构
---

## 本文目标

读完本文，你将能够：

- 理解 Softmax 中 3 遍读写带来的 Memory Bound，并掌握 Online Softmax 的单遍流式计算推导
- 认识灾难性相消对 LayerNorm 的毁灭性打击，掌握 Welford 算法以防止方差精度丢失
- 解析 RMSNorm 砍掉均值归约带来的同步豁免加速比
- 基于算术强度和 SFU 周期，解释超越函数（sin/cos）如何堵塞 RoPE 算力流水线
- 理解 FlashAttention 的 SRAM Tiling 原理，以及 V3 宏块对微观控制流的掩盖

## 对应代码路径

> **硬件环境**：NVIDIA RTX 4090 (Ada Lovelace, sm_89)
> 128 SMs | FP32 82.6 TFLOPS | HBM 1008 GB/s | L2 72 MB | Roofline 拐点 81.9 FLOP/Byte

| 源文件 | Kernel 名称 | 核心技术 | 测试规模 |
|--------|-------------|----------|----------|
| `05_LLM_Ops/01_softmax/softmax.cu` | `warp_reduce_softmax` | Online Softmax / 二叉树规约 | `Seq=4096`<br>`Batch=128` |
| `05_LLM_Ops/02_layernorm/layernorm.cu` | `layer_norm_welford` | Welford 在线方差递推 | `Hidden=4096`<br>`Batch=128` |
| `05_LLM_Ops/05_rmsnorm/rmsnorm.cu` | `rmsnorm_warp` | 砍去均值缩放 / Warp Shuffle | `Hidden=4096`<br>`Seq=2048` |
| `05_LLM_Ops/04_rope/rope.cu` | `rope_vectorized` | `float2` 向量化加载 / RoPE | `Dim=128`<br>`Seq=2048` |
| `05_LLM_Ops/03_flash_attention/flash_attention.cu` | `flash_attention_v3` | SRAM Tiling / 宏分块掩盖控制流 | `BR=32, BC=32`<br>`Seq=2048` |

> Kernel 名称与源码中 `__global__` 函数签名完全一致。

## Baseline

**问题陈述**：在千亿参数大模型推理中，占据运算量 90% 以上的并非全是矩阵乘法，非线性激活、位置编码、归一化以及注意力机制的 $S = QK^T$ 严重拖垮了总带宽。这些层通常具备极其轻微的计算密度（$I \ll 81.9\text{ FLOP/Byte}$），极易撞死在物理显存墙上。
我们以这些算子的原版教科书式数学实现作为基准线，以评估优化手段对吞吐带宽的榨取率。

| Baseline 类别 | 测试场景 | 指标 | 值 | 数据来源 |
|---------------|----------|------|----|----------|
| Naive Softmax (3遍扫描) | `Seq=4096` | 运算执行耗时 | 0.0053 ms | [实测] Results/05_LLM_Ops.md |
| Naive LayerNorm (分离求值) | `Hidden=4096` | 有效显存吞吐 | 644.72 GB/s | [实测] Results/05_LLM_Ops.md |
| Vectorized RoPE (float2) | `Seq=2048` | vs Naive 加速比 | 1.03x | [实测] Results/05_LLM_Ops.md |
| Naive Attention ($N^2$ 落盘) | `Seq=2048, Head=4` | 中间态 HBM 体积| 128.00 MB | [理论] |
| Naive Attention ($N^2$ 裸算) | `Seq=2048, Head=4` | 执行推测耗时 | 6.60 ms | [实测] Results/05_LLM_Ops.md |

## 瓶颈分析

切开标准 Transformer 模块，各种非线性微型算子的拥堵成因可解构为以下四点：

1. **Softmax 的多遍扫描阻断 (Memory Bound)**
   - $\text{Softmax}$ 为防止数值溢出需 $x_i - \max(x)$。朴素做法必须完整读取第一遍找最大值、第二遍求 $e^x$ 分母和、第三遍做除法写回。这种 $3\text{ Read} + 1\text{ Write}$ 的模式导致其算术强度 $I \approx 0.31\text{ FLOPs/Byte}$ [理论]，严重依赖 HBM 带宽。
2. **LayerNorm 方差精度的灾难性相消 (Numerics)**
   - $\sigma^2 = E(x^2) - (E(x))^2$ 在单遍遍历中如果 $x$ 基数极大而波动极小，FP32 仅 23 位的尾数会在大数相减中把真正的微小方差生生抹平（截断误差）。
3. **RoPE 超越函数的流水线阻塞 (Compute Bound 特例)**
   - 虽然使用了 `float2` 一次读取 64-bit 彻底喂饱了 LSU，但 $\sin \theta, \cos \theta$ 属于超越函数。CUDA 核心需几十个极慢的特殊功能单元（SFU）周期去用多项式甚至查表逼近。运算管线的极长等待硬性接管了整体耗时。
4. **Attention 分数矩阵的全盘物化炸弹 (Capacity & Bandwidth)**
   - $S = QK^T$ 所产生的矩阵面积呈 $\mathcal{O}(N^2)$ 级。在 $N=2048$ 时，产生的高达 $128 \text{ MB}$ 临时显存（$Q \cdot K$）不但立刻耗尽 SRAM 空间，还会导致向外 HBM 倾泻写入后再行读入激活，让显存来回颠簸（Thrashing）。

## 优化思路

针对各个算子的致命瓶颈，工业界给出了以下标准手术切除：

### 优化 1：Online Softmax 动态修正重标

**解决的瓶颈**：必须提前锁定全局最大值引起的强制多遍内存扫描。
**核心思想**：只做单遍循环扫描流！我们在寄存器内置状态机 $m_{old}, d_{old}$。当新传入未知数据 $x_k$ 发现更大的 $m_{new}$ 时，强行将以往积淀的所有旧发力分母项乘以补偿衰减系数 $e^{m_{old} - m_{new}}$ 实施全局折旧。
**预期收益**：成功将全局内存读取频次硬砍至 1 遍并保持精度的绝对数学一致。

### 优化 2：Welford 递推与 RMSNorm 摘除

**解决的瓶颈**：方差失真与大批量 `__syncthreads()` 卡口。
**核心思想**：LayerNorm 使用单刀直入追踪差分 $\Delta_k = x_k - \mu_{k-1}$ 更新均值的 Welford 在线方程来规避平方项相减炸膛。
其次，对于更普遍的场景：干脆暴力抛弃原本的均值归算动作，只余留纯粹向后的均方根乘方归一，成为 **RMSNorm**。它摘除了一场全员对齐等待计算平均值的强制屏障。
**预期收益**：将 LayerNorm 提速 7%，并将简化版的 RMSNorm 通过 256并发与蝶形网络打满至几十微秒级 [实测]。

### 优化 3：FlashAttention 宏块 SRAM Tiling (V3)

**解决的瓶颈**：千万级 Token 的 $N^2$ 无底限外显存物化。
**核心思想**：彻底推翻大矩阵落地法则。仅仅撕下一微型窗口的 $Q \ (B_R)$ 和 $K, V \ (B_C)$ 输入到极小的高速 SRAM ($< 100 \text{ KB}$)内，依靠 Online Softmax 在里面原位消化掉所有的积和修正后立刻吐出最终解。进一步选用极限超大 Tile 块尺幅以摊薄掉密密麻麻的内核边界循环控制判断周期。
**预期收益**：完全摧毁 $128 \text{ MB}$ 中继文件，反杀获得最高至数十数百倍的高效计算提速 [实测]。

## 关键代码解释

### State Machine 驱动下的 Online Softmax

```cpp
// 来源：05_LLM_Ops/01_softmax/softmax.cu : 局部片选简写
    float m_old = -INFINITY;
    float d_old = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float x_k = input[i];
        
        // [1] 不断探取前方高能，动态攫取新的霸主峰值
        float m_new = max(m_old, x_k);
        
        // [2] 这一条核爆级公式：将以前全部算出来积压在手上的分母权重
        // 用最新的最大值与旧最大值的落差 exp 强行等比剥落拉平！
        d_old = d_old * expf(m_old - m_new) + expf(x_k - m_new);
        
        m_old = m_new;
    }
```

**要点解读**：

- `[1]-[2]` 彻底终结了 Softmax 的两阶段定律。我们完全可以在不断吸收未知边界数据时，通过 `expf(m_old - m_new)` 的极其精美的常系数惩罚，让早先算错偏差的求和基底乖乖地缩回它该在的位置上。这种剥落机制是后面 Flash Attention 中重计算能生效的最重要理论基石。

### Welford 方差抗相消计算

```cpp
// 来源：05_LLM_Ops/02_layernorm/layernorm.cu : L28-L38
    float mu = 0.0f, m2 = 0.0f, count = 0.0f;
    for(int i = tid; i < hidden; i += blockDim.x) {
        float val = input[row * hidden + i];
        count += 1.0f;
        
        // [1] 从来不直接将所有数莽干加在一起，永远只研究当前值到基线的“摇摆偏度” (delta)
        float delta = val - mu;
        mu += delta / count;         
        // [2] 更新平方和累加库，注意连乘的偏度一个是基于老的均值，一个是除以权证后的新均值
        m2 += delta * (val - mu);    
    }
```

**要点解读**：

- `[1]-[2]` 这个看似晦涩的方程叫 Welford 增量更新法。就算你的自然输入流里充斥着上百万数值的宏大漂浮（例如 LLM 后期极大激活的 Outliers），$val$ 减去刚更新的 $\mu$ 时剥离出来的增量（$\Delta$）永远很扁平微弱安全，将浮点精度的末尾几位极其完美的保留了下来。

## 结果与边界

### 性能对比

> **测试条件**：双 RTX 4090 ($sm\_89$), nvcc -O3
> **数据来源**：`Results/05_LLM_Ops.md` 原始实机日志，均以 50-100 次打脸求均值避开冷启动

**1. Softmax 通道扫荡对绝**

*对于规模 `Seq=4096, Batch=128` (全尺寸2MB)*

| 实现手段 | 运算执行耗时 | 带宽榨取率 | 加速对标 | 数据性质 |
|----------|------------|----------|----------|----------|
| Naive 多阶段共享内扫描 | 0.0053 ms | 785.19 GB/s | 1.00x | [实测] |
| Online 递减归并法 | 0.0041 ms | - | 1.30x | [实测] |
| Warp 原语蝶形连打归约 | **0.0035 ms** | **1180.62 GB/s** | **1.50x** | [实测] |

在线衰推在无损下砍出了 1.3 倍的绝对提速。至于最终版为什么能冲破物理天际达 1180 GB/s？这是由于 72MB 庞大的 L2 缓存对于只有 2MB 体量测试的高命中缓冲假象，但这无不例证我们成功把代码榨干直至把 GPU 的晶体管管线给逼迫到绝境。

**2. LayerNorm 与 RMSNorm (Hidden=4096)**

| Kernel | 执行耗时 | 有效物理吞吐带宽 | 数据性质 |
|----------|------------|------------------|----------|
| Naive LayerNorm | 0.0065 ms | 644.72 GB/s | [实测] |
| Welford 精准不丢版 | **0.0061 ms** | **691.89 GB/s** | [实测] |
| Naive RMSNorm | 0.32 ms | 212.46 GB/s | [实测] |
| **Warp-level RMSNorm**| **0.026 ms** | **2620.64 GB/s** | [实测] (含 L2 极度增益) |

抛除均值的等候墙使得极其惨烈的 12.33 倍断档杀伤加速比于同等规模上在 RMSNorm 间上演 [实测]。

**3. Flash Attention 的惊天时空大碰撞**

*对于微型规模 `Seq=2048, HeadDim=64, BR=32, BC=32`* 

| 执行阶段版本 | HBM 中存盘体积 | 核上纯累加时 | CPU 基准倍杀 | 数据性质 |
|-------------|--------------|--------------|--------------|----------|
| Naive $N^2$ 落盘暴力流 | **128.00 MB** | 6.60 ms | - | [实测] |
| Flash V1 (细散砖切块) | 仅缓存 O | 9.58 ms | 惨降 | [实测] |
| Flash V3 (Macro 切配+Float4) | 0.00 MB | **5.33 ms** | 1279.17x | [实测] |

为什么最初版比原版还慢？因为在 $2048$ 这个还未膨胀的超短句段下，$32\times 32$ 极其细碎严密的控制流（循环判断锁及各种同步界线）死死拖垮了运算器流水线掩护了极短周期的带宽补足。将 Tile 快幅扩充至极值并利用粗管抽水 (Flash V3)，流水线终于得以疏导并正式掀翻 Native 原生矩阵体系落盘霸权。

### 边界条件与局限

- **超越函数的硬伤**：利用 `float2` 吸入参数仅给 RoPE 带来了可悲的 **1.03×** 提升 [实测]。这也揭露了一个极其无情的算力真相——当 $sin$ 等三角算子因为依赖几十周期的 SFU 指令周期彻底将调度队列堵塞发烫时；在后段给再粗放多少条高速路数据带入也只不过杯水车薪。只能通过粗鲁截断降级的 LUT 查询表来做有损更换。

## 常见误区

1. **误区**：一旦出现算出来的概率极高为 `NaN`（Not a Number），就是因为显存爆了。
   **实际**：这极有可能是底层没有插入 `val - max_v` 或者是没有去上用 Welford 防灾保护算法。FP16 和 FP32 的阶码浮位非常可怜，哪怕一个极其微弱的 `exp(100)` 早已在显存上将其彻底撕裂开溢崩盘。
2. **误区**：Flash Attention 在所有场景之下都神挡杀神。
   **实际**：在序列极其短小的首包输入期间或者是你长了极其庞大的高速 L2 分担墙时，它的超低吞吐收益将会悉数被内部极度繁琐沉重且要重复计算的 $SRAM$ 内状态机推算墙掩埋从而呈现被原版 Native 手段极尽全方位屠戮乃至碾压变慢的尴尬现象。这也是它只专供处理超级长难下文及极大模型体面的根源机制。

## 系列导航

### 前置阅读

| 文章 | 关系 |
|------|------|
| [02_Reduction_Tree_Algo_and_Coarsening.md](02_Reduction_Tree_Algo_and_Coarsening.md) | 在处理层归一或 Attention 并行前必读的底层蝶形打法逻辑 |
| [06_Warp_Primitives_Register_Shuffle.md](06_Warp_Primitives_Register_Shuffle.md) | 获取了解 12 倍加速的 RMSNorm 内含有的 `__shfl_down_sync` 寄存器越界通信法器 |

### 推荐后续

| 文章 | 关系 |
|------|------|
| [11_Inference_Optimization_Fusion_KVCache.md](11_Inference_Optimization_Fusion_KVCache.md) | 去认识到以上所有的绝学拼凑打通入到大语言模型的整周期推理中到底如何降低死算发热 |
