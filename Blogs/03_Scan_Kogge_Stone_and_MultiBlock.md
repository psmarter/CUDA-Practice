---
title: "03_Scan：并行前缀和算法路线选择与端到端扩展"
date: 2026-03-12 13:30:00
tags: [CUDA, Prefix Sum, Scan, Kogge-Stone, Brent-Kung, Shared Memory]
categories: 深度学习系统架构
---

## 本文目标

读完本文，你将能够：

- 厘清 Inclusive 与 Exclusive Scan 的数学区别与工程平移关系
- 掌握以高工作量换取极低并发深度的 Kogge-Stone 算法模型（SIMT 最适配打法）
- 分析 Brent-Kung 算法树形完美但在 GPU 发散性与长屏障中的水土不服
- 通过三遍扫描（3-Pass Scan）构架完成百万级别元素的宏观跨 Block 制霸

## 对应代码路径

> **硬件环境**：NVIDIA RTX 4090 (Ada Lovelace, sm_89)
> 128 SMs | FP32 82.6 TFLOPS | HBM 1008 GB/s | L2 72 MB | Roofline 拐点 81.9 FLOP/Byte

| 源文件 | Kernel 名称 | 核心技术 | 测试规模 |
|--------|-------------|----------|----------|
| `03_Scan/01_prefix_sum/prefix_sum.cu` | `kogge_stone_scan_kernel`<br>`brent_kung_scan_kernel` | 单 Block 下极致 Span 与极致 Work 的对撞演练 | `N=1024` |
| `03_Scan/02_segmented_scan/segmented_scan.cu` | `scan_pass1_coarse`<br>`scan_pass2_block_sums`<br>`scan_pass3_add_base` | Thread Coarsening（4K）与百万级 3-Pass 端到端扩展 | `N=4096`<br>`N=1048576` |

> Kernel 名称与源码中 `__global__` 函数签名完全一致。

## Baseline

**问题陈述**：前缀和要求每个第 $i$ 个结果必须强捆绑于第 $0 \dots i-1$ 个过程计算底量，无法通过随意顺序叠加。如果说归约由于无需记录过程轨迹相对自由，前缀和就是对整个张量内存位移图谱极其吃紧的重构操作。它作为后续 Radix Sort 的先行地基。

| Baseline 类别 | 测试场景 | 指标 | 值 | 数据来源 |
|---------------|----------|------|----|----------|
| CPU 参考推演时长 (单核) | `N=1024` (4KB) | 极限速推时 | 0.00 ms | [实测] Results/03_Scan.md |
| CPU 参考推演时长 (14核)| `N=1M` (4.00 MB) | 极化耗时 | 1.79 ms | [实测] Results/03_Scan.md |
| GPU Brent-Kung 推演   | `N=1024` | 单核耗时 | 0.0037 ms | [实测] Results/03_Scan.md |
| GPU Segmented         | `N=1M` | 极大跨块耗时 | 0.0221 ms | [实测] Results/03_Scan.md |

## 瓶颈分析

试图将并行前缀和暴力化，会面临两个难以调和的工程与架构天堑：

1. **跨级深度等待税 (Span 瓶颈)**
   - 并行计算必须兼顾两项标尺：总操作量 (Work) 与并行总深度 (Span)。部分算法像 Brent-Kung 虽然省下了 $\mathcal{O}(n \log n)$ 退致 $\mathcal{O}(n)$ 的工作总量计算器，但是因为要向上爬树归纳再向下遍历分发，Span（跨度）暴拉至接近 $2 \cdot \log N$ 步。这对每一次都要等待全员 `__syncthreads()` 卡死落地的 GPU 并发流水而言是极其拖沓的等待税收。
2. **多 Block 全局同步断崖 (Scalability 瓶颈)**
   - GPU 的 Shared Memory 不能超界同步邻接的 SM Block。在 $N=1,048,576$ 时，1024 个 Block 中，Block $n$ 不先拿到 Block $n-1$ 所有元素的累计极大值时，它是坚决无法正确给自家队伍所有下挂元素加上底层偏移量的，这引发了全域执行壁垒。

## 优化思路

### 优化 1：Kogge-Stone 算法拥抱 SIMT (力砖飞降维)

**解决的瓶颈**：强求低操作总量但拖垮极长并发长度的等待延时。
**核心思想**：彻底放弃算法学家崇尚的 $\mathcal{O}(N)$ “优美”体态，直接将工作量激化至 $\mathcal{O}(N \log N)$ 的 Kogge-Stone（KS）。它的直觉只聚焦于：**如何把 Span（深度）强行压到只有最扁平的 $\log_2 N$ 步**。在每一次循环，所有不越界的线程一律去把相对步长更远的元素抓过来加到自己身上。
**预期收益**：保证每一步没有任何空闲线程发呆，以 24% 的压倒性速度打赢了理论更优但严重 Divergence 停摆的 Brent-Kung [实测]。

### 优化 2：三遍扫描结构打破 Block 孤岛 (Multi-Block Scan)

**解决的瓶颈**：上百万序列跨网段阻拦的全局前置偏置传导。
**核心思想**：建立解耦的三部曲 (3-Pass Scan)：
1. 派发所有 Block 内自己做自己的闭合 KS Scan。在最后退出时，把末尾最大的一块（即本组累计权和）单独投射写死到一个外挂备用小数组 `block_sums` 中去；
2. 拉起一个全新的挂载 Kernel，**只针对这个极小的外挂 `block_sums` 自己单独进行一次宏观扫卷**；
3. **退回原始 Block 端**：第三次启动全域 Kernel，所有的局域 Block 直接拿对应小黑板上的第 `bid - 1` 号刻度基准，直接灌入自身的每一个元素底线中。
**预期收益**：在 $N=1M$ 的大型突击战中稳定拿到 0.0221 毫秒的高阶耗时且保证了强一致性，比 CPU 狂增 **80.69x 加速比** [实测]。

## 关键代码解释

### Kogge-Stone 必须双锁护体

```cpp
// 来源：03_Scan/01_prefix_sum/prefix_sum.cu : L15-L26局部片选简写
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        
        __syncthreads(); // 【护城河 1】确保刚刚大伙全写完了
        
        float val = 0.0f;
        if (tid >= stride) {
            // 所有人都去极具贪婪地抓取远方数据并缓存进局部变量
            val = shared_data[tid] + shared_data[tid - stride];
        }
        
        __syncthreads(); // 【护城河 2】确保大家全都【读取保存】完毕了再动手
        
        if (tid >= stride) {
            shared_data[tid] = val; // 原地下死手覆盖
        }
    }
```

**要点解读**：

- KS 的致命微操在于：整个读取与修改全部砸在同一片物理 Shared Memory 里。因为步数被砍得极低极暴力，各个线程交战区重叠面积极其庞大。假如拔掉中间的 `__syncthreads()`，右方速度爆表的线程会瞬间把刚刚的最新和写回覆盖在左边那帮还甚至没来得及作为基础读取的原本数上 ($RAW \ Hazard$)。双护城河锁与局部私自缓存变量 `val` 是隔离时空冲突的不二之法。

### 工业架构的分段粗化补齐 (Coarsening)

```cpp
// 来源：03_Scan/02_segmented_scan/segmented_scan.cu : scan_pass1_coarse 内
// 在应对不那么长的片段（N=4096），不必切入昂贵耗时三段排队流水，直接让1024人每个吃饱4个
    float thread_sum = 0.0f;
    float local_vals[COARSE_FACTOR]; // 私自寄存器扣押缓冲

    #pragma unroll
    for (int i = 0; i < COARSE_FACTOR; ++i) {
        int idx = tid + i * blockDim.x;
        // 把数据读断进私有并先给自己身上强加一次内部顺序和
        if (idx < length) {
            thread_sum += input[idx]; 
            local_vals[i] = thread_sum; 
        }
    }
    // 把 thread_sum 这个极致的高级摘要抛向 shared_data 然后调用唯一的 Kogge Stone 
```

**要点解读**：

- 当数据量小浮至 4096 这条线，切 `3-Pass` 架构会导致三次跨 HBM 存读唤醒税极不划算。只要直接开挂 `COARSE_FACTOR` 并用纯串行的 `local_vals` 作为自己的一亩三分地吃下冗余，最后只需做极其唯一的一次 `KS` 打平然后反退回即可，该法比前者三核起转还要压下近 30% 耗时。

## 结果与边界

### 性能对比

> **测试条件**：RTX 4090 ($sm\_89$) , 参数 `100 迭代求均`
> **数据来源**：`Results/03_Scan.md` 原始实机日志

**1. 单 Block 内算法硬派决选 (N=1024)**

| 算法模型 | 步长 (Span)| 操作量 (Work) | 测算有效实机时 | 数据性质 |
|----------|------------|---------------|----------------|----------|
| Kogge-Stone | 10 步 | $\approx 10240$ 次加 | **0.0028 ms** | [实测] |
| Brent-Kung  | $\approx 20$ 步 | $\approx 2048$ 次加 | 0.0037 ms (慢 24%) | [实测] |

这反映了一条铁血定局：在拥有成百上千并行引擎的底层怪兽里，你为它少省了极其鸡肋的八千次计算，却为它惹来了接近**翻倍的执行管线同步时断屏障还有前中期极度残破不全的待命分支 (Warp Divergence)**。GPU 只在乎流水压没压满，Kogge-Stone 虽然工作量极其重口，但它能逼迫所有的管线以一往无前的连贯态一直突进致底。

**2. 宏观跨网段大规模攻城战 (N=1,048,576 元素)**

| 测试环境 | Total Kernel 耗时 | 对比基数 | 带宽折现 | 数据性质 |
|----------|-------------------|----------|----------|----------|
| CPU 参考推演 (14核) | 1.79 ms | 1.00x | - | [实测] |
| **GPU Segmented (3-Pass)**| **0.0221 ms** | **狂飙 80.69x 加速** | **378.77 GB/s** | [实测] |

这个 `378 GB/s` 的成绩在 4090 身上的显现是耐人回味的：为何相较于 Reduce 操作可以轻松爆掉 900+ 带宽的事态，Scan 甚至碰触不到四成峰值力道？
原因便是无可脱逃的 `3-Pass` 物理落地律！在这条底线上，算量翻到了 1M，底层强迫要求内核跨 Kernel 切出、必须将这部分流在 HBM 之中深埋硬写，随即再一次启动主进程将这些补偿池逐位提出相注入。**强制全局落盘与提取往复，是前缀和算法注定无法如黑洞般爆吸带宽的核心命门**。但也仅靠此举我们平摊消耗，相对于 $256 \times 4096$ 的规模扩大倍率，其实际时间成本只微涨了 **3.78x**。

### 边界条件与局限

- **强制写回惩罚**：在处理全局大任务时，只要牵扯到了三核扫描拆档打法，哪怕算法再极致，也必须要付那笔中间数据打底到 HBM 的巨额路费，并且还会严重消耗缓存。未来如果要规避掉，唯有通过类似于 `CUB` 这种采用的“解耦预读重写”(Decoupled Look-back) 架构强行去抹灭跨步掉那一次隔离内核层。

## 常见误区

1. **误区**：以为 Inclusive 和 Exclusive 是两套体系代码模型。
   **实际**：绝大多数标准库和底层算法全都一门心思写 Inclusive 版本就够了。想转 Exclusive 怎么办？额外起一个极其轻薄的平移核 `y[i] = y[i-1]` 第一项推给个零就行了！
2. **误区**：在写 `3-Pass` 的内核扫描第三步的时候只需要随便扔给所有的 Block 去提取全局底分。
   **实际**：在第 3 步做加注时，`Block x` 要读取并吃下的永远是前一步里存进 `index: x - 1` 的外挂 `sum` 基线底面（前继包袱量）。并且在 `tid=0` 的老祖宗节点那层，由于自身前面没有任何包袱基站，**绝不能去抽取 -1 的死线越界数据**。这种边角边界处理如果崩掉一点就会牵连数百兆段数据错位。

## 系列导航

### 前置阅读

| 文章 | 关系 |
|------|------|
| [02_Reduction_Tree_Algo_and_Coarsening.md](02_Reduction_Tree_Algo_and_Coarsening.md) | 在 Shared Memory 还未明晰前对树形折返原理操作体系底座的宏观补充 |

### 推荐后续

| 文章 | 关系 |
|------|------|
| [06_Warp_Primitives_Register_Shuffle.md](06_Warp_Primitives_Register_Shuffle.md) | 用寄存器底层的 `__shfl_up_sync` 原理解开最后那二十多下 `__syncthreads` 死锁紧闭墙的神级跳跃，带你了解为什么寄存器级永远没有护城河数据竞态！ |
