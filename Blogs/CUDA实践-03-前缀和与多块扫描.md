---
title: CUDA-Practice：03 并行前缀和算法路线选择与端到端扩展
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - Prefix Sum
  - Scan
  - Kogge-Stone
  - Brent-Kung
  - Shared Memory
categories:
  - CUDA-Practice
cover: /img/Nvidia_CUDA_Logo.jpg
abbrlink: bcb510f9
date: 2026-03-12 13:30:00
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
| `03_Scan/01_prefix_sum/prefix_sum.cu` | `kogge_stone_scan`<br>`brent_kung_scan` | 单 Block KS（$\log N$ 步）/ BK（两阶段树），Shared Memory 双缓冲防 RAW | `N=1024` |
| `03_Scan/02_segmented_scan/segmented_scan.cu` | `coarse_scan`<br>`segmented_scan`<br>`add_block_sums` | 单 Block 粗化 + 段末 KS（N≤4096）；多 Block 3-Pass（Pass1 块内 KS+block_sums，Pass2 对 block_sums Scan，Pass3 add_block_sums） | `N=4096`<br>`N=1048576` |

> Kernel 名称与源码中 `__global__` 函数签名完全一致。

> **本篇在系列中的位置**：承接 [02 归约与线程粗化](/posts/44fe4eb3/) 的树形折叠与 Shared Memory 同步，本篇在同一层级做**需保留每一步中间结果**的前缀和（Scan），并推广到多 Block（3-Pass）。后续 [06 线程束原语与寄存器通信](/posts/fec051fc/) 可用 `__shfl_up_sync` 等做 Warp 内无 Shared Memory 的 Scan；[12 标准库与工程实践](/posts/a1e20e80/) 的 Thrust 提供工业级 `inclusive_scan` 可对比本实现。

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

### 单 Block 粗化扫描 (coarse_scan) 与 3-Pass 分工

```cpp
// 来源：03_Scan/02_segmented_scan/segmented_scan.cu : coarse_scan 核心思路（N≤4096 单 Block）
// shared_data[0..n-1] 存数据，section_sums 紧接其后存每段末尾值
    for (int i = 0; i < COARSE_FACTOR; ++i) {
        int index = tid * COARSE_FACTOR + i;
        if (index < n) shared_data[index] = input[index];
    }
    __syncthreads();
    // 段内顺序前缀和
    for (int i = 1; i < COARSE_FACTOR; ++i) {
        int index = tid * COARSE_FACTOR + i;
        if (index < n) shared_data[index] += shared_data[index - 1];
    }
    __syncthreads();
    // 收集每段末尾 → section_sums，再对 section_sums 做 KS，最后把 section_sums[section_id-1] 加回并写 output
```

**要点解读**：

- 当 N=4096 时，单 Block 用 `coarse_scan`：每线程负责 COARSE_FACTOR 个元素，先段内顺序前缀和，再对「段末值」做一次 KS 得到段前缀和，最后加回。避免 3-Pass 的多次 Kernel 与 HBM 往返，实测比同规模下 3-Pass 的 Segmented Scan 略快（约 0.0047 ms vs 0.0059 ms [实测]）。大规模 N=1M 必须用 3-Pass（`segmented_scan` + 对 block_sums 再 Scan + `add_block_sums`）。

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

| 文章 | 与本篇的衔接 |
|------|----------------|
| [02 归约与线程粗化](/posts/44fe4eb3/) | 树形折叠、Shared Memory 与 `__syncthreads` 的用法；本篇在同一套同步与存储层级上做「保留中间结果」的前缀和并扩展到多 Block |

### 推荐后续（承上启下）

| 文章 | 与本篇的衔接 |
|------|----------------|
| [06 线程束原语与寄存器通信](/posts/fec051fc/) | 用 `__shfl_up_sync` 等在 Warp 内做无 Shared Memory 的 Scan，减少 `__syncthreads` 与 Bank 冲突，与本篇 KS 双缓冲形成对照 |
| [12 标准库与工程实践](/posts/a1e20e80/) | Thrust 的 `inclusive_scan` 等可与本实现做正确性与性能对比，理解工业库的封装与优化取舍 |

---

## 顺序导航

- 上一篇：[CUDA实践-02-归约与线程粗化](/posts/44fe4eb3/)
- 下一篇：[CUDA实践-04-矩阵乘优化与寄存器分块](/posts/1a09f6f/)
