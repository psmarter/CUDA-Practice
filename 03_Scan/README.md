# 03_Scan 前缀和与分段扫描

## 一、 全景导览与学习目标

该子项目在 CUDA-Practice 学习体系中属于 **经典算子与并发 (L2)** 阶段。在上一个模块 `02_Reduction` 中我们掌握了将数组折叠为一个标盘的能力，而 **前缀和 (Prefix Sum / Scan)** 则是更具挑战性的操作：它不仅要累加，还必须保留每一个中间状态的累加结果。前缀和是现代 GPU 计算中极其重要的基础构件，它广泛应用于流压缩 (Stream Compaction)、基数排序 (Radix Sort)、以及建树操作等。

本项目深入探讨了经典计算机科学在面临 GPU 海量并行时所产生的两套截然不同的设计哲学：

- `01_prefix_sum/prefix_sum.cu`：**基础前缀和算法**。对比了著名的 **Kogge-Stone**（Step-Efficient，高并发低深度）与 **Brent-Kung**（Work-Efficient，低总运算量高深度）算法。
- `02_segmented_scan/segmented_scan.cu`：**分段与超大规模扫描**。解决了如何通过**多 Block 加跨 Block 块和（Block Sum）**处理超出单个 Block 容纳极限的超大数组，并演示了如何进行高效的局部粗化扫描（Coarse Scan）。

---

## 二、 原理推导与数学表达

**前缀和 (Scan)** 的数学定义：给出一个输入序列 $[x_0, x_1, x_2, \ldots, x_{n-1}]$ 和一个满足结合律的二元操作符 $\oplus$（通常是加法），我们要生成输出序列 $y$：
$$ y_i = \sum_{j=0}^{i} x_j \quad (\text{Inclusive Scan, 包含扫描}) $$
$$ y_i = \sum_{j=0}^{i-1} x_j, \; y_0 = 0 \quad (\text{Exclusive Scan, 不包含扫描}) $$
*(本项目默认基于 Inclusive Scan 进行实现与测试。)*

当并行化时，我们无法像 CPU 那样简单使用 $y_i = y_{i-1} + x_i$ 的串行形式，于是诞生了两种主流策略：

### 1. Kogge-Stone 算法 (Step-Efficient)

每个元素在每一步 $d$ 寻找并加上距离自己 $2^d$ 的历史元素。
$$ y^{(d)}_i = y^{(d-1)}_i + y^{(d-1)}_{i - 2^{d-1}} \quad \text{对于} \; i \ge 2^{d-1} $$
它需要的并行深度仅为 $\log_2(n)$ 步，非常适合 GPU 这种核心数极多、不怕“多干无用功”的架构。但在理论上其总加法次数达到了 $\mathcal{O}(n \log n)$，而非最优的 $\mathcal{O}(n)$。

### 2. Brent-Kung 算法 (Work-Efficient)

分为两个阶段：

- **Up-sweep (归约阶段)**: 与上一章的 Reduction 极为相似，逐步建立出拥有 $\mathcal{O}(n)$ 复杂度的二叉树内部节点和。
- **Down-sweep (分发阶段)**: 树根将总和再逆向散发回所有节点。
这种做法在数学上严丝合缝地把加法运算量卡在了 $\mathcal{O}(n)$，被称为 Work-Efficient（工作量高效）。但它的并行深度达到了 $2\log_2(n) - 2$ 步。

---

## 三、 硬核内存映射解析

我们以对 GPU 最直观友好的 **Kogge-Stone** 为例，看看在 Shared Memory 中数据是如何互相“交织”累加的。

### Kogge-Stone 并行加法时序 (BlockDim = 8)

| 线程 ID (tid) | 初始值 ($d=0$) | Stride=1 ($d=1$) 跨 1 步累加 | Stride=2 ($d=2$) 跨 2 步累加 | Stride=4 ($d=3$) 跨 4 步累加 | 最终结果含义 |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **0** | $x_0$ | $x_0$ (越界不加) | $x_0$ | $x_0$ | $y_0$ (=$x_0$) |
| **1** | $x_1$ | $x_1+x_0$ | $x_{0..1}$ | $x_{0..1}$ | $y_1$ (=$x_{0..1}$) |
| **2** | $x_2$ | $x_2+x_1$ | $x_{1..2}+x_0$ = $x_{0..2}$ | $x_{0..2}$ | $y_2$ (=$x_{0..2}$) |
| **3** | $x_3$ | $x_3+x_2$ | $x_{2..3}+x_{0..1}$ = $x_{0..3}$ | $x_{0..3}$ | $y_3$ (=$x_{0..3}$) |
| **4** | $x_4$ | $x_4+x_3$ | $x_{3..4}+x_{1..2}$ = $x_{1..4}$ | $x_{1..4}+x_{0}$ = $x_{0..4}$ | $y_4$ (=$x_{0..4}$) |
| **5** | $x_5$ | $x_5+x_4$ | $x_{4..5}+x_{2..3}$ = $x_{2..5}$ | $x_{2..5}+x_{0..1}$ = $x_{0..5}$| $y_5$ (=$x_{0..5}$) |
| **6** | $x_6$ | $x_6+x_5$ | $x_{5..6}+x_{3..4}$ = $x_{3..6}$ | $x_{3..6}+x_{0..2}$ = $x_{0..6}$| $y_6$ (=$x_{0..6}$) |
| **7** | $x_7$ | $x_7+x_6$ | $x_{6..7}+x_{4..5}$ = $x_{4..7}$ | $x_{4..7}+x_{0..3}$ = $x_{0..7}$| $y_7$ (=$x_{0..7}$) |

*(注：相比 Reduction 丢弃了一半线程，Kogge-Stone 随着步数加深，参与运算的线程数却越来越多（例如 Stride=4 时，只有 tid 0~3 闲置，4~7 皆在运算），这也解释了为什么它对 GPU 这类 SIMT 架构高度亲和。)*

---

## 四、 关键源码逐行解剖

我们来看在处理 **超大数组 ($>1024$)** 时，`segmented_scan.cu` 是如何通过三遍扫描（Three-pass Scan）将多个孤立的 Block 缝合起来的：

```cpp
// 场景：在 GPU 端管理多个 Block 的扫锚整合 (Host 端调用代码级)
// 1️⃣ 第一遍 (Pass 1)：各自扫各自的门前雪
// 每个 Block (大小 1024) 独立对自己的分配区间进行 KS Scan。
// 亮点：顺手将本 Block 的最末尾一个元素（即本 Block 的总和）抽出来，塞进 d_block_sums 数组！
segmented_scan<<<grid, block, sharedMemSize>>>(d_input, d_output, d_block_sums, n);

if (num_blocks > 1) {
    // 2️⃣ 第二遍 (Pass 2)：扫描 "块的和" 组成的数组
    // 使用单一的 Block，对刚才收集起来的 d_block_sums 再进行一次前缀和扫描！
    // 这产生了 d_scanned_block_sums，即“我这个 Block 前面所有 Block 的总和历史”。
    const dim3 block2(min(num_blocks, BLOCK_SIZE));
    const dim3 grid2(1);
    segmented_scan<<<grid2, block2, sharedMemSize>>>(d_block_sums, d_scanned_block_sums, nullptr, num_blocks);
    
    // 3️⃣ 第三遍 (Pass 3)：天女散花，补齐基数
    // 每个线程回到自己原本的元素上，将属于上一个 Block 产生的巨型基数 d_scanned_block_sums[blockIdx.x - 1] 
    // 平等地加到自己的头上。此时，全局扫描闭环完成！
    add_block_sums<<<grid, block>>>(d_output, d_scanned_block_sums, n);
}
```

**⚠️ 经典的反直觉之处**：
在 Shared Memory 的 Kogge-Stone 算法内部：

```cpp
float val = 0.0f;
if (tid >= stride) { val = shared_data[tid] + shared_data[tid - stride]; }
__syncthreads(); // 必须！读取完毕才能写入

if (tid >= stride) { shared_data[tid] = val; }
__syncthreads(); // 必须！写入完毕才能开启下一次 stride 跨步读取
```

很多新手为了省事直接写 `shared[tid] += shared[tid - stride]` 并只跟一个 sync。这是致命错误！因为后面的线程可能在你前面那个线程还没更新完时就去读了脏数据。**读-算-存必须打断为两截进行屏障隔离**。

---

## 五、 性能基准与分析

所有数据提取自 `Results/03_Scan.md` 真实日志：

- **测试硬件**: NVIDIA GeForce RTX 4090 (sm_89) × 2, Linux 环境
- **测试规模**:
  - `prefix_sum`: 1024 元素 (小规模探测单 Block 极限)
  - `segmented_scan`: $1,048,576$ ($1\text{M}$) 元素，产生 $1024$ 个 Block 多次调度拼接。

### 1. 1024 元素单 Block 内核心算法决战

| 实现版本 | Kernel 时间 | 算法特性 | vs 基准加速比 |
| -------- | ----------- | ---------------- | ------------- |
| CPU 参考 | ~0.00 ms (太小测不准) | $\mathcal{O}(n)$ 串行 | 1x |
| **GPU V1 (Kogge-Stone)** | **0.0028 ms** | $\mathcal{O}(\log n)$ 步, $\mathcal{O}(n\log n)$ 算力 | **基准** |
| GPU V2 (Brent-Kung) | 0.0037 ms | $2\mathcal{O}(\log n)$ 步, $\mathcal{O}(n)$ 算力 | 0.76x |

### 2. 1M 级别跨 Block 巨量规模扫描 (Segmented Scan / 3-Pass)

| 实现版本 | Kernel 时间 | 有效带宽 | vs CPU (1.79ms) 加速比 |
| -------- | ----------- | ---------------- | ------------- |
| CPU 参考 | 1.79 ms | — | 1x |
| **GPU (Segmented Scan 1M)** | **0.0221 ms** | **378.77 GB/s** | **80.69x** |

````mermaid
xychart-beta
  title "Kernel 耗时对比：算法决战与规模扩展爆炸力 (ms, 越低越好)"
  x-axis ["KS (1K)", "BK (1K)", "Seg(1M)"]
  y-axis "时间 (ms)" 0 --> 0.03
  bar [0.0028, 0.0037, 0.0221]
````

**📊 深入分析：**

1. **反直觉的理论与实践背离 (KS vs BK)**：纯数学理论上，Brent-Kung 因其保持了严苛的 $\mathcal{O}(n)$ 工作量而被认为是“优雅”的。但在单块 1024 元素的 RTX 4090 决斗中，多干了无用加法的 **Kogge-Stone 却以 $0.0028\text{ms}$ 击败了 BK 的 $0.0037\text{ms}$**！原因是 KS 算法分支少、并行度更密集（浅深度 $\log n$），完全兑现了 GPU SIMT 对高并行的渴望；而 BK 树状穿梭过程极其容易造成 Warp 发散和同步延迟，导致实际开销反超。
2. **多线程调度之美 (Segmented Scan 1M)**：在处理 1M （1048576个）元素时，数据量膨胀了惊人的 **$256$ 倍**！但得益于 GPU 海量的 SM 并发机制切割了这个问题，Kernel 时间仅仅只增长了可怜的 **$3.78$ 倍**（0.0059ms -> 0.0221ms）。展现出了面对海量数据时近无可撼动的强大缩放（Scale）能力。

---

## 六、 编译及参考资料

### 编译与标准运行指令

借助根目录的统一 `CMakeLists.txt` 构建目标：

```bash
# 1. 切换至项目根目录并执行整体配置（首次构建）
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. 独立编译对应的子项目 Target
cmake --build build --target prefix_sum -j8
cmake --build build --target segmented_scan -j8

# 3. 运行基础验证程序进行观测
./build/03_Scan/01_prefix_sum/prefix_sum
./build/03_Scan/02_segmented_scan/segmented_scan

# 4. 可选测试：借助 compute-sanitizer 检测由于树状越界产生的显存泄漏
compute-sanitizer ./build/03_Scan/02_segmented_scan/segmented_scan
```

### 推荐阅读

- [GPU Gems 3: Chapter 39. Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) —— **(全网目前最经典的 GPU 扫描必读物，详细图解了 Bank Conflict 的 Padding 解法)**
- [Blelloch, G. E. (1990). Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf) —— 追溯 Brent-Kung 骨干基础理论的老派经典论文。
