---
title: "02_Reduction：并行归约的三次进化与带宽极限榨取"
date: 2026-03-10 22:30:00
tags: [CUDA, 高性能计算, Reduce, Warp Divergence, Thread Coarsening, atomicAdd, Shared Memory]
categories: 深度学习系统架构
---

> 📖 **前置阅读**：01_Basics（Shared Memory 基础）  
> 📖 **推荐后续**：06_Warp_Primitives（寄存器级归约）、03_Scan（Prefix Sum）

把 $N$ 个数加起来。听着简单到不值得专门写一篇文章——任何学过编程的人都知道，在 CPU 上跑一个简单的 `for` 循环就能搞定。

但在 GPU 高性能计算的语境下，这个问题变得既迷人又残酷。归约（Reduction）是将一个包含 $N$ 个元素的数组，通过某种满足结合律的二元运算符（如求和、求最大值、逻辑或）折叠成单个标量的操作。它是几乎所有复杂算法（无论是深度学习中的 Softmax 还是物理模拟中的粒子碰撞计算）绕不开的基础原语。

串行计算需要 $N-1$ 步操作。并行计算利用二叉树结构的"两两配对"折叠，理论上只需要 $\log_2 N$ 步。假设你要处理 100 万个元素，串行需要 100 万步，而理想的并行算法只需要区区 20 步。

麻烦在于，GPU 上的"两两配对"有很多种配法。配错了队伍，不仅体现不出 20 步的优雅，反而会因为线程的内耗导致性能惨不忍睹。本文将以真实测试数据为依据，拆解基于 Shared Memory 的归约算法是如何经历从 naive 到专业的关键进化，最后如何通过 Thread Coarsening（线程粗化）去榨干硬件的显存带宽极限的。

---

## 一、小规模归约演进：对抗 Warp Divergence 的战役

我们先剔除跨 Block 同步的复杂性，纯粹考察单个 Block 内部的协作模型。设想一个小规模场景：$N = 2048$。只需要 2 个 1024 线程的 Block 就能包揽。

### Baseline：教科书里的 Simple Reduce

如果要你马上写一个并行归约，你的第一直觉大概率是这样设计的：
第一轮，相隔为 1 的线程结组，偶数索引的线程把相邻的奇数元素加到自己身上；第二轮，跨度拉宽到 2，每 4 个元素由一个线程加和；以此类推，直到跨度达到 Block 的总宽度。

这体现在代码里非常直截了当：

```cpp
__global__ void simple_reduce_sum(float* input, float* output) {
    int i = 2 * threadIdx.x;
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *output = input[0];
}
```

数学与逻辑都没有缺陷，但在硅片上跑起来却是灾难。核心元凶就在那个看似无害的判定条件：`if (threadIdx.x % stride == 0)`。

在 NVIDIA 的硬件架构中，线程并不是各自为战的。它们被死死地捆绑在 32 人一组的作战单位——Warp 里。一个 Warp 内的所有线程共享一个指令提取单元（Instruction Fetch Unit），这意味着这 32 个线程在同一时刻**必须且只能**执行相同的指令。

当 `stride=1` 时，Warp 中编号为 0, 2, 4... 的一半线程满足 `if` 条件去执行加法，而编号为 1, 3, 5... 的另一半线程怎么办？
答案是：它们必须陪跑。硬件会把 `if` 路径和隐含的 `else`（发呆）路径都走一遍，利用内部的谓词掩码（predicate mask）选择性地提交写操作。这种同一 Warp 内线程走入不同分支的现象，就是大名鼎鼎的 **Warp Divergence（Warp 发散）**。
在这个最乐观的 `stride=1` 阶段，SM（流多处理器）的有效管线利用率直接被砍半，只有 50%。

更荒谬的是随着循环推进。当 `stride=16` 时，一个拥有 32 个线程的 Warp 里，只有**两个**线程在计算。另外 30 个线程像看客一样锁定在那里等待。虽然有大量线程还在系统中"存活"，但它们像满天星散落在各个 Warp 里，导致几乎所有的 Warp 都处于低效状态但又霸占着 SM 的寄存器和调度坑位。

### 第一步破局：Convergent Reduce (收敛归约)

既然 Divergence 痛点在于"活跃的线程不连续"，那解法就是强行把它们挤在一起。

我们只需要倒转乾坤，让步长 `stride` 从最大值 `blockDim.x`（其实是一半，因为一开始两两配对只需 N/2 个活体）开始，逐渐折半到 1。

```cpp
__global__ void convergent_reduce_sum(float* input, float* output) {
    int i = threadIdx.x;
    for (int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    // ...
}
```

注意判定条件变成了什么：`if (threadIdx.x < stride)`。

数学本质上，它依然是每轮把剩余元素的数量对折。但物理调度上发生了翻天覆地的变化：
存活执行任务的线程，其编号始终是从 `0` 开始严格连续的。

我们做个推演：

- 当 `stride` 从 1024 减半到 512 时，只有 `threadIdx.x` 从 0 到 511 的线程在干活。
- 因为编号连续，这 512 个线程恰好完美拼满了前面的 16 个 Warp（16 × 32 = 512）。这 16 个 Warp 的利用率是 **100%**。
- 而后面的 16 个 Warp（线程 512-1023）完美地避开了所有计算。调度器侦测到它们全员进入休眠，可以直接将这 16 个 Warp 置于失活状态，不再为它们分配指令周期。
- 当 `stride` 缩小到 32 以下时，全局只剩下一个 Warp 在跑，但也再没有任何 Divergence 发生。

这个小小的调整，一没改变计算次数，二没改变访存次数，却因为顺应了底层调度器的脾气，带来了立竿见影的收益。

实测数据不会说谎（环境：N=2048, RTX 4090, 100次平均）：

- **Simple Reduce** 耗时：0.0051 ms
- **Convergent Reduce** 耗时：**0.0038 ms（1.36 倍提速）**

### 第二步破局：引入 Shared Memory

无论是 Simple 还是 Convergent 版本，都有一个致命缺陷：它们在每一次 `__syncthreads()` 后进行的重组，操作的对象都是 `input` 数组，也就是远在 HBM（高带宽显存）那头的全局内存。哪怕我们解决了发散，数百个周期的访存延迟也会严重拖慢节奏。

所以经典的解决方案是引入 Shared Memory。在 Kernel 启动第一阶段，让每个线程先把数据搬回到 SM 内部极低延迟的 Shared Memory  SRAM 中，然后在这片自家小院里关起门来做树形归约。

为了进一步优化，我们可以在第一次从在全局内存搬砖的时候，顺带让每个线程多抓取一个元素做一次免费的预加和：

```cpp
__global__ void shared_reduce_sum(float* input, float* output) {
    __shared__ float shared_data[BLOCK_SIZE];
    int i = threadIdx.x;
    
    // 第一次加载时立刻做一次折叠，少存一半数据到 SRAM
    // i + BLOCK_SIZE 是一种非常好的合并访存（Coalesced Memory Access）模式
    shared_data[i] = input[i] + input[i + BLOCK_SIZE]; 
    
    // 后面就是同样的 Convergent 逻辑，但是针对 shared_data 操作
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            shared_data[i] += shared_data[i + stride];
        }
    }
    // 写回
}
```

这里有一个违背直觉的实验细节。在笔者相同的 N=2048 实测中，**Shared Memory 版耗时同样是 0.0038 ms，和刚才的 Global Convergent 版打平了。**

为什么引入了高级的 SRAM 却没有拉开差距？答案是 L1 Cache 的自动兜底。2048 个 float 仅仅占据 8 KB，现代的 RTX 4090 拥有巨大的 L1 数据缓存（每个 SM 128KB，动态切分）。当 Convergent 版在全局内存上来回倒腾时，数据早就被死死锁在了 L1 层级，实际并未触发真实的 HBM 往返。
**但这不意味着 SRAM 没有意义，而是战场的规模还不够大。**当我们要归约百万、千万级的大规模张量时，Shared Memory 的价值才会真正爆发。

---

## 二、百万规模实战：Thread Coarsening 的降维打击

现在，我们要把目标变成 $N = 1,048,576$（即 1M 元素）。4MB 的数据不可能塞进任何一个 Block 的 Shared Memory 中。我们来到了多 Block 协作的深水区。

最标准的打法（我们称之为 Segmented Reduce）是将数组切段。1M 元素，每个 Block 消化 2048 个（利用刚才的首轮双发优势，分配 1024 个线程），于是启动 512 个 Block。归约完毕后，这 512 个 Block 算出的 512 个局部总和，统一争抢一条全局的原子锁 `atomicAdd(output, shared_data[0])`。

这个模型很健康，实测 N=1M 下跑出了 0.0084 ms 的成绩。作为物理参照，笔者的 14 核 CPU 跑同一个任务耗时 4.69 ms，GPU 提供了 **559 倍** 的碾压级暴力美学。

但是高级的资深 GPU 工程师不会就此止步。因为 0.0084 ms，根本没有摸到带宽的护城河。他们用来榨干最后几滴油的武器叫做——**Thread Coarsening（线程粗化）**。

其第一性原理的拷问是：既然 SM 算力有盈余，为什么我们要发射多达 512 个 Block 去排队抢调度，并在 Kernel 内部进行多达 10 次的 `__syncthreads()` 同步？为什么不能让一个线程少做同步，多干重活？

```cpp
__global__ void coarsened_reduce_sum(float* input, float* output, int length) {
    __shared__ float shared_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    // 线程读取全局数据的偏移不再是单个 BlockSize，而增加了 COARSE_FACTOR 倍率
    int sid = 2 * COARSE_FACTOR * blockDim.x * blockIdx.x + tid;

    // --- 粗化核心阶段：寄存器级无锁累加 ---
    float sum = 0.0f;
    if (sid < length) {
        sum = input[sid];
        // 连续跨步提取数据，直接累加在私有寄存器变量 `sum` 内部
        for (int i = 1; i < COARSE_FACTOR * 2; ++i) {
            if (sid + i * BLOCK_SIZE < length) {
                sum += input[sid + i * BLOCK_SIZE];
            }
        }
    }
    
    // 把线程独立浓缩好的高纯度数据放入 SRAM，开启同步协作
    shared_data[tid] = sum;
    // ... 后续一模一样的 SRAM 归约 ...
    if(tid == 0) atomicAdd(output, shared_data[0]);
}
```

这段代码看似只是在前面插了一个不起眼的 `for` 循环，但它是物理层面的降维打击。假设 `COARSE_FACTOR = 4`，意味着原本一次只管两个数的线程，现在在遇到第一个栅栏 `__syncthreads()` 之前，先孤独连贯地吞下了 8（即 4×2）个元素：

1. **并行代价锐减**：原本需要发射 512 个 Block，现在所需 Block 数量直接断崖跌落至只有 `512 / 4 = 128` 个。Kernel Launch 和线程块映射调度的固定开销大幅缩水。
2. **免除同步税**：每个线程在这个被解开的内部 `for` 循环里狂做加法时是独狼行动，绝对不需要等待他人的同步信号，节省了巨量时钟滴答。
3. **消除锁竞争瓶颈**：原本有 512 股洪流要去争夺那一个最终的全局 `atomicAdd` 指针口，互相自旋阻塞；现在缩减成了仅 128 个竞争者，锁冲突大大缓解。
4. **提升指令级并行度 (ILP)**：寄存器上的连续累加指令，没有内存访问后置的数据依赖断层，极大地有利于编译器的流水线调度。

这笔账反映在实测数据上惊人地划算（N=1,048,576）：

- **Segmented Reduce** 耗时：0.0084 ms
- **Coarsened Reduce** 耗时：**0.0047 ms （快了 1.77 倍，相比 CPU 加速比拉至极限的 992 倍！）**

对应的有效显存带宽达到了实打实的 **887.48 GB/s**。要记得，加法操作的算术强度（FLOP/Byte）在 GPU 范畴近似为 0，这纯粹是在比拼谁能更快地从显存这口池子里往外抽水。RTX 4090 的标称理论带宽仅约 1008 GB/s，我们凭借着 Thread Coarsening 的手法，已经将硬件极限压榨到了标称值的 88%。

> *这一优化心法并不是孤证：后续我们在研究 GEMM（矩阵乘法）时会遭遇的 Register Tiling，同样是对 Thread Coarsening 思想更为复杂和深邃的映射。*

---

## 三、触及算力天顶：点积变种与指令融合的魔法

归约只是一个数学框架，业务经常要用它求解各种复杂式子。比如典型的内积（Dot Product）：求 $\sum_i (A_i \times B_i)$。
它只比纯 Reduce 多出了一步前置的对应元素乘法而已，我们自然而然地将刚才提炼出的 Thread Coarsening 代码拷贝过来，只需要改写内循环：

```cpp
float sum = 0.0f;
for (int i = 0; i < COARSE_FACTOR * 2; ++i) {
    if (sid + i * BLOCK_SIZE < size) {
        // 先乘，后加到 sum
        sum += a[sid + i * BLOCK_SIZE] * b[sid + i * BLOCK_SIZE];
    }
}
```

针对 1M 对向量的点积测试，在 CPU 上跑出了 1.69 ms，而这版粗化的 GPU 代码只需 **0.0056 ms**（303 倍加速）。但这还没抵达硬件优化的极境。

在现代 NVIDIA 以及大多数现代处理器的 ALU 管线中，加法和乘法常常紧密联系在一起。对于上述代码序列 `T = A * B; Sum = Sum + T`，传统汇编指令流需要两条物理指令。但其实有一个专门面向此场景的魔法指令：**FMA (Fused Multiply-Add)”**。

FMA 直接在硅片物理层面将乘法累加两个独立操作熔在一个执行单元内一并结算。它的计算形式是 $A \times B + C$。最关键的是这**两项操作在硬件层合并只需要消耗一个时钟周期！**并且，由于整个累加过程中没有中间精度的断裂舍入，它不仅更快，甚至比分步算要更加精确。

你大可显式修改代码宣告你的企图：

```cpp
sum = fmaf(a[sid + i * BLOCK_SIZE], b[sid + i * BLOCK_SIZE], sum);
```

那么实测跑出的结果性能是否有本质的不同呢？
**答案是否定的，手写了 FMA 版本后记录下的耗时毫无悬念地也是一模一样的 0.0056 ms**

但这并不是 FMA 这个指令言过其实，而是恰恰反应了当前工业界编译器惊人的进化能力——当我们挂上 `-O3` 给到 nvcc 时，PTX 编译器面对这明文书写的 `sum += a * b;`，早已一眼看穿了程序员的算盘，在底层自作主张替你优化成了融合指令流。我们手写它更多是一种对硬件控制力掌控和代码意图的确立。

**等等，1506 GB/s 的反物理带宽是怎么回事？**
在看点积这几次版本的验证 Log 时，大家必然注意到了极其违和的一点：测算评估出的带宽高达 1506.49 GB/s。难道我们终于通过点积运算把 RTX 4090 的 HBM 总线干穿破理论峰值了？

没这回事！1506 GB/s 是我们踩到了另一项微架构机制：**L2 Cache Hit**。
评估时我们的 $A$ 数组和 $B$ 数组都是 1M 个 float（合各 4MB 大小），共计读取仅要求 8MB 数据。在这个测试脚本的百次迭代评议循环中，第二轮起，这 8MB 数据早已安安稳稳、完完整整地镶嵌在了 RTX 4090 奢侈的 **72 MB L2 缓存**肚子里！

所谓的 1506 GB/s 带宽，是你用这 0.0056 毫秒除去的 8 MB 数据流速，测出的根本不是从几寸开外 GDDR6X 缓慢爬升的带宽，测出的是被极速直供的 L2-SM 总线带宽，这才是造成数据狂飙的真正推手。如果你敢把输入翻到 50M 元素（总规模200MB）撑爆 L2 ，那带宽立马又会跌落回 900 GB/s 出头的踏实水准。

---

## 终极视角：算法与体系结构的协奏曲

如果你顺着全文读到这儿，大概已脱离了"调用个 `cublasSdot` 就好"的思维惰性，真正深入体系结构的底座：

1. **Warp 排布即调度法则**：解决资源竞争不是堆人头，往往让活跃的人站成紧密的队伍（连号）比站得散（跨度交替）能激发出倍增的效能。这是第一板斧——驯服发散（Divergence）。
2. **跨维挤压的通用哲学**：线程粗化（Coarsening）是反直觉却绝妙的思想。不要以为扔给并行的数量越密越好，在一个线程能吃进数据不被噎死（寄存器超售溢出）的前提下，把原本横向铺开的数据在单线程上往纵深累叠，不仅减少了指令调度浪费，还回避了昂贵的内存屏障税。
3. **数据总是在底层发生奇迹**：只有亲眼见过计算吞吐量被缓存效应（Cache Effect）在物理层面上疯狂放大的魔术（L2 超光速），还有当一维运算遭遇融合指令 FMA 发动机带来的增益，你才会明白写 CUDA 很多时候是跟内存物理布局跳交际舞。

**但这远远还没终结**。在这个已经将所有 Shared Memory 发挥到机制上限的程序里，依然残存着一种微小的遗憾——Shared Memory Bank Conflict 依然偶发存在，而且只要走显存放 SRAM 再同步，就是多走了一遭硬件链路。那么，存不存在不借助任何中间件、就在几十个寄存器之间**空中接力、直接劈扣数据**的玩法呢？

下一章 `06_Warp_Primitives`，我们将引入极致的 Warp 寄存器洗牌，那是属于底层调卷专家的下一个猎物。
