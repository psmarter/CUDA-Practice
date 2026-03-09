# 深入 CUDA 优化：05. LLM 级核心算子开发与 Flash Attention 推演

在本系列第五篇博客中，我们正式把目光转向当下炙手可热的 **大语言模型（LLM）算子优化**。本期主要包括：标准的在线 Softmax、LayerNorm/RMSNorm (Welford 算法推演)、旋转位置编码 RoPE（基于浮点数向量化设计）、以及风靡 LLM 界的核心算子——**Flash Attention**。
这些模块对于目前火热部署的 LLaMA、GPT 系列都扮演了最为基础也是最耗时的角色，在模型推理引擎（如 TensorRT-LLM 乃至 vLLM）中，针对上述算子的定制开发可以说是核心竞争力之一。而在我们的两张 **NVIDIA GeForce RTX 4090** 测试机器上，经过高度重构的 C++ 代码展现出了极其惊人的吞吐量与性能潜力。

## 为什么普通 Softmax 和 Attention 都成了显存杀手？

在经典的 Attention 算法中，由于我们要计算 `Softmax(Q * K^T) * V`，如果序列长度 $N$ 很大，`Q * K^T`（即上文得到的得分矩阵）将会生成一个大小达到 $N \times N$ 的中间变量矩阵。这不仅带来了极大的主存存储空间负担，更严重的是我们要从 Global Memory（HBM 或 GDDR）不断读取它、写入它、再读取它算 Softmax、再计算最终输出，这就是所谓 IO 带宽受限（Memory-Bound）。

更头疼的是 Softmax 操作本身的特点：计算中我们必须知道这一手序列的最大值（求 Max），并且还要对整个序列求指数和（求 Sum），最后再反过来为每个元素除以和。也就是传统算法需要扫描三遍（读-最大值、写-指数、读-最后相除）数组。

为了解决这个问题，研究者们设计了 **Online Softmax**，其核心推导如下：
我们在遍历数组时，只保存当前的临时最大值 $m^{(i)}$ 和 临时的指数累和 $\ell^{(i)}$。一旦发现更大的数值 $x_{new} > m^{(i-1)}$，我们不必回头改前面的结果，只需要用一个修正系数差值将老结果按比例 “衰减” 即可：
$$ \text{Scale Factor} = e^{m^{(i-1)} - m^{(i)}} $$
这让 Softmax 的三次遍历直接简化为一次遍历，我们只把寄存器里累计的值乘以这个重构系数，不仅计算精确，还节约了大量的显存回合操作。

## Flash Attention：SRAM 分块策略的颠覆

Flash Attention（闪电注意力机制）正是把 Online Softmax 的魔法用到了矩阵乘法的分块（Tiling）和重计算当中。它聪明地意识到，既然我们可以一块一块地乘出 `Q*K`，那么是不是可以不用把中间结果写回到慢速的 Global Memory？
答案是肯定的，利用 **SRAM（共享内存 Shared Memory）**可以临时驻留每个分块的 $Q_i$ 与 $K_j$，直接在 SRAM 计算出当前局部块的 Attention 分数 $S_{ij}$ 后，立刻应用局部最大值进行 Online Softmax 的 “缩放” 修正。修正完毕之后，我们当场把 $V_j$ 也乘上累计输出里，抛弃掉这个 $N \times N$ 对焦分数的驻留副本。因为从始至终所有的操作都没有“漏”到主显存以外，完全是 `IO-Aware` (IO感知) 的，性能由此爆表！

我们来看一下代码里最重要的核心逻辑：
```cpp
// 在遍历 K 和 V 分块时
float m_ij = compute_local_max(S_ij); // 局部最新极大值
float m_i_new = fmaxf(m_i, m_ij);     // 更新后的全大值

// 计算修正常数，让曾经的历史指数和 l_i 自动"扁平化"衰减
float diff = expf(m_i - m_i_new);
l_i = l_i * diff + l_ij;

// 更神奇的是在 O 的迭代计算上也叠加：
// 历史的 O 数值也同比例缩减，加上最新的基于局部概率加权出的 V
O_i[idx] = diff * O_i[idx] + S_ij * V_j[idx];
m_i = m_i_new;
```
这一巧妙的做法从根本上削去了一大阻力，在我们自己编写的 **Flash Attention V3 Macro-Block + Vectorization** 评估中，对比纯 CPU 版本（`6809.63 ms`），该 CUDA 版本单核执行仅用时 `5.33 ms`（**达到恐怖的 1278 倍加速比！**），其显著表现完全抹除了 $128 \text{MB}$ 临时存储需求。

## 极端访存榨干：RMSNorm 跑到 2600 GB/s！？

另一个必须要提及的组件是 LLaMA 引领的 **RMSNorm**，去掉了传统的 LayerNorm 需要减去均值的开销，只保留了均方根（Root Mean Square）。
在 CUDA Kernel 层级，我们引入了 `Warp-level Reduce` 神器，让每条指令都在 GPU 的内联网（Warp Shuffle 指令：`__shfl_down_sync`）中交互平方和。由于这些操作只生发在同一个 Warp（32个线程）的流处理器中，不需要借由任何外存介入，最终导致了缓存（L1 Cache）带宽利用率的疯狂飙升。

在测试基准（单卡 RTX 4090 带宽峰值约 `1008 GB/s` 的外存带宽理论下）时：
- 我们测算到 RMSNorm **Warp-level 优化版**爆发出了 `2609.55 GB/s` 的等效超值带宽！这不是由于读写穿透了物理硬件上线，而是借助高速的 L1/L2 Cache 击穿且规避了读写回 Global Memory 的物理往还。
- 相较于 CPU 版的 21ms，其仅仅需要 `0.00 ms`（平均核执行仅 0.0X 级别）。

而在 **RoPE** 的实验中，我们将传统的单独拉取浮点数操作改为了 `float2`（甚至是 `float4`）指令向量化并入读。将原本单元素分散的读写请求“打包合并”，让整个 RoPE 实现了类似拷贝代码般的高阶加速，有效访存也压住了 `1724.49 GB/s`。相比起没使用向量化的传统 kernel 又有着实打实的增强效果。

## 小结
大模型（LLM）算子的编写并不是高深莫测的东西，抛开其海量的参数概念后，落到 CUDA 算子实现层：**一切的核心都在于消除不需要的 Global Memory 读写，想尽一切办法在片内（Register/SRAM）做完了事！** 通过分块、重计算、Warp 原语交互以及极致的在线公式变形（Online Softmax），才能让吞吐量得到成千上万倍的回报。
下一步，在完全解锁了硬件极限潜能后，我们即将触碰到更为深入的话题……