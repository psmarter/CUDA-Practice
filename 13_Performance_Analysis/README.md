# 13_Performance_Analysis: NCU/NSYS 深度剖析与瓶颈定位

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

不善用剖析器（Profiler）的程序员只能在黑箱外猜想，“测量胜于直觉”，在底层硬件优化上更是绝对的真理。本章的学习目标是掌握 NVIDIA 第一方御用两大性能杀器：Nsight Systems (nsys) 与 Nsight Compute (ncu)。通过学习分析屋顶线模型（Roofline Model）、SM 驻留率（Occupancy）等关键指标，精确锁定自己的 Kernel 究竟是“跑太慢”还是“饿死在等待通道上”。

目录下的子模块通过不同维度展示了性能度量的基本原理：

- `01_occupancy/`：提供一个可调变 Shared Memory 与寄存器占用量的示例，用来展示 SM 的驻留率（Occupancy）是如何被某一个超越阈值的硬件资源所瞬间卡死拖累的。
- `02_roofline/`：使用算数强度可调的 Kernel 函数作为标本，通过 Nsight Compute 实绘制并体会经典的屋顶线模型（Compute vs Memory Bound）。
- `03_nsight_profiling/`：包含两张使用 `ncu` 和 `nsys` 具体分析的截图与对应的代码标本，介绍如何直观地阅读瀑布流时序图与源级汇编。

## 2. 原理推导与数学表达 (Math & Logic)

**屋顶线模型 (Roofline Model)** 的数学表达：
横轴：算数强度（Arithmetic Intensity）$I$ = (以 FLOPs 计的总计算量) / (以 Byte 计的总内存流量)
纵轴：内核算力性能 $P$ (FLOPs/second)

理论峰值计算力定格为平坦的屋顶：$P_{peak}$
而受限于内存带宽（带宽带斜率 $BW_{peak}$）的左斜坡极限方程式为：
$$ P = I \times BW_{peak} $$

内核的最终成就速度只能取两者的边界下限最小值：
$$ P_{achieved} = \min(P_{peak}, \; I \times BW_{peak}) $$
一旦计算落在左斜区，这就是极其绝望的存在：说明你在花极大的代价等待显存，算力核心实际上是在闲置中打盹。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

以 Occupancy 为例，展示有限的物理槽位对并行的硬约束。
假设每个 SM 只有 100KB 容量的 L1/Shared Memory 总槽位，且单个 SM 最多能容纳 1024 个活跃 Thread（比如分布在 4 个 Block 中，每 Block 256 人）：

```text
[SM 硬件限制] 最大并发 Block 数量被两方钳制：
资源1: Thread 数 (上限 1024)
资源2: Shared Mem 容量 (上限 100KB)

情况 A (好实现): 
    每个 Block(256Thread) 只申请 20KB Shared Mem
    ==> SM 刚好挤满 4 个 Block (4x256=1024, 4x20K=80K)。Occupancy=100%
    如果某个 Block 发成访存阻塞，立刻就有其他排队的 Block 换入！

情况 B (差实现):
    每个 Block(256Thread) 粗心多申请了一点点，变为 35KB
    ==> SM 最多只能塞进 2 个 Block ！(因为 3x35K > 100K 物理炸了)
    ==> 该 SM 里只有 512 个 Thread 在活跃。Occupancy 暴跌至 50%！
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `01_occupancy/occupancy.cu`，通过 API 运行时探测最大承力点：

```cpp
// 并不是开满 Block 就能跑满显卡。我们需要探查编译器给我们用了多少寄存器
int numBlocks;
int blockSize = 256; 
size_t dynamicSMemSize = 0; // 这将受到你使用 malloc 或者 shared 取决

// ✨ CUDA runtime 提供了极其硬核的占据率测算黑魔法 API
// 它会根据你这套代码编译时的「寄存器消耗量」和「静态共享内存消耗量」
// 告诉你在该硬件上，一个 SM 最高能并发容纳几个这样的 Block
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, test_kernel, blockSize, dynamicSMemSize);

printf("单 SM 最大可并发承载该 Kernel 的 Block 数: %d\n", numBlocks);
```

## 5. 性能基准与分析视角 (Performance & Profiling)

- **基准目标**：找出瓶颈到底在 Compute, Memory Bandwidth 还是 Latency 侧。
- **典型分析**：`nsys` 给出的全局时间轴线能瞬间发现 API 发射之间的气泡与 PCI-E 异常拥堵。而 `ncu` 主控微观：它提供的 `Memory Workload Analysis` 和 `Source Counters`，能精准地在一个源代码句子旁边标注：此处触发了由于 L2 Cache 未命中导致的巨量长挂起（Stall Long Score）。

## 6. 编译指引与参考资料 (Compile & References)

```bash
# 务必挂上 -lineinfo 标签，否则 NCU 无法把汇编映射回原始 C++ 源码
nvcc -O3 -arch=sm_89 -lineinfo roofline.cu -o run_roofline
# 获取极为详尽带分析建议的命令行性能报告
ncu --set full ./run_roofline
```

- 参考资料: NVIDIA Nsight Compute User Interface Manual / Roofline Performance Model。
