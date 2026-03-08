# 08_Advanced: CUDA 执行图与异步并发架构

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

随着 Kernel 级别的优化到达天花板（如 Shared Mem、Register Tiling 耗尽极限），整个程序的瓶颈往往会转移到**宿主机（CPU 端）的调度开销**与**物理总线的利用率**上。本章的学习目标是升维到系统级高度，学习如何编排 CUDA 异步流和计算图，使得 CPU 驱动层不再拖后腿，PCIe 总线不再单向空载。

目录下的实现涵盖系统编排能力：

- `01_cuda_graphs/`：学习如何捕捉（Capture）一段包含成百上千次 Kernel Launch 的闭环，并将其编译成一个执行图，在一瞬间全部下发给 GPU，消除由于每次调用 `<<< >>>` 带来的 CPU 发射延迟。
- `02_multi_stream/`：在默认单流（Stream 0）会导致一切按顺序串行阻塞的背景下，引入自定义 Stream 并结合 Pinned Memory（锁页内存），实现 H2D传输、Kernel计算、D2H回收的完美重叠流水线。
- `03_pytorch_extension/`：演示工程落地的最后一步，如何将裸写的 CUDA C++ 代码使用 ATen 和 pybind11 安全封装并动态打入 Python 侧，给 PyTorch 增加自定义高性能算符。

## 2. 原理推导与数学表达 (Math & Logic)

以多次小计算发射延迟（Launch Latency）为数学化概括：
一次内核的实际耗时 $T = T_{launch\_overhead} + T_{compute}$。
如果存在 $N$ 轮序列计算，未使用 Graph 的总耗时将是 $\sum (T_{launch} + T_{compute})$。
对极小量级的 Kernel 而言（即 $T_{launch} \approx T_{compute}$ 甚至更大），
通过构造 CUDA Graph 并重播（Replay），耗时骤降为 $1 \times T_{launch} + \sum T_{compute}$。理论上，调度开销由 $O(N)$ 降至 $O(1)$ 级。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

展示并发流水线（Pipeline）在物理总线层面的时序重叠效果：

```text
[单流 Default Stream] 执行过程，总线利用极度不饱和
| H2D (Memcpy) | Kernel | D2H (Memcpy) | H2D | Kernel | ...
(一条干线在任何时刻都只有一部分部件在运作)

[多流 Multi-Stream 且使用锁页内存 Pinned Memory]
Stream 1:  |  H2D_1  | Kernel_1 | D2H_1 |
Stream 2:            |  H2D_2   | Kernel_2 | D2H_2 |
Stream 3:                       |  H2D_3   | Kernel_3 | D...

硬件层面：
PCIe 发送总线   | H2D_1 | H2D_2 | H2D_3 |   <--- 满载
计算引擎(SM)            | K_1   | K_2   | K_3 |     <--- 满载
PCIe 接收总线                   | D2H_1 | D2H_2 |   <--- 满载
```

**关键限制**：该重叠必须依赖 `cudaHostAlloc` 分配的锁页内存，只有操作系统保证内存不被交换（Pageable）时，DMA 控制器才能无 CPU 介入地跨总线异步拉取。

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `01_cuda_graphs/cuda_graphs.cu` 中对静态执行拓扑图的构造和发射：

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;
cudaStream_t stream;
cudaStreamCreate(&stream);

// 1. 开始录制：进入到捕捉范围内，此时所有的 Launch 都不真正执行，而是变成图中的 Node
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

for(int i = 0; i < NUM_ITER; i++){
    kernel<<<grid, block, 0, stream>>>(...);
}

// 2. 结束录制，返回拓扑图对象
cudaStreamEndCapture(stream, &graph);

// 3. ⚠️ 实例化图：在这个阶段 CUDA Driver 会极度优化图内部节点的依赖关系并写入底层
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// 4. 以极低的 CPU 开销秒发整个巨型循环网络
cudaGraphLaunch(instance, stream);
```

## 5. 性能基准与分析视角 (Performance & Profiling)

- **基准**：对比使用默认 Stream 串行执行，和使用多次普通 `<<<>>>` 发射执行这两种经典未优化场景。
- **典型分析**：使用 Nsight Systems (`nsys`) 而不是 Nsight Compute。你会从长时钟序列表上清晰地看到，原本间隔巨大的空白（由 CPU 开销引起）被瞬间压扁。而在多流测试中，物理层面上代表 H2D 与 D2H 的橙色与绿色条形域会产生美妙的时序错位交织。

## 6. 编译指引与参考资料 (Compile & References)

```bash
nvcc -O3 -arch=sm_89 multi_stream.cu -o run_streams
# 对于调度瓶颈的分析，必须使用 nsys 抓取带时间轴的追踪文件 (qdrep)
nsys profile -t cuda,osrt,nvtx -o my_profile_report ./run_streams
```

- 参考资料: NVIDIA Developer Blog: "Getting Started with CUDA Graphs".
