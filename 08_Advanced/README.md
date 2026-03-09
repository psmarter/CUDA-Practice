# 08_Advanced: CUDA 执行图与异步并发架构

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

随着 Kernel 级别的优化到达天花板（如 Shared Mem、Register Tiling），整个程序的执行瓶颈往往会转移到**宿主机（CPU 端）的调度开销**与**物理总线带宽的利用率**上。本章的学习目标是升维到系统级高度，学习如何编排 CUDA 异步流和静态图控制，使得 CPU 驱动层零延迟，PCIe 总线在物理引脚上进行完美的全双工重叠（Overlap）。

目录下的实现涵盖了目前成熟推理引擎和模型系统的底层工程核心：
- `01_cuda_graphs/`：学习如何捕捉（Stream Capture）包含多层迭代循环的 Kernel Launch 闭环，并将其编译成一个全局拓扑执行图一次下发给 GPU，消除高频度 `<<< >>>` 带来的 CPU 发射（Launch）延迟开销。
- `02_multi_stream/`：破除唯一的默认串行流（Stream 0）导致的硬件长期阻塞浪费，引入多个异步 Stream 并结合锁页内存（Pinned Memory），实现“数据拷贝H2D”同时“执行计算”与“结果回传D2H”的完美流水线。
- `03_pytorch_extension/`：演示如何在工业界真正落地 CUDA C++ 代码。通过 libtorch (ATen) 与 pybind11 安全封装这些算子，将其绑定成 Python 侧可以被大模型库直接调用的原生自定义扩展节点并支持前/后向传播。

## 2. 原理推导与数学表达 (Math & Logic)

**Kernel Launch 发射延迟效应**
一次小内核计算总耗时为 $T = T_{launch\_overhead} + T_{compute}$。
如果存在 $N$ 轮相互依赖的连续计算，不用 Graph 的情形下总耗时将被 CPU 限死：
$$ T_{total} = \sum_{i=1}^N \Big(T_{launch} + T_{compute} \Big) $$
当处理的是极大并发的极小负载时（如 LLM 的小型逐词推理或大量极小规模矩阵操作），$T_{launch}$ (约 5微秒左右) 可以远大于 $T_{compute}$ (几百纳秒)。
通过构造 CUDA Graph 并实例化进行重播（Replay），调度发射开销骤降为 $O(1)$ 级：
$$ T_{total\_graph} = 1 \times T_{launch} + \sum_{i=1}^N T_{compute} $$

## 3. 内存与并发映射解析 (Memory & Thread Mapping)

展示多流流水线（Pipeline）在物理总线层面的时序重叠效果：

```text
[单流 Default Stream] 执行过程，PCIe总线与计算单元极度干瘪
| H2D 拷贝 | Kernel 工作 | D2H 拷贝 | 闲置 | H2D_2 |...

[多流 Multi-Stream 且使用锁页内存]
Stream 1:  |  H2D_1  | Kernel_1 | D2H_1 |
Stream 2:            |  H2D_2   | Kernel_2 | D2H_2 |
Stream 3:                       |  H2D_3   | Kernel_3 | D...

================== 【底层物理组件的负荷】 ==================
PCI-e 发送控制 (DMA) | H2D_1 | H2D_2 | H2D_3 |   <--- 饱和
SM 浮点计算核心              | K_1   | K_2   | K_3 |     <--- 饱和
PCI-e 接收控制 (DMA)                 | D2H_1 | D2H_2 |   <--- 饱和
```

**关键限制**：实现多流重叠**绝对依赖** `cudaHostAlloc` 分配的锁页内存（Pinned Memory）。只有内存被锁定不可被系统 Paging（页面交换）时，GPU 的 DMA 硬件控制器才能在无 CPU 的介入下直接穿透 PCIe 发起跨境异步拷贝请求。

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `01_cuda_graphs/cuda_graphs.cu` 中静态图（Topology Graph）的无缝录制和并发构造代码：

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;
cudaStream_t stream;

// 1. 挂钩 Stream 进入 Capture（录制）模式
// 注意：在这之后调用的任何 cudaMemcpyAsync 或 <<<...>>> 都不会立刻被 GPU 执行！
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 发射巨量零碎的算子流水线，仅仅作为图节点 (Nodes) 注册进 Graph 中
for(int i = 0; i < NUM_ITER; i++) {
    kernel_A<<<grid, block, 0, stream>>>(...);
    kernel_B<<<grid, block, 0, stream>>>(...);
}

// 2. 结束录制，返回打包成型的 graph
cudaStreamEndCapture(stream, &graph);

// 3. 实例与优化：此时 CUDA 驱动将全面解析内部各个节点间的依赖边 (Edges)，写入硬件底层
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// 4. 发射执行：只需向硬件提交 instance 指针即可瞬间引爆刚才的成百上千次核心执行
cudaGraphLaunch(instance, stream);
```

## 5. 基准表现与评估剖析 (Performance Data)

在双卡 **RTX 4090** 下评估：

- **CUDA Graphs 发射性能**：
  - 在串联 1000 轮小型算子步骤（涉及 2.67MB 数据）时，常规多次发包的单跑被锁死在极慢的限度里。但 CUDA Graph Launch 单趟执行被压进了 `0.0044 ms` 的极低境地。
  - GPU Graph 使得发射开销全面消失，实测对纯 CPU 多次 Launch 的纯发包**排除了高达 16% 以上**的多余开销浪费。相比 CPU 执行，整个图获得了近 **38x 的微秒级硬件算力纯净加成**。
- **多流重叠 (Multi-Stream Pipeline)**：
  - 测试高达 192MB 的重型数据链路在四个并发流（Stream 0~3）中穿插时，单纯串行 H2D->计算->D2H 的整个任务链路耗时为 `17.72 ms`，且显存传输与核心存在严格串行隔离。
  - 构建多流流水线并发后，管道时钟周期陡然缩减到了 **`13.93 ms`**，并发加速比达到 **`1.27x`**！整个管道带宽拉升到了稳固的 `14.46 GB/s`。成功把显存交互的时间“折叠”到了计算周期中隐藏了起来。
- **PyTorch 自定义算子扩展后端**：
  - `Swish` 自定义激活函数的底层 `ATen` 前后向推导执行成功被绑定，Forward 的显存带宽压满了约 `1019.54 GB/s`（相比 Python 侧获得了 **~368x GPU Kernel 原生加速比**），真正完成了在 PyTorch 的前端调度 Python 图表和底层的 C++ 指令闭环。

## 6. 编译指引参考 (Compile & References)

```bash
cd build && make -j4 cuda_graphs multi_stream pytorch_extension
./08_Advanced/02_multi_stream/multi_stream
```
*注：`pytorch_extension` 若要完整部署为 python 插件需调用根目录下的 `setup.py` 或者使用 `torch::jit::load` 模式构建动态共享库 (.so).*

参考引鉴：
- NVIDIA Developer Blog: [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
- PyTorch 官方指南: [Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
