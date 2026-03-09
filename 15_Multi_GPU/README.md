# 15_Multi_GPU 多卡分布式与集合通信

## 一、 全景导览与学习目标

该子项目处于 CUDA-Practice 学习体系的 **高阶系统级 (L4)** 阶段，同时也是整个项目通关的**最终大轴**。现代深度学习（特别是大跨度 LLM 模型）早已不可避免地突破了单卡显存的物理限制（例如一块 RTX 4090 仅有 24GB VRAM），多卡分布式并行 (Distributed Data Parallel / Tensor Parallel) 成为了终极出路。

在这个阶段，单卡的微观指令掩盖已经退居次位，真正的性能瓶颈转移到了**节点间与 GPU 之间那缓慢的 PCIe/NVLink 互联带宽上**。

- `01_nccl_allreduce`：**NCCL (NVIDIA Collective Communication Library) 入门**。通过工业界最高级别的多卡通信标准组件 NCCL，演示如何在两块独立物理显卡之间，跨越设备边界完成高效的张量梯度规约求和 (AllReduce)。

---

## 二、 原理推导与数学表达

### 1. 集合通信的数学定义 (Collective Communication)

以分布式训练中的数据并行 (DP) 为例，在做反向传播后，需要将位于不同 GPU 上的梯度进行汇总，然后再分发给所有卡更新参数。这正是 **AllReduce** 操作的用武之地。

假设有 $N$ 张 GPU 设备，将其逻辑编号为 Rank $0, 1, \dots, N-1$。每张卡持有长度为 $M$ 的局部数据向量 $X_i$。
执行 `ncclAllReduce` 并指定归约算子为 `ncclSum` 后，所有卡的输出接收缓冲区将获得统一的全局归约结果 $Y$：
$$ Y = \sum_{i=0}^{N-1} X_i $$
这意味着对于任意索引 $j \in [0, M-1]$：
$$ Y[j] = X_0[j] + X_1[j] + \dots + X_{N-1}[j] $$

### 2. 底层图论拓扑优化 (Ring / Tree 算法)

传统的规约方法可能采用“单点收集 (Gather)”：所有卡把数据发给 GPU 0，GPU 0 算出总和后再广播 (Broadcast) 给所有人。这会让 GPU 0 的网络带宽瞬间成为雪崩瓶颈（$O(N)$ 的带宽压力卡死在单一节点）。
NCCL 在内网调度上采用了 **Ring-AllReduce (环形规约)** 或 **Tree-AllReduce (树形规约)**：
在 Ring-AllReduce 下，数据被拆分成 $N$ 份切片，每张卡同时向右侧邻居发送不同切片的数据，并同时接收来自左侧邻居的数据。每一轮只需传输 $\frac{M}{N}$ 的数据量。经过 $2(N-1)$ 轮后完美归约，极大地逼近了点对点互联链路的理论上限。

---

## 三、 硬核内存映射解析

### 跨设备异构控制同步流与 P2P 拓扑图

利用单台宿主机（Host CPU）统一统筹和控制 2 张独立 GPU 之间的高速显存交互机制：

```mermaid
graph TD
    classDef HostGroup fill:#fdf4ff,stroke:#d946ef;
    classDef GpuGroup fill:#f0f9ff,stroke:#0ea5e9;

    subgraph "宿主机 主板端系统"
        CPU["Host Main Thread (单线程控制中心)"]:::HostGroup
    end

    subgraph "PCIe 主板槽 1"
        GPU0["GPU 0 (Rank 0) \n d_sendbuffs[0]"]:::GpuGroup
        Stream0["cudaStream_t[0] \n (GPU 0 的专属执行流)"]:::GpuGroup
        GPU0_Out["GPU 0 \n d_recvbuffs[0]"]:::GpuGroup
    end

    subgraph "PCIe 主板槽 2"
        GPU1["GPU 1 (Rank 1) \n d_sendbuffs[1]"]:::GpuGroup
        Stream1["cudaStream_t[1] \n (GPU 1 的专属执行流)"]:::GpuGroup
        GPU1_Out["GPU 1 \n d_recvbuffs[1]"]:::GpuGroup
    end

    CPU -- "ncclGroupStart() 发送屏障" -.-> Stream0
    CPU -- "ncclAllReduce 组派发" -.-> Stream1
    
    GPU0 --> Stream0
    GPU1 --> Stream1
    
    Stream0 <--"NCCL 底层隐式调用 P2P \n (通过 PCIe Gen4 / NVLink)"--> Stream1
    
    Stream0 --> GPU0_Out
    Stream1 --> GPU1_Out
```

> **解析**：在传统的 CUDA 中，`cudaMemcpy` 只能解决 Host 与 Device 之间的数据搬运。要跨卡传输，你本需要用 `cudaMemcpyPeer` 进行手工显存复制，甚至需要手动拉管线去计算求和逻辑。而 NCCL 将这段图里底部最为复杂的 `P2P 同步` + `Ring 归约计算` + `分发回填` 完美打包，开发者只需面对抽象级别的 API。

---

## 四、 关键源码逐行解剖

### Host 端单线程原子组调度

摘自 `01_nccl_allreduce/nccl_allreduce.cu`：

```cpp
// 1. 发起 NCCL 任务推排的组屏障（Group Barrier）
// 为什么需要 GroupStart / End？
// 因为在一个 Host 线程中轮询操作多个 GPU 发起通信时，如果没有组装包裹，
// 调用完第一张卡的归约函数后，该线程可能因为旧版 NCCL API 行为发生阻塞，
// 从而导致它永远无法为第二张卡发出归约指令，触发多卡死锁！
NCCLCHECK(ncclGroupStart());

// 2. 依次遍历控制的物理卡
for (int i = 0; i < nDev; ++i) {
    // 强制将 Host 当前线程的环境挂载到第 i 张卡上
    CUDA_CHECK(cudaSetDevice(i));
    
    // 3. 发布异步协同指令。从各自的发送区拿数据，指定长度及 Float 类型。
    // 使用 ncclSum 表示在传输途经显存时顺便将相同偏移处的值全部累加
    // 放入其对应的通信子 (comms) 与自己的异步流 (streams) 里
    NCCLCHECK(ncclAllReduce((const void*)d_sendbuffs[i], (void*)d_recvbuffs[i], 
                            num_elements, ncclFloat, ncclSum, comms[i], streams[i]));
}

// 4. 将以上积累的拓扑命令一次性打入各卡的命令引擎并允许流开跑
NCCLCHECK(ncclGroupEnd());
```

**解剖结论**：这是多卡代码中最容易出错的致命盲点之一—死锁 (Deadlock)。对于多线程/多进程启动的 MPI 程序可能不需要 `ncclGroup*` 的辅助，但在单线程下调度全局多显卡，这是保证安全时序的最核心护栏。

---

## 五、 性能基准与分析

所有数据提取自 `Results/15_Multi_GPU.md` 真实日志：

- **测试硬件**: NVIDIA GeForce RTX 4090 × 2, Linux 环境, nvcc -O3
- **通信库**: NCCL
- **测试规模**: `num_elements = 1,048,576` 元素数组 (百万级阵列, FP32 精度，约 4MB 单卡载荷)

### NCCL 双卡 AllReduce 执行性能表现

| 执行环节 | 初始化策略 (Kernel分配) | 全局预期归约和 | 跨设备归约耗时 (ms) |
| -------- | ----------- | ---------------- | ------------- |
| GPU 0 载荷 | 所有元素置为 `0.0` (Rank 0) | — | — |
| GPU 1 载荷 | 所有元素置为 `1.0` (Rank 1) | — | — |
| **AllReduce 同步** | **1MB 长度并行投掷** | **1.00 (0.0 + 1.0)** | **28.20 ms** |

**分析结论**：
通过日志验证，卡 0 上取回的张量结果彻底全盘正确变为了 `1.00`。虽然只有屈指几行的 `ncclAllReduce` 接口，在底层这套 28 ms 的通信里却穿越了**显存->PCIe控制器->主板总线->对端PCIe控制器->对端显存**的极其幽深的物理链路。在没有专用的 NVLink 网桥支持的消费级 RTX 4090 平台上（NVIDIA 为了区分生产力线物理砍掉了桥接支持），能利用纯 PCIe Gen4 x16 总线高效地跑通跨卡原子求和，本身即印证了 NCCL 极其粗壮和成熟的数据链路重组能力。

---

## 六、 编译及参考资料

### 编译与标准运行指令

借助根目录的统一 `CMakeLists.txt` 构建目标：

> ⚠️ 警告：依赖本章需要你的 Linux 系统提前通过包管理器 (如 `apt install libnccl2 libnccl-dev`) 安装了 NCCL 系列组件。

```bash
# 1. 切换至项目根目录并执行整体配置（首次构建，确保 NCCL 检测通过）
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. 独立编译对应的子项目 Target 
cmake --build build --target nccl_allreduce -j8

# 3. 标准二进制验证运行 (需至少具备双卡物理环境)
./build/15_Multi_GPU/01_nccl_allreduce/nccl_allreduce
```

### 推荐阅读

- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) —— 官方用户指南，囊括了所有的集合操作（Broadcast, Scatter, ReduceScatter）图文说明。
- [Massively Scale Your Deep Learning Training with NCCL (DevBlogs)](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) —— 了解 NCCL Ring 算法的底层带宽拆分原理和时序流转奥秘。
- [PyTorch DistributedDataParallel 原理概述](https://pytorch.org/docs/stable/notes/ddp.html) —— 工业界最为主流的框架之一是如何将底层的 NCCL 调用内嵌进模型的反向传播钩子 (Hook) 中的。
