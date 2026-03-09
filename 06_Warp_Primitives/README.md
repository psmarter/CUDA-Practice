# 06_Warp_Primitives: Warp 级指令与寄存器极速通信

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

本章旨在打破使用 Shared Memory 作为线程间主要通信手段的传统范式。在 CUDA 架构中，Warp（包含 32 个线程的锁步执行单元）不仅是调度的最基本单位，其内部各个线程的寄存器之间也具备在底层硬件网格上的互相**直接跨路访问能力**（Warp Shuffle）。学习并掌握 Warp 级原语（Warp Primitives / Shuffle Instructions），是写出零 Shared Memory 延迟、无须 Barrier (`__syncthreads()`) 同步的极限榨干 GPU 性能核心法则。

目录结构及演示逻辑：
- `01_warp_shuffle/`：演示基础的 `__shfl_sync` / `__shfl_down_sync` / `__shfl_up_sync` 等指令。实现同一 Warp 中基于基础广播 (Broadcast)、相邻交换 (Up/Down) 以及蝴蝶图 (XOR) 等操作；
- `02_warp_reduce/`：实战演示如何彻底抛弃传统基于 Shared Memory 的归约逻辑，使用不超过十行的纯 Shuffle 寄存器操作完成超高速的规约；
- `03_warp_scan/`：利用类似的寄存器 `__shfl_up_sync` 流水线式完成 Warp 内的前缀和，并拼接构建 Block 级 Scan。

## 2. 原理推导与数学表达 (Math & Logic)

Warp 内通信的基础在于硬件级的 `__shfl` 特权网络。
以 Warp Reduce Sum 为例，我们在处理带有 32 个线程的归约和时：如果使用 Shared Memory，需要不断的 `LDS/STS` 操作外加 `__syncthreads`。

如今在寄存器尺度使用**下降交换**(Down-Shuffle) 法则：
$$ val_{t} = val_{t} + \text{shfl\_down}(val_t, ~\text{offset}) $$
在第 $k$ 步中，每个线程 $t$ 直接读取自己右侧间隔为 $2^{k}$ 的线程寄存器值并累加（分别针对 `offset=16, 8, 4, 2, 1` 共 5 步即可归约 32 个元素），完全不经过任何访问级缓存、共享内存甚至内存的参与周期。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

Warp Shuffle 是对寄存器阵列的直接微操映射。

```text
[Warp 0 的独立寄存器空间] (并行 32 个 Thread)
  T0    T1    T2    ...    T16   T17   ...   T31
 [v0]  [v1]  [v2]         [v16] [v17]       [v31]
   \_______________^________/
     __shfl_down_sync(mask, val, 16)
        (T0 不用访存，直接拿 T16 计算出最新的 v0)
        (T1 不用访存，直接拿 T17 计算出最新的 v1)
```

注意：上述所有的 `__shfl` 系列指令第一个参数必须提供活跃掩码 Mask（通常我们保证满 Warp 工作，即为 `0xffffffff`），它能确保防止由于部分发散（Warp Divergence）导致某些线程挂起时数据不同步引起死锁。

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `02_warp_reduce/warp_reduce.cu` 的无锁且无全局内存干预归约逻辑：

```cpp
// ⚠️ 极其硬核的寄存器通信：彻底去掉了 shared memory 和 __syncthreads 屏障
__device__ float warpReduceSum(float val) {
    // 每次缩减一半步幅 (16 -> 8 -> 4 -> 2 -> 1)
    // 0xffffffff 表示强制整个 Warp(32线)参与同步
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val; // 仅经历仅仅 5 次寄存器相加。完毕后，Lane 0 (也就是0号线程) 存有 Warp 的有效总和
}

__global__ void kernel_block_reduce_sum(CPFloat input, PFloat output, CInt n) {
    float sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) sum = input[tid];
    
    // 第1步：完成 Warp 内的 32 归一极速累加
    sum = warpReduceSum(sum);

    // 第2步：将 32 归 1 的极少数结果写入 Shared/Register 以便交给最后的主线程归约
    // ...
}
```

## 5. 基准表现与评估剖析 (Performance Data)

在双卡 **NVIDIA GeForce RTX 4090** (极限理论带宽 ~1008 GB/s) 的评测结果震撼发布。对于 128 MB (约3355万个元素) 规模的数据开展测试：

- **基础 Warp Shuffle**:
  - `__shfl_sync` (Broadcast) 、`XOR`及`Up/Down`：时间完全稳锁在 **0.29 ms**（测出带宽达到 **922.92 GB/s**），相比 CPU 单进程快约 **93x**。
- **Warp/Block Reduce Sum**:
  - 极为令人震惊的是，在无共享内存归约后，单趟仅耗时 **0.14~0.15 ms**！有效带宽飙升至 **937.68 GB/s**，完美榨干 4090 的显存控制器（已经无限接近于 `cudaMemcpy` 的理论值上限），相比于 CPU 取得了 **~340 倍** 加速！
- **Warp Scan 前缀和**:
  - 利用 Up-Shuffle 指令级联拼接的并行 Prefix-Sum，处理 128MB 的大规模扫瞄仅耗费 **0.30 ms**（带宽利用达 884.8 GB/s），相对于 C++ 单端获得了 **170 倍的吞吐量提升**！

Warp 指令彻底摆脱了此前需要由 `__syncthreads` 和 L1 SRAM 组成的中间瓶颈。

## 6. 编译及参考 (Compile & References)

```bash
cd build && make -j4 warp_shuffle warp_reduce warp_scan
./06_Warp_Primitives/warp_reduce/warp_reduce
```

深度资料指引与文献参考：
- NVIDIA Developer Blog: [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives)
- CUDA C++ Programming Guide - [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
