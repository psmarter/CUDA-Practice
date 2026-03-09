# 10_Memory_Optimization: 层级访存掌控与带宽压榨的艺术

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

在深度学习的底层算子开发中，**“计算是皮毛，而内存是核心”**。当 CUDA Core 与 Tensor Core 速度快到难以想象时（几十到几百 TFLOPS），程序的真实瓶颈几乎总是落在内存墙（Memory Wall）上。如何把数据以极高的效率“喂”给运算单元，就是内存优化的课题。

本章你将深入学习 CUDA 内存分级体系（Global / Shared / Registers）的高级操控技巧：

- `01_coalesced_access/`：全局内存的 **合并访问 (Coalesced Access)** 与利用向量级 `alignas(16)` 进行 AoS 极致提速；
- `02_bank_conflict/`：剖析并根除共享内存的访存冲突 **(Bank Conflict)** 现象，学习转置/Padding机制；
- `03_async_copy/`：探秘 Ampere 架构后引入的神级异步管道 **Async Copy (**`cuda::memcpy_async`**)**，让数据搬运与计算实现完美隐藏重叠。

## 2. 原理推导与数学表达 (Math & Logic)

在共享内存 (Shared Memory) 中，内存被等宽划分为 32 个独立的 Bank，每个 Bank 在同一时钟周期只能响应一个请求。

**Bank Conflict 惩罚公式：**  
若 Warp 内有 $N$ 个不同的线程由于地址映射问题不小心访问了**同一个 Bank 里的不同地址**（例如跨行读列），我们将这种情况称为 **$N$-way Bank Conflict**。
完成这波读取所需的指令发送周期数 $\text{Cycles} \propto N$。其中严重的情况会导致读写耗时翻几倍乃至三十几倍。

通过引入偏移量 **Padding**，能主动错开不同行的起始 Bank 映射：
$$ \text{Bank}_{\text{pad}} = (\text{Row\_Index} \times (\text{Col\_Size} + 1) + \text{Col\_Index}) \pmod{32} $$
如此可使原本冲突的同列不同行数据，均匀散落在 32 个不同的 Bank 内。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

以带 Padding 的矩阵转置 `shared[TILE_SIZE][TILE_SIZE + 1]` 操作为例：

```text
[无 Padding 时，同一列处于同一 Bank (Bank 0)]
Warp Threads -> T0  T1  T2 ... T31 (列取数据)
shared[0][0] -> B0
shared[1][0] -> B0 (冲突！)
shared[2][0] -> B0 (冲突！) 造成 32次线性排队等待！

[有 Padding (加1列) 后的魔法内存阵列]
shared[0][0] -> B0
shared[1][0] -> 偏移了一个float, 变成了 B1 (安全！)
shared[2][0] -> 偏移了两个float, 变成了 B2 (安全！)
... 这 32 个线程的列访问请求被硬件交叉分发给了 B0~B31，一次性满速返回！
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `02_bank_conflict/bank_conflict.cu` 中对 Padding 优化的精妙实现：

```cpp
__global__ void padded_no_conflict(CPFloat input, PFloat output, CInt n) {
    // 关键改变：列数 + 1 Padding，改变了每一行的长度，错开 bank 映射
    __shared__ float shared[TILE_SIZE][TILE_SIZE + 1];  
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE, by = blockIdx.y * TILE_SIZE;

    // ... 前置数据写入 shared (连续行读取，正常)
    
    // 我们修改写回逻辑，保证 Global Memory 写回是合并的 (coalesced)
    // 即连续的 tx (同一 Warp 线程) 负责写入连续的 out_x (转置后的列)
    int out_x = by + tx;  
    int out_y = bx + ty;  
    if (out_y < n && out_x < n) {
        int out_idx = out_y * n + out_x;
        // tx 作为递增，在此代表读取 Shared 的【行】
        // 原理：因为 Shared 列维度变为 TILE_SIZE+1，所以 tx 变化恰好错开 Bank
        output[out_idx] = shared[tx][ty] * 2.0f;
    }
}
```

## 5. 性能基准与分析视角 (Performance & Profiling)

在 RTX 4090 (理论带宽峰值 ~1008 GB/s) 下完成极高强度测速：

**合并访问 (Coalesced Access):**
*   合并访问: Kernel **0.15 ms**, 有效显存带宽跑满 **924.41 GB/s** (逼近卡皇物理极限)。
*   跨步访问(Stride=2): 有效利用带宽骤降至 **427.99 GB/s**。
*   *彩蛋*: 我们使用了强制 16 字节对齐的 `struct alignas(16) AoS`。由于符合 128-bit 矢量加载指令 (`LDG.E.128`) 排布，它的 AoS 带宽没有发生雪崩降级，仍高达 922.39 GB/s。

**Bank Conflict 实测:**
*   无冲突(只做映射写入): 876 GB/s。
*   严重冲突(未优化列读取): 耗时 **0.183 ms**，带宽跌落 **729 GB/s**（在非转置一维阵列测试中，Stride 32 直接引发 2.27 倍耗时暴涨）。
*   Padding 优化版: 耗时退回 **0.160 ms**，利用率重回 **819 GB/s**，通过小小的一个 `+1`，把转置的硬件障碍降到最低。

## 6. 编译指引与参考资料 (Compile & References)

```bash
# 由于涉及到特殊的异步操作，必须指定较新的 C++ 标准及架构支持
mkdir build && cd build
cmake ..
make -j4

# 观察不同 kernel 下 global load efficiency 指标
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./10_Memory_Optimization/01_coalesced_access/coalesced_access

# 抓取并观察 shared memory conflict 指标
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.avg.pct_of_peak_sustained_elapsed ./10_Memory_Optimization/02_bank_conflict/bank_conflict
```

**参考资料:**
- [NVIDIA CUDA C++ Programming Guide: Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [NVIDIA Blog: Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
