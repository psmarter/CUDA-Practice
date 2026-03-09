# 03_Scan: 前缀和与多 Block 协作算法

## 1. 全局定位与学习价值
在掌握了 `02_Reduction` (归约) 之后，`03_Scan` (前缀和) 进一步展示了如何处理具有极强前后数据依赖的并行运算。
- **阶段位置**：Scan 属于 CUDA 并行计算的高难度基础算子，它是基数排序（Radix Sort）、词法分析、动态内存分配和流压缩的核心部件。
- **硬件瓶颈突破**：展示了如何在不使用 `atomicAdd` 的情况下，利用树状结构保留计算的中间状态（如 Brent-Kung 算法的两阶段扫描）。
- **后续承载**：为处理不定长数据段并行输出和负载均衡提供了基础。

## 2. 子项目解析
### 2.1 块内前缀和 (`01_prefix_sum`)
该例子演示了如何在一个 Block 的边界内完成前缀和的计算，包含了两种经典算法：
1. **Kogge-Stone 算法 (KS)**：Step-Efficient 但非 Work-Efficient。算法在 $\log N$ 步完成，并行度极高，但是总计算量达到了 $O(N \log N)$ 级别。
2. **Brent-Kung 算法 (BK)**：Work-Efficient 但基于上下向（Up-Sweep 和 Down-Sweep）两阶段合并。算法需要 $2 \log N$ 步，但总计算量为 $O(N)$。在资源受限或规模增大时，BK 会由于占用更少的指令周期而在调度上取得优势。
- **验证数据** (RTX 4090, N=1024)：由于单 Block 较小，Kogge-Stone 以轻微的优势（0.0028 ms vs 0.0037 ms）胜出。

### 2.2 Segmented Scan 与大规模分段 (`02_segmented_scan`)
为了处理超出单个 Block 容量限制的海量数组，简单的同步原语 `__syncthreads()` 已不足够（因为它不跨 Block 生效）。
- **多 Block 协作思想**：
  1. 通过粗化（Thread Coarsening）和局部的归并得到每个 Block 的所有局部 Scan 数据，并提取出 `block_sum`。
  2. 对这些抽离出来的 `block_sum` 执行次级 Scan。
  3. 通过额外的加法 Kernel 把全局和拼接到对应网格数据位上。
- **性能验证** (RTX 4090, 1048576 (1M) 元素)：
  - **加速比**：实现了约 79.1x 的显存 Kernel 计算加速。
  - **精度验证**：由于 1M 个 `float` 串行与并行相加的先后顺序不同，会产生可接受的 FP32 截断误差（0.001% 内算作 PASS）。

## 3. 编译与运行
```bash
cd build
cmake ..
make -j
./03_Scan/01_prefix_sum/prefix_sum
./03_Scan/02_segmented_scan/segmented_scan
```
