# 02_Reduction: 并行归约与树形算法

## 1. 全局定位与学习价值
在掌握了 `01_Basics` 基础的按元素映射（Element-wise）后，`02_Reduction` 介绍了 CUDA 中最核心的集体协作模式——**归约（Reduction）**。
- **阶段位置**：它是所有高级算法（Scan、Sort、Histogram）的基础组件。
- **硬件瓶颈突破**：展示了如何通过树形访存消除下垂的分支发散（Warp Divergence），以及如何利用 Shared Memory 避免全局内存的冲突和延迟。同时介绍了**线程粗化（Thread Coarsening）**以压榨指令级并行度并摊销控制开销。
- **后续承载**：为 `03_Scan` 前缀和算法与后续的 RMSNorm、Softmax 算子开发提供直接的实现参考。

## 2. 子项目解析
### 2.1 基础归约 (`01_reduce_sum`)
展示了归约操作的核心演进：
1. **交错寻址导致的发散**：使用模运算导致 Warp 内线程发散。
2. **收敛寻址**：跨步内存访问，避免了 Warp Divergence，保证连续的线程处于活跃状态。
3. **共享内存 (Shared Mem)**：将其移植到共享内存，进一步加速块内数据规约。
- **性能提示**：由于测试数据量为单 Block 容量（2048 元素），各种优化的绝对收益不如大规模下明显。

### 2.2 优化归约与原子操作 (`02_reduce_optimized`)
由于 `01_reduce_sum` 只处理单 Block 数据，本例跨越到了**任意长度**和**多 Block 协作**。
- **核心概念**：
  - **Thread Coarsening (线程粗化)**：每个线程在进入 Shared Memory 之前自己串行处理多个元素（如 4 个），极大减少了参与同步的线程和树形宽缩减层数。
  - **原子操作 (`atomicAdd`)**：巧妙地将各个网格块的中间结果聚合到总输出区内。
  - **自定义浮点原子 (`atomicCAS`)**：演示了因为 CUDA 缺失浮点 `atomicMax` 时，如何通过 Compare-And-Swap 从底层指令构建浮点锁。
- **性能验证** (RTX 4090, 1M 元素):
  - 粗化版 (Coarsened) 相较于分段版加速 **1.75x**
  - CPU vs GPU (Kernel) 加速比达 **983x**

### 2.3 向量点积运算 (`03_dot_product`)
将归约应用扩展到了具体的数学操作中。
- **核心概念**：
  - 加载阶段实现乘法计算，规约阶段累加。
  - 引入了 `fmaf(a, b, sum)`，即指令融合乘加 (Fused Multiply-Add)，比单一的 `a * b + sum` 具有更高的计算精度。
- **性能验证** (RTX 4090, 1M 元素点积):
  - 利用 FMA 和 粗化的点积达到了 ~1500+ GB/s 的超高有效内存读写带宽 (L2 Cache 击中提速)。

## 3. 编译与运行
```bash
cd build
cmake ..
make -j
./02_Reduction/01_reduce_sum/reduce_sum
./02_Reduction/02_reduce_optimized/reduce_optimized
./02_Reduction/03_dot_product/dot_product
```
