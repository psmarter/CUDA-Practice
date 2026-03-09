# 04_GEMM_Optimization：跨越内存墙，逼近 cuBLAS 极限的寄存器分块

## 1. 为什么 Shared Memory 还不够快？
在 `01_Basics` 我们学习了共享内存分块（Shared Memory Tiling），将全局内存重复访问次数降为原来的 $1/TILE\_SIZE$。然而，即便是极慢的 Global Memory 得到救赎后，Shared Memory 依然会成为新的瓶颈。
计算一个点的结果，线程每一次循环仍然要从 Shared Memory 读两次（A一个，B一个）！大量的指令被用去执行共享内存 Load 操作，ALU （逻辑运算单元）的饥饿状态没能完全解决。

## 2. 寄存器：显卡上最昂贵但最快的资产
**Thread Coarsening**（寄存器分块，Register Tiling）的核心哲学是：**一次多读取，反复去剥削**。
如果一个线程不再只计算 $C$ 中的一个点，而是计算 $C$ 中的一个 $2 \times 2$、$4 \times 4$ 甚至 $8 \times 8$ 的矩阵块呢？
- 我们将从 A 读到的元素存到私有寄存器。
- 我们将从 B 读到的元素存到私有寄存器。
- 寄存器里的值可以利用外积（Outer Product）的数学性质组合乘加！一条存活在寄存器里的A元素，可以和8个B元素连续乘加，这就省下了7次从 Shared Memory 索要数据的开销！

## 3. 向量化拾取与 Double Buffering
要在 $128 \times 128$ 这样的巨型 Block 下维持满血循环，还需要使用 **`float4` 指令加载**。一次拿到四份数据，大幅降低指令发送频次（Issue Rate）。
而 **Double Buffer**（乒乓缓冲）就是将 `Shared Memory` 的容量开成平常的两倍：
- 当 ALU 正在全力吞咽 Bank 0 里的数据做运算时，
- LD/ST 管线（访存单元）正在忙着把下一块全局数据搬入 Bank 1，
完美做到了 计算与访存的重叠（Overlap）。

## 4. 真实对比验证：硬刚 cuBLAS
在本次测试中我们把矩阵放大到 `2048 x 2048`，这需要 $O(8 \times 10^9)$ 次乘加操作。
- **纯串行 CPU** 不出意料地长考了超过两万多毫秒（~23秒），性能落寞在 0.74 GFLOPS。
- **手写寄存器分块 (Register Tiling)**，单轮只需 0.60 毫秒，怒飙 **28.86 TFLOPS**。这几乎是完全手打 CUDA C++、没有碰到底层汇编的顶级水平！
- 作为对比，我们接入了 NVIDIA 的大杀器 **cuBLAS SGEMM**。它依靠极其紧缩的汇编调度（甚至偷偷使用 Tensor Core）跑到了 **57.63 TFLOPS**。
实现了商业库 50.1% 的压榨率，这标示着我们已经叩开了高级算子定制的大门。
