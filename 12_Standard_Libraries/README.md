# 12_Standard_Libraries: CUDA 生态工业标准库

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

虽然徒手编写 Kernel 可以获得对硬件的无上掌控感，但在生产环境中，盲目造轮子是不推荐的——NVIDIA 原厂提供的库往往历经数百乃至数千工时的手写汇编级调优，几乎构成了特定领域的算力极值点。本章旨在学习如何优雅、正确并高效地调用官方的高性能标准库，以替代日常繁复的基建代码。

目录包含以下业界标准级基石库的运用：
- `01_cublas_gemm/`：调用 cuBLAS（CUDA Basic Linear Algebra Subprograms）。这是业界最高水准的线性运算基础库，也是各种深度学习框架（如 PyTorch）的默认核心底座。
- `02_cufft/`：调用 cuFFT，执行极致并行的快速傅里叶变换。在信号处理、图像频域变换中占据统治地位。
- `03_thrust/`：使用 Thrust 库。它是 CUDA 中的 STL（标准模板库），提供了与 C++ `<algorithm>` 和 `<vector>` 语法高度相似的设备端容器与高阶规约、排序、扫描算法接口。

## 2. 原理推导与数学表达 (Math & Logic)

以 cuBLAS 中的标准 SGEMM (Single Precision GEMM) 为例，它计算的是：
$$ C = \alpha \, op(A) \, op(B) + \beta \, C $$

这里的数学难点不在于公式，而在于 cuBLAS 默认采用**列优先（Column-Major）**的内存布局（源于 Fortran 遗产），而 C/C++ 是行优先（Row-Major）的。换言之，如果在 C 语言中开辟了 $A$ 与 $B$，直接传给 cuBLAS 时，它会将其逻辑上“看作”是 $A^T$ 与 $B^T$。

由此，要计算 $C = A \cdot B$，我们利用转置的数学性质：
$$ C^T = (A \cdot B)^T = B^T \cdot A^T $$
所以在 C++ 中调用时，只需要将 $B$ 当作第一操作数，$A$ 当作第二操作数并传入，硬件就能基于列优先算出行优先格式的正确结果，巧妙地避开显存的物理重排。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

以 Thrust 库的高级模板和隐式内存传输为例：

```text
Host Code (CPU 侧)                     Device Memory (GPU 侧)

std::vector<int> h_data = {1, 2, 3};
             |
             v
thrust::device_vector<int> d_data = h_data;  
// ⬇️ Thrust 自动拦截，调用 cudaMalloc 并在后台发起隐式 cudaMemcpy
             |
             +---------------------> [1, 2, 3] 完全驻留并行显存

thrust::reduce(d_data.begin(), d_data.end());
// ⬇️ Thrust 自动推导最优 Block 数量与 Shared Memory
//    并在后台执行多层级 (Multi-stage) 规约 Kernel，透明返回结果
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `01_cublas_gemm/cublas_gemm.cu`，演示工业级的 cuBLAS 调用与巧妙的维度抗衡机制：

```cpp
cublasHandle_t handle;
cublasCreate(&handle); // 极重负载操作，必须事先创建并复用于整个生命周期

float alpha = 1.0f, beta = 0.0f;

// ⚠️ 极其巧妙的利用了转置重组特性
// 我们期望的是 C(MxN) = A(MxK) * B(KxN)
// 我们分别传入 d_B 然后是 d_A 以抗衡列优先机制
// 将目标结果矩阵的宽高度设为：N, M，共用维度设为：K
// 由于数据是连续分配的，主导维度 (Leading Dimension, lda/ldb/ldc) 就是其在 C 语言里的原始列宽
cublasSgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            N, M, K, 
            &alpha, 
            d_B, N,  // ldb = N
            d_A, K,  // lda = K
            &beta, 
            d_C, N); // ldc = N

cublasDestroy(handle);
```

## 5. 性能基准与分析视角 (Performance & Profiling)

在 RTX 4090 上，我们抓取了工业级库无以伦比的真实巅峰效能（这通常是我们手写基准难以企及的水平）：

**1. cuBLAS (M=N=K=1024): FP32**
*   CPU 串行耗时: **2854.91 ms**
*   基于启发式自动调优的 `cublasLtMatmul`: **0.04 ms (高达 49.74 TFLOPS)**
*   加速比打底 **65878 倍**，且库会自动切分布局，调度最适配底层硬件 (TensorCore) 的汇编指令。

**2. cuFFT (4096 点 1D 离散傅里叶变换):**
*   CPU 本地双层 for 循环 ($O(N^2)$): 392.06 ms
*   GPU 并行 `cufftExecC2C` ($O(N\log N)$): 单次仅需 **0.0036 ms (加速十万倍量级)**。
*   在超大 Batch(65536) 频域推演下，显存吞吐率直逼 **457.01 GB/s**。

**3. Thrust 核心范式算法 (千万级元素 / 38MB):**
*   `thrust::sort` (基数排序): 1.30 ms (**1625 倍** CPU std::sort)。
*   `thrust::reduce` (归约): 0.08 ms (实测 **485 GB/s** 带宽)。
*   `thrust::transform` (SAXPY 访存密集型): 0.13 ms (高达 **850.92 GB/s**)。几乎跑满了物理极限带宽，这就是封装底座的力量。

## 6. 编译指引与参考资料 (Compile & References)

```bash
mkdir build && cd build
cmake ..
# CMake 脚本中利用了 CMAKE_CUDA_STANDARD 并链接预装的库
make cublas_gemm cufft_example thrust_algorithms
```

**参考资料:**
- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [NVIDIA cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/index.html)
- [Thrust - The C++ Parallel Algorithms Library](https://thrust.github.io/)