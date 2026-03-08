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
这里的数学难点不在于公式，而在于 cuBLAS 默认采用**列优先（Column-Major）**的内存布局（源自百年 Fortran 遗产），而 C/C++ 是行优先（Row-Major）的。
换言之，如果在 C 语言中开辟了 $A$ 与 $B$，直接传给 cuBLAS 时，它会将其“看作”是 $A^T$ 与 $B^T$。
由此，要计算 $C = A \cdot B$，我们利用转置的数学性质：
$$ C^T = (A \cdot B)^T = B^T \cdot A^T $$
所以在调用时，只需要交换传参顺序 $B$ 与 $A$，就可以用列优先算出行优先结果。

## 3. 硬核内存映射解析 (Memory & Thread Mapping)

展现 Thrust 库中 `device_vector` 带来的透明传输：

```text
Host Code (CPU 侧)                     Device Memory (GPU 侧)

std::vector<int> h_data = {1, 2, 3};
             |
             v
thrust::device_vector<int> d_data = h_data;  
// ⬇️ Thrust 自动拦截，调用 cudaMalloc 并在后台发起隐式 cudaMemcpy
             |
             +---------------------> [1, 2, 3] 完全驻留显存

thrust::reduce(d_data.begin(), d_data.end());
// ⬇️ Thrust 在后台自动推导 Block 数量、Shared Mem 配置，并执行一次最佳规约 Kernel
```

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `01_cublas_gemm/cublas_gemm.cu`，演示工业级的 cuBLAS 调用与巧妙的参数交互：

```cpp
cublasHandle_t handle;
cublasCreate(&handle); // 极重负载，必须事先创建并复用于整个生命周期

float alpha = 1.0f;
float beta = 0.0f;

// ⚠️ 极其巧妙的利用了转置重组特性
// 我们期望的是 C(MxN) = A(MxK) * B(KxN)
// 但我们传入 (B, A) 以抗衡它的列优先特性，参数设置：
// 矩阵宽高度为：N, M, K
// 对应的列前导维度（Leading Dimension，即物理一行间距）设为 N, K, N
cublasSgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            N, M, K, 
            &alpha, 
            d_B, N, 
            d_A, K, 
            &beta, 
            d_C, N);

cublasDestroy(handle);
```

## 5. 性能基准与分析视角 (Performance & Profiling)

- **基准**：对比我们之前 `04_GEMM_Optimization` 和 `02_Reduction` 中徒手编写的最顶配 Kernel。
- **典型分析**：使用 NCU 时，我们抓取的并不是单一种类的 Kernel。因为像 cuBLAS 和 Thrust 会依据输入的大小自动分步调度多个极为庞大且深度的汇编 Kernel（如包含了 TensorCore 专用的 `cutlass` 系列内核，或带有 `split_k` 分式的规约内核），这些才是工业标杆的绝对巅峰，通常能够触及理论设备上限的 95% 效率以上。

## 6. 编译指引与参考资料 (Compile & References)

```bash
# 必须显式并入特定的运行时动态链接库标志 -lcublas 和 -lcufft
nvcc -O3 -arch=sm_89 cublas_gemm.cu -o run_libs -lcublas -lcufft
# 注意 Thrust 是内联模板库，只需头文件而无需外部链接库。
```

- 参考资料: NVIDIA cuBLAS Library Documentation / Thrust Quick Start Guide。
