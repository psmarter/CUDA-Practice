# 13_Performance_Analysis: 性能分析与瓶颈定位

## 1. 全景导览与学习目标 (Overview & Learning Objectives)

不善用剖析器（Profiler）的程序员只能在黑箱外猜想，“测量胜于直觉”，在底层硬件优化上更是绝对的真理。本章的学习目标是深入掌握 NVIDIA 的原厂性能杀器：Nsight Systems (`nsys`) 与 Nsight Compute (`ncu`)，并结合理论分析模型剖析 Kernel 性能瓶颈。

目录下的子模块通过不同维度展示了性能度量的基本原理与实践：

- `01_occupancy/`：探讨 Occupancy（SM 驻留率）与 ILP（指令级并行）的权衡。展示如何通过调整线程数和每线程处理的数据量（Data per Thread），在低 Occupancy 下利用 ILP 完美隐藏延迟并跑满带宽。
- `02_roofline/`：基于实际测量的硬件峰值（Peak TFLOPS 与 Peak Bandwidth），验证经典的屋顶线模型（Roofline Model）。展示 Vector Add（Memory Bound）与 GEMM（Compute Bound）在算术强度（Arithmetic Intensity）层面的差异。
- `03_nsight_profiling/`：展示如何使用 NVTX (NVIDIA Tools Extension) 在 C++ 源码中植入 `nvtxRangePush` 和 `nvtxRangePop` 性能标记，以方便 `nsys` 追踪时序以及验证 Coalesced/Non-coalesced 访存行为的带宽差异。

## 2. 原理推导与数学表达 (Math & Logic)

**屋顶线模型 (Roofline Model)** 的核心表达反映了计算量与访存量的制约关系。
- 横轴：算术强度（Arithmetic Intensity）$I$ = 总计 FLOPs / 总计内存读取与写入 Bytes
- 纵轴：内核算力性能 $P$ (FLOPs/second)

屋顶的平坦上限由硬件理论计算峰值限制：$P_{peak}$
受限于内存带宽的左侧斜坡限制方程式为：带宽高斜率 $BW_{peak}$
$$ P = I \times BW_{peak} $$

内核的最终可达性能 $P_{achieved}$ 是二者的下限：
$$ P_{achieved} = \min(P_{peak}, \; I \times BW_{peak}) $$

拐点（Ridge Point）计算公式：
$$ I_{ridge} = \frac{P_{peak}}{BW_{peak}} $$
当内核的算术强度 $I < I_{ridge}$ 时，处于 Memory Bound 区；反之则处于 Compute Bound 区。

## 3. 内存与线程的硬核博弈 (Mapping & Occupancy)

Occupancy（驻留率）指的是当前 SM 上活跃的 Warp 数占该 SM 最大可并发 Warp 数的比例。经典的直觉认为 100% 的 Occupancy 能最大化隐蔽全局访存延迟。然而，通过 `01_occupancy` 可证实指令级并行（ILP）的威力。

在同样的 256M 个浮点数拷贝任务中：
- **高 Occupancy (256 Threads, 1 element/thread)**：虽然有更多的活跃 Warp 在等待，但由于单线程执行指令的跨度太短，带宽利用率约为 1212 GB/s。
- **低 Occupancy + ILP (64 Threads, 16 elements/thread)**：通过寄存器展开，单个线程一次性发射多条独立访存指令。此时 Occupancy 极低，但掩盖延迟的效率极高，带宽飙升至 1365 GB/s。

> **核心结论**：盲目追求 100% Occupancy 是误区。适当增加每个线程的工作量（Register Pressure / ILP），只要不引发严重的 Register Spilling，往往能带来突破物理占用上限的极致性能。

## 4. 关键源码逐行解剖 (Code Deep-Dive)

来自 `03_nsight_profiling/nsight_profiling.cu`，利用 NVTX 辅助 profiling 发射测定。
通过 RAII 机制包装 `nvtxRangePush`，我们可以在 Nsight 系统时间轴上看到极为干净明了的色带，完美避开 CPU/GPU 异步执行带来的测量杂讯。

```cpp
// NVTX (NVIDIA Tools Extension SDK) 标记初始化结构
#include <nvtx3/nvToolsExt.h>

// 利用 C++ RAII 原则封装的 NVTX 区间追踪器
class NVTXRange {
public:
    NVTXRange(const char* name) {
        nvtxRangePush(name); // 压入带有名称的阶段标签，方便 nsys 可视化
    }
    ~NVTXRange() {
        nvtxRangePop();      // 生命周期结束时自动弹出标签
    }
};

void run_benchmark() {
    // 实例化后，它会自动统计后续 kernel_launch 直到作用域结束的完整时长
    NVTXRange range("Benchmark_Coalesced_Memory");
    
    // 发射 Kernel
    good_kernel<<<grid, block>>>(d_in, d_out, N);
    
    // 强制系统同步，确保 NVTX 的 Pop 操作准确打在 Kernel 结束后 
    cudaDeviceSynchronize(); 
}
```

## 5. 性能基准与诊断基准 (Performance & Profiling)

| 场景 | 配置 | 带宽(GB/s) / 算力(GFLOPS) | 瓶颈分析 / 结论 |
|----|----|---------|---------------|
| `Occupancy 满载` | 256 Threads, 1 Data | 1212.02 GB/s | 虽然活跃 Warp 多，每线程吞吐贫弱。 |
| `ILP 指令级并行` | 64 Threads, 16 Data | **1365.11 GB/s** | 显著逼近 4090 的峰值显存带宽，指令流水线填满。 |
| `Roofline: VecAdd` | 算术强度 I ≈ 0.083 | 78.69 GFLOPS | $I < 85.33$，极其缺乏复用，标准的 Memory Bound。 |
| `Roofline: GEMM` | 算术强度 I ≈ 170.6 | **5215.82 GFLOPS** | $I > 85.33$，计算密集型，被 Peak TFLOPS 限制。 |
| `Nsight: 坏访存` | 步长 32 跳跃访问 | 273.52 GB/s | Non-coalesced 触发 Cache 惩罚，读取大量无效数据。|
| `Nsight: 连续访存`| 顺延物理内存寻址 | **1223.57 GB/s** | Coalesced 完全命中了 128 Bytes 一次性提取事务局域性。|

## 6. 编译构建与调试指引 (Compile & Build)

本目录的性能分析示例依赖于完整的编译报告和 `-lineinfo`。
在 `03_nsight_profiling` 中还需要链接 NVTX SDK 库（`-lnvToolsExt`）。

```bash
# 所有项目已集成于 CMakeLists.txt，标准构建：
cd build && make -j4

# Nsight Systems 系统级全局时序追踪分析（宏观）
nsys profile -t cuda,nvtx -o my_profile --stats=true ./13_Performance_Analysis/03_nsight_profiling/nsight_profiling

# Nsight Compute 内核级深度硬件监控分析（微观），必需加 -lineinfo 编译
ncu --set full -o ncu_report ./13_Performance_Analysis/03_nsight_profiling/nsight_profiling
```
