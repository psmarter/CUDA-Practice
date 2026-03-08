#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>

// 检查一般 CUDA API 返回值
// (call) 加括号是避免宏参数优先级出问题
// 不使用std::endl，防止重复刷新，endl会强制刷新输出
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA 错误：\n" \
                << "    表达式：" << #call << "\n" \
                << "    文件：" << __FILE__ << "\n" \
                << "    行数：" << __LINE__ << "\n" \
                << "    错误代码：" << static_cast<int>(err) << "\n" \
                << "    错误名称：" << cudaGetErrorName(err) << "\n" \
                << "    错误信息：" << cudaGetErrorString(err) << "\n" \
                << std::flush; \
            std::exit(1); \
        } \
    } while (0)

// 检查 kernel 启动错误
// cudaGetLastError() 获取最后一个 CUDA API 调用的错误，会清除错误信息
#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA 内核启动错误：\n" \
                << "    文件：" << __FILE__ << "\n" \
                << "    行数：" << __LINE__ << "\n" \
                << "    错误代码：" << static_cast<int>(err) << "\n" \
                << "    错误名称：" << cudaGetErrorName(err) << "\n" \
                << "    错误信息：" << cudaGetErrorString(err) << "\n" \
                << std::flush; \
            std::exit(1); \
        } \
    } while (0)

// 检查 CUDA 运行时错误
// cudaDeviceSynchronize() 等待所有 CUDA 操作完成
#define CUDA_SYNC_CHECK() \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA 运行时错误：\n" \
                << "    文件：" << __FILE__ << "\n" \
                << "    行数：" << __LINE__ << "\n" \
                << "    错误代码：" << static_cast<int>(err) << "\n" \
                << "    错误名称：" << cudaGetErrorName(err) << "\n" \
                << "    错误信息：" << cudaGetErrorString(err) << "\n" \
                << std::flush; \
            std::exit(1); \
        } \
    } while (0)

// 打印设备信息
inline void printDeviceInfo() {
    using std::cout;
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        cout << "未检测到 CUDA 设备\n";
        return;
    }
    cout << "检测到 " << deviceCount << " 块 CUDA 设备\n";
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        cout << "设备 " << i << "： " << prop.name << "\n";
        cout << "  计算能力：" << prop.major << "." << prop.minor << "\n";
        double totalGB = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        cout << "  全局显存：" << std::fixed << std::setprecision(2) << totalGB << " GB\n";
        cout << "  每个 Block 共享内存：" << prop.sharedMemPerBlock << " Bytes\n";
        cout << "  每个 Block 最大线程数：" << prop.maxThreadsPerBlock << "\n";
        cout << "  Block 维度上限：("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")\n";
        cout << "  Grid 尺寸上限：("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")\n";
        cout << "  Warp 大小：" << prop.warpSize << "\n";
        cout << "  SM 数量：" << prop.multiProcessorCount << "\n";
        cout << "  每个 SM 最大线程数：" << prop.maxThreadsPerMultiProcessor << "\n";
        // cout.unsetf(std::ios::floatfield);
        // cout << std::setprecision(6);
    }
    cout << "\n";
}

// 辅助函数：向上取整除法
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// 辅助函数：Warp 级最大值归约（GPU 函数，手写）
/* 
    早期的 CUDA 编程中，如果两个线程想要交换数据，必须把数据写到 Shared Memory 里，然后调用 __syncthreads()，再由另一个线程去读。这很慢。
    后来 NVIDIA 引入了 Shuffle 指令。它允许同一个 Warp（32个线程组）内的线程，直接读取其他线程的寄存器，延迟极低，且不需要显式同步屏障（barrier）。
    __shfl_down_sync(0xffffffff, val, offset) 的意思是：
        0xffffffff (Mask): 参与掩码。表示 Warp 中的 32 个线程全部参与这次操作（每一位代表一个线程，32个 1 就是全 F）。
        val: 我要拿出来分享的值。
        offset: 偏移量。“向高位线程索取数据”。例如，如果 offset 是 16，那么 0 号线程会去读 16 号线程的 val，1 号读 17 号的，依此类推。
*/
__inline__ __device__ float warp_reduce_max(float val) {
#pragma unroll      // 循环展开（offset是编译期已知的，把for循环拆解成5行连续的代码，消除了循环控制的指令开销）
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// 辅助函数：Warp 级求和归约（GPU 函数，手写）
__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 辅助函数：Warp 级求和归约（GPU 函数，手写，使用 XOR 模式）
__device__ inline float kernel_warp_reduce_sum(float val) {
    // 所有 lane 都得到 warp 总和
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset); // 与异或距离内的线程相加
    }
    return val;
}

// 辅助函数：Warp 级求最大值归约（GPU 函数，手写，使用 XOR 模式）
__device__ inline float kernel_warp_reduce_max(float val) {
    // 所有 lane 都得到最大值
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// 辅助函数：Warp 级前缀和（Inclusive，GPU 函数，手写）
__device__ inline float kernel_warp_scan_inclusive(float val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        
        // 只有当当前线程的 laneId >= offset 时，上面那个线程的值才是有效的前缀部分，需要累加
        int laneId = threadIdx.x % 32;
        if (laneId >= offset) {
            val += n;
        }
    }
    return val;
}

// 辅助函数：Warp 级前缀和（Exclusive，GPU 函数，手写）
__device__ inline float kernel_warp_scan_exclusive(float val, float& total) {
    float inclusive_val = kernel_warp_scan_inclusive(val);
    
    total = __shfl_sync(0xffffffff, inclusive_val, 31);
    
    return inclusive_val - val;
}

//将 FP32 数据转换至 FP16 辅助辅助转换（GPU kernel，手写）
__global__ void float2half_kernel(const float* in, half* out, const int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = __float2half(in[tid]);
    }
}