// Thrust - CUDA 并行算法库
#include <code_abbreviation.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>


// 用于和 Thrust transform 做对比的手写 SAXPY (GPU kernel，手写)
__global__ void saxpy_kernel_manual(CPFloat x, PFloat y, CFloat a, CInt n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}


// CPU 排序 (CPU，手写)
void sort_cpu(std::vector<float>& vec) {
    std::sort(vec.begin(), vec.end());
}

// CPU 归约 (CPU，手写)
// 使用 double 累加避免 float 顺序求和的大规模舍入误差
float reduce_cpu(const std::vector<float>& vec) {
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return static_cast<float>(sum);
}

// CPU 变换 SAXPY (CPU，手写)
void saxpy_cpu(const std::vector<float>& x, std::vector<float>& y, float a) {
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = a * x[i] + y[i];
    }
}

// CPU 前缀和 (CPU，手写)
void inclusive_scan_cpu(const std::vector<float>& input, std::vector<float>& output) {
    if (input.empty()) return;
    output[0] = input[0];
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i - 1] + input[i];
    }
}


// 验证结果（AI 生成）
bool verify_results(const std::vector<float>& gpu_result, const std::vector<float>& cpu_result, const string& kernel_name) {
    bool success = true;
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        if (std::abs(gpu_result[i] - cpu_result[i]) > 1e-4f) {
            cout << "✗ " << kernel_name << " FAILED: 索引 " << i 
                 << " 结果 " << gpu_result[i] << " (期望 " << cpu_result[i] << ")\n";
            success = false;
            break;
        }
    }
    if (success) {
        cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result[0] << " (期望 " << cpu_result[0] << ")\n";
    }
    return success;
}

bool verify_value(float gpu_result, float cpu_result, const string& kernel_name) {
    float diff = std::abs(gpu_result - cpu_result);
    float mag  = std::max(std::abs(gpu_result), std::abs(cpu_result));
    float tol  = std::max(1e-2f, 1e-5f * mag);  // 相对+绝对混合容差
    if (diff > tol) {
        cout << "✗ " << kernel_name << " FAILED: 结果 " << gpu_result << " (期望 " << cpu_result << ")"
             << " (diff=" << diff << ", tol=" << tol << ")\n";
        return false;
    }
    cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result << " (期望 " << cpu_result << ")\n";
    return true;
}


// GPU 计时结果结构体（AI 生成）
struct GpuTimingResult {
    float h2d_ms;      // Host to Device 传输时间
    float kernel_ms;   // Kernel 执行时间（多次平均）
    float d2h_ms;      // Device to Host 传输时间
    float total_ms;    // 总时间
};


// 注意：Thrust API 屏蔽了 Kernel 配置，我们使用 CudaTimer 直接包裹 thrust 闭包执行

// Thrust Sort 测试 (GPU，手写)
GpuTimingResult thrust_sort_gpu(const std::vector<float>& h_input, std::vector<float>& h_output, CInt iterations) {
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    thrust::device_vector<float> d_vec = h_input;
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    // 预热 (复制一份因为原地排序只能发生一次结构变动，反复排已排好数组无意义)
    thrust::device_vector<float> d_temp = d_vec;
    thrust::sort(d_temp.begin(), d_temp.end());
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for(int i=0; i<iterations; ++i) {
        d_temp = d_vec; // 每次重置回乱序状态
        thrust::sort(d_temp.begin(), d_temp.end());
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    thrust::copy(d_temp.begin(), d_temp.end(), h_output.begin());
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    return result;
}

// Thrust Reduce 测试 (GPU，手写)
GpuTimingResult thrust_reduce_gpu(const std::vector<float>& h_input, float& output_val, CInt iterations) {
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    thrust::device_vector<float> d_vec = h_input;
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    // 预热
    output_val = thrust::reduce(d_vec.begin(), d_vec.end());
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for(int i=0; i<iterations; ++i) {
        output_val = thrust::reduce(d_vec.begin(), d_vec.end());
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    result.d2h_ms = 0.0f; // reduce 返回主机的结果是标量内部含带的
    result.total_ms = result.h2d_ms + result.kernel_ms;
    return result;
}

// Thrust Transform 测试 (包含 functor)
struct saxpy_functor {
    const float a;
    saxpy_functor(float _a) : a(_a) {}
    __host__ __device__
    float operator()(const float& x, const float& y) const {
        return a * x + y;
    }
};

// Thrust Transform 测试 (包含 functor) (GPU，手写)
GpuTimingResult thrust_transform_gpu(const std::vector<float>& h_x, const std::vector<float>& h_y, std::vector<float>& h_out, float a, CInt iterations) {
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    thrust::device_vector<float> d_out(d_x.size());
    
    // 预热
    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_out.begin(), saxpy_functor(a));
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for(int i=0; i<iterations; ++i) {
        // 利用 d_out 接住输出，避免 d_y 原地叠加
        thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_out.begin(), saxpy_functor(a));
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    thrust::copy(d_out.begin(), d_out.end(), h_out.begin());
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    return result;
}


// 主函数（部分手写，部分AI 生成）
int main() {
    CInt N = 10000000; // 1000万元素测试规模
    CInt iterations_base = 100;

    CSize size_bytes = N * sizeof(float);
    const double total_mb = size_bytes / (1024.0 * 1024.0);

    printDeviceInfo();
    cout << "========================================\n";
    cout << "      Thrust 核心算法性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << N << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Kernel 迭代次数：" << iterations_base << " 次\n";
    cout << "\n";

    // 设置数据
    std::mt19937 gen(1048576);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = dist(gen);
    }
    
    cout << "---------- [1] Sort (排序) ----------\n";
    std::vector<float> h_sort_cpu = h_input;
    std::vector<float> h_sort_gpu(N);
    
    CpuTimer sortTimer;
    sortTimer.start();
    sort_cpu(h_sort_cpu);
    sortTimer.stop();
    double cpu_sort_ms = sortTimer.elapsed_ms();
    
    GpuTimingResult res_sort = thrust_sort_gpu(h_input, h_sort_gpu, 5); // 排序耗时较长，迭代5次
    cout << "CPU std::sort 时间：" << setw(8) << cpu_sort_ms << " ms\n";
    cout << "GPU thrust::sort 时间：" << setw(8) << res_sort.kernel_ms << " ms (5次平均)\n";
    cout << "GPU Sort 加速比：" << setprecision(2) << cpu_sort_ms / res_sort.kernel_ms << "x\n";
    verify_results(h_sort_gpu, h_sort_cpu, "thrust::sort\t");
    cout << "\n";
    
    cout << "---------- [2] Reduce (归约 sum) ----------\n";
    CpuTimer reduceTimer;
    reduceTimer.start();
    float cpu_sum = reduce_cpu(h_input);
    reduceTimer.stop();
    double cpu_reduce_ms = reduceTimer.elapsed_ms();
    
    float gpu_sum = 0.0f;
    GpuTimingResult res_reduce = thrust_reduce_gpu(h_input, gpu_sum, iterations_base);
    cout << "CPU std::accumulate 时间：" << setw(8) << cpu_reduce_ms << " ms\n";
    cout << "GPU thrust::reduce 时间：" << setw(8) << res_reduce.kernel_ms << " ms (" << iterations_base << " 次平均)\n";
    cout << "GPU Reduce 加速比：" << setprecision(2) << cpu_reduce_ms / res_reduce.kernel_ms << "x\n";
    
    double bw_reduce = (total_mb / 1024.0) / (res_reduce.kernel_ms / 1000.0);
    cout << "Reduce 有效带宽：" << setprecision(2) << bw_reduce << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    verify_value(gpu_sum, cpu_sum, "thrust::reduce\t");
    cout << "\n";

    cout << "---------- [3] Transform (SAXPY元素级) ----------\n";
    std::vector<float> h_x = h_input; // 复用刚才的随机数
    std::vector<float> h_y(N, 1.0f);
    std::vector<float> h_y_cpu = h_y;
    std::vector<float> h_y_gpu(N);
    float a = 2.5f;

    CpuTimer transformTimer;
    transformTimer.start();
    saxpy_cpu(h_x, h_y_cpu, a);
    transformTimer.stop();
    double cpu_trans_ms = transformTimer.elapsed_ms();
    
    GpuTimingResult res_trans = thrust_transform_gpu(h_x, h_y, h_y_gpu, a, iterations_base);
    cout << "CPU for-loop SAXPY 时间：" << setw(8) << cpu_trans_ms << " ms\n";
    cout << "GPU thrust::transform 时间：" << setw(8) << res_trans.kernel_ms << " ms (" << iterations_base << " 次平均)\n";
    cout << "GPU Transform 加速比：" << setprecision(2) << cpu_trans_ms / res_trans.kernel_ms << "x\n";
    
    // 访存量：读X、读Y、写Y，即 3 * total_mb
    double bw_trans = (total_mb * 3 / 1024.0) / (res_trans.kernel_ms / 1000.0);
    cout << "Transform 有效带宽：" << setprecision(2) << bw_trans << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    // 只能校验一次的数学结果（因为迭代会叠加Y值），所以预处理重新单次调用用于校验：
    std::vector<float> h_y_gpu_verify(N);
    thrust_transform_gpu(h_x, h_y, h_y_gpu_verify, a, 1);
    verify_results(h_y_gpu_verify, h_y_cpu, "thrust::transform");
    cout << "\n";

    cout << "========================================\n";

    return 0;
}
