#pragma once
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>
#include <cuda_utils.cuh>

class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
public:
    CpuTimer() = default;
    ~CpuTimer() = default;
    CpuTimer(const CpuTimer&) = delete;         // 禁止拷贝
    CpuTimer& operator=(const CpuTimer&) = delete; // 禁止拷贝赋值
    CpuTimer(CpuTimer&&) = delete;              // 禁止移动
    CpuTimer& operator=(CpuTimer&&) = delete;   // 禁止移动赋值

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }

    void cpu_print_time(const std::string& msg = "CPU 消耗的时间为：") const {
        using std::cout;
        using std::endl;
        cout << msg << std::fixed << std::setprecision(2) << elapsed_ms() << "ms" << endl;
    }
};

class CudaTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
public:
    // 创建事件
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    // 销毁事件
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    CudaTimer(const CudaTimer&) = delete;           // 禁止拷贝
    CudaTimer& operator=(const CudaTimer&) = delete; // 禁止拷贝赋值
    CudaTimer(CudaTimer&&) = delete;                 // 禁止移动
    CudaTimer& operator=(CudaTimer&&) = delete;      // 禁止移动赋值

    // 开始记录事件
    void start() {
        cudaEventRecord(start_event, 0);
    }

    // 停止记录事件
    void stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);   // 等待事件完成
    }

    // 这里必须使用float，因为cudaEventElapsedTime 只接受 float*
    float elapsed_ms() const {
        float duration;
        cudaEventElapsedTime(&duration, start_event, stop_event);
        return duration;
    }

    void cuda_print_time(const std::string& msg = "CUDA 消耗的时间为：") const {
        using std::cout;
        using std::endl;
        cout << msg << std::fixed << std::setprecision(2) << elapsed_ms() << "ms" << endl;
    }
};