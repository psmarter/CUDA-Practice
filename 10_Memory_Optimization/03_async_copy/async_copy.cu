// 异步内存拷贝 - cuda::memcpy_async
#include <code_abbreviation.h>
#include <string>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

// 同步拷贝（基准）（GPU kernel，手写）
__global__ void sync_copy_kernel(CPFloat input, PFloat output, CInt n) {
    __shared__ float shared[ASYNC_TILE];
    
    int tid = threadIdx.x;
    int items_per_block = ASYNC_TILE;
    int total_blocks = gridDim.x;
    
    // 一个 Block 处理多个 Tile 以掩盖延迟，体现流水线优势
    for (int tile_idx = blockIdx.x; tile_idx < cdiv(n, items_per_block); tile_idx += total_blocks) {
        int gid = tile_idx * items_per_block + tid;
        
        // 阶段 1: 同步加载到 shared memory
        if (gid < n) {
            shared[tid] = input[gid];
        }
        __syncthreads();
        
        // 阶段 2: 计算并写回 global memory
        if (gid < n) {
            output[gid] = shared[tid] * 2.0f;
        }
        __syncthreads();
    }
}

// 异步单步拷贝（GPU kernel，手写）
// cg::memcpy_async(block, dst, src, size) 是集体操作：
// 整个 block 协作拷贝 size 字节，必须所有线程无条件调用。
__global__ void async_copy_kernel(CPFloat input, PFloat output, CInt n) {
    __shared__ float shared[ASYNC_TILE];
    
    auto block = cg::this_thread_block();
    int tid = block.thread_rank();
    int items_per_block = ASYNC_TILE;
    int total_blocks = gridDim.x;
    
    for (int tile_idx = blockIdx.x; tile_idx < cdiv(n, items_per_block); tile_idx += total_blocks) {
        int tile_start = tile_idx * items_per_block;
        // 计算当前 tile 实际需要拷贝的元素数（处理边界）
        int tile_elems = min(items_per_block, n - tile_start);
        
        // 集体异步拷贝：整个 block 协作将 tile_elems 个 float 从 global 拷到 shared
        // 所有线程必须无条件调用
        cg::memcpy_async(block, shared, &input[tile_start], sizeof(float) * tile_elems);
        
        // 等待所有异步拷贝完成
        cg::wait(block);
        
        // 计算并写回 global memory
        int gid = tile_start + tid;
        if (gid < n) {
            output[gid] = shared[tid] * 2.0f;
        }
        block.sync();
    }
}

// 多阶段流水线（Multi-stage Pipeline）（GPU kernel，手写）
__global__ void pipeline_kernel(CPFloat input, PFloat output, CInt n) {
    __shared__ float shared[STAGES][ASYNC_TILE];
    
    auto block = cg::this_thread_block();
    int tid = block.thread_rank();
    int items_per_block = ASYNC_TILE;
    int total_blocks = gridDim.x;
    int total_tiles = cdiv(n, items_per_block);
    
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    
    // 我们将整个任务划分为给当前 block 处理的一系列 tiles
    int current_tile = blockIdx.x;
    
    // 预热流水线：填充 STAGES-1 个阶段
    for (int s = 0; s < STAGES - 1; ++s) {
        if (current_tile < total_tiles) {
            int gid = current_tile * items_per_block + tid;
            pipe.producer_acquire();
            if (gid < n) {
                // 使用 cuda::memcpy_async 到 pipeline 中
                cuda::memcpy_async(&shared[s][tid], &input[gid], sizeof(float), pipe);
            }
            pipe.producer_commit();
            current_tile += total_blocks;
        }
    }
    
    // 主循环
    int compute_tile = blockIdx.x;
    
    for (; compute_tile < total_tiles; compute_tile += total_blocks) {
        // 请求下一个需要加载的 tile
        if (current_tile < total_tiles) {
            int load_stage = (compute_tile / total_blocks + STAGES - 1) % STAGES;
            int gid = current_tile * items_per_block + tid;
            
            pipe.producer_acquire();
            if (gid < n) {
                cuda::memcpy_async(&shared[load_stage][tid], &input[gid], sizeof(float), pipe);
            }
            pipe.producer_commit();
            current_tile += total_blocks;
        }
        
        // 消费者等待当前需要计算的阶段就绪
        pipe.consumer_wait();
        
        int compute_stage = (compute_tile / total_blocks) % STAGES;
        int gid = compute_tile * items_per_block + tid;
        
        // 执行计算
        if (gid < n) {
            output[gid] = shared[compute_stage][tid] * 2.0f;
        }
        
        // 释放已消费的阶段
        pipe.consumer_release();
    }
}

// CPU 计算基准（CPU，手写）
void compute_cpu(CRMatrix input, RMatrix output, CInt n) {
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] * 2.0f;
    }
}

// 验证结果（AI 生成）
bool verify_results(CRMatrix gpu_result, CRMatrix cpu_result, CInt n, const string& kernel_name) {
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(gpu_result[i] - cpu_result[i]) > EPSILON) {
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


// GPU 封装函数（GPU，手写）
template<typename KernelFunc>
GpuTimingResult async_gpu(CRMatrix h_input, RMatrix h_output, CInt n, CInt iterations, KernelFunc kernel) {
    PFloat d_input = nullptr, d_output = nullptr;
    CSize size_input = n * FSIZE;
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_input));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_input));
    CUDA_CHECK(cudaMemset(d_output, 0, size_input));
    
    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};
    
    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size_input, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();
    
    // 为了展现 Pipeline 的效果，不能让 gridDim 把所有的 tile 都一次性分配完。
    // 我们强制使用较少的 block 数量，使其必须循环处理多个 tile。
    const dim3 block(ASYNC_BLOCK);
    
    int max_blocks = 2048; // 人为限制活跃 block 数量，产生 pipeline 效用
    int req_blocks = cdiv(n, ASYNC_BLOCK);
    const dim3 grid(std::min(req_blocks, max_blocks));
    
    kernel<<<grid, block>>>(d_input, d_output, n);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_input, d_output, n);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);
    
    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_input, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();
    
    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt n = 1 << 26; // 64M elements, enough work to see pipeline benefits
    CInt iterations = 100;

    CSize size_input = n * FSIZE;
    const double total_mb = size_input / (1024.0 * 1024.0);

    // 打印设备信息
    printDeviceInfo();

    // 打印测试配置
    cout << "========================================\n";
    cout << "      异步内存拷贝 (Async Copy) 性能基准测试\n";
    cout << "========================================\n";
    cout << "数组大小：" << n << " 元素\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << ASYNC_BLOCK << " 线程\n";
    cout << "Pipeline 阶段数：" << STAGES << "\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    Matrix h_input(n);
    Matrix h_cpu_output(n, 0.0f);
    Matrix h_gpu_sync(n, 0.0f);
    Matrix h_gpu_async(n, 0.0f);
    Matrix h_gpu_pipe(n, 0.0f);
    
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i % 100);
    }

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    compute_cpu(h_input, h_cpu_output, n);
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";

    // GPU 版本 1：传统同步拷贝
    cout << "--- GPU 版本 1: 同步共享内存拷贝 (Sync Copy) ---\n";
    GpuTimingResult res_sync = async_gpu(h_input, h_gpu_sync, n, iterations, sync_copy_kernel);
    cout << "H2D 传输时间：   " << setw(8) << res_sync.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_sync.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_sync.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_sync.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2：异步拷贝
    cout << "--- GPU 版本 2: 异步内存拷贝 (Single Stage Async) ---\n";
    GpuTimingResult res_async = async_gpu(h_input, h_gpu_async, n, iterations, async_copy_kernel);
    cout << "H2D 传输时间：   " << setw(8) << res_async.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_async.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_async.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_async.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 3：多阶段异步流水线
    cout << "--- GPU 版本 3: 多阶段异步流水线 (" << STAGES << " Stages Pipeline) ---\n";
    GpuTimingResult res_pipe = async_gpu(h_input, h_gpu_pipe, n, iterations, pipeline_kernel);
    cout << "H2D 传输时间：   " << setw(8) << res_pipe.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << res_pipe.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << res_pipe.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << res_pipe.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析与带宽利用率 ---\n";
    double bytes = 2.0 * n * FSIZE; // 读写各一次
    double bw_sync = (bytes / 1e9) / (res_sync.kernel_ms / 1000.0);
    double bw_async = (bytes / 1e9) / (res_async.kernel_ms / 1000.0);
    double bw_pipe = (bytes / 1e9) / (res_pipe.kernel_ms / 1000.0);

    cout << "同步拷贝     有效带宽：" << setw(8) << setprecision(2) << bw_sync << " GB/s\n";
    cout << "单阶异步     有效带宽：" << setw(8) << setprecision(2) << bw_async << " GB/s\n";
    cout << "多阶流水线   有效带宽：" << setw(8) << setprecision(2) << bw_pipe << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";
    cout << "\n";

    // Kernel 性能对比
    cout << "--- Kernel 性能对比 ---\n";
    cout << "同步拷贝 (基准) : " << setw(8) << setprecision(4) << res_sync.kernel_ms << " ms\n";
    cout << "单阶异步 相比同步: " << setprecision(2) << res_sync.kernel_ms / res_async.kernel_ms << "x 加速\n";
    cout << "多阶流水 相比同步: " << setprecision(2) << res_sync.kernel_ms / res_pipe.kernel_ms << "x 加速\n";
    cout << "多阶流水 相比单阶: " << setprecision(2) << res_async.kernel_ms / res_pipe.kernel_ms << "x 加速\n";
    cout << "\n";

    // 结果验证
    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_gpu_sync, h_cpu_output, n, "Sync Copy");
    bool pass2 = verify_results(h_gpu_async, h_cpu_output, n, "Single Stage Async");
    bool pass3 = verify_results(h_gpu_pipe, h_cpu_output, n, "Multi-stage Pipeline");

    // GPU/CPU 结果一致性验证
    if (pass1 && pass2 && pass3) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
