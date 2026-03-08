// NCCL Multi-GPU AllReduce 示例
// 演示如何在多个 GPU 之间通过 NCCL 同步和规约数据
//
// 注意：NCCL (NVIDIA Collective Communication Library) 主要在 Linux 平台上支持，
// 并在多卡环境下发挥最好性能。
#include <code_abbreviation.h>
#include <iostream>
#include <vector>

#ifdef __has_include
  #if __has_include(<nccl.h>)
    #define HAS_NCCL 1
  #else
    #define HAS_NCCL 0
  #endif
#else
    #define HAS_NCCL 0
#endif

// NCCL 如果存在，引入头文件并宏定义
#if HAS_NCCL
#include <nccl.h>

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("NCCL ERROR: %s\n", ncclGetErrorString(res)); \
    exit(1);                                        \
  }                                                 \
} while(0)
#endif // HAS_NCCL

// 初始化每张卡的数据 (GPU kernel，手写)
__global__ void init_data_kernel(PFloat data, int rank, CInt size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // 使用 rank 初始化，比如卡0为0，卡1为1
        data[tid] = static_cast<float>(rank);
    }
}

// ========================= 主函数 =========================

int main() {
    printDeviceInfo();

    cout << "========================================\n";
    cout << "      NCCL AllReduce 多卡规约演示\n";
    cout << "========================================\n\n";

#if HAS_NCCL
    int nDev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDev));
    
    if (nDev < 2) {
        cout << "[提示] NCCL AllReduce 需要至少 2 张 GPU 进行演示。\n";
        cout << "当前可用 GPU 数量: " << nDev << "。跳过执行。\n";
        return 0;
    }

    cout << "检测到 " << nDev << " 张 GPU，正在初始化 NCCL 环境...\n";

    // 假设每个设备处理的数据量
    const int num_elements = 1024 * 1024; 
    const size_t size_bytes = num_elements * sizeof(float);

    // 准备指针数组
    std::vector<float*> d_sendbuffs(nDev);
    std::vector<float*> d_recvbuffs(nDev);
    std::vector<cudaStream_t> streams(nDev);

    // 设置 NCCL 通信子 ID
    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));

    // 创建 nccl 实例
    std::vector<ncclComm_t> comms(nDev);

    // 为每个设备分配显存，并分组初始化 NCCL comm
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc((void**)&d_sendbuffs[i], size_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_recvbuffs[i], size_bytes));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        
        // 分别为每张卡初始化一条数据
        int threads = 256;
        int blocks = (num_elements + threads - 1) / threads;
        init_data_kernel<<<blocks, threads, 0, streams[i]>>>(d_sendbuffs[i], i, num_elements);
        
        NCCLCHECK(ncclCommInitRank(&comms[i], nDev, id, i));
    }
    NCCLCHECK(ncclGroupEnd());

    cout << "NCCL 初始化成功，开始执行 AllReduce ...\n";

    CudaTimer global_timer;
    global_timer.start();

    // 启动通信 (AllReduce = sum everything and distribute back to all)
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        // 将所有设备的 sendbuff 元素规约 (sum) 然后写入每个设备的 recvbuff 中
        NCCLCHECK(ncclAllReduce((const void*)d_sendbuffs[i], (void*)d_recvbuffs[i], 
                                num_elements, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    // 等待所有设备的流完成
    for (int i = 0; i < nDev; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    global_timer.stop();
    cout << "AllReduce 跨设备执行耗时: " << global_timer.elapsed_ms() << " ms\n";

    // 验证结果
    cout << "--- 结果验证 ---\n";
    // 如果 rank=0 -> 0.0， rank=1 -> 1.0, 那么总求和 expected_val = sum_over_i(i)
    float expected_val = 0.0f;
    for (int i = 0; i < nDev; ++i) {
        expected_val += static_cast<float>(i);
    }
    
    // 我们从卡 0 上取回数据进行验证
    CUDA_CHECK(cudaSetDevice(0));
    std::vector<float> h_recvbuff(num_elements);
    CUDA_CHECK(cudaMemcpy(h_recvbuff.data(), d_recvbuffs[0], size_bytes, cudaMemcpyDeviceToHost));
    
    bool pass = true;
    for (int i = 0; i < num_elements; ++i) {
        if (fabs(h_recvbuff[i] - expected_val) > 1e-5f) {
            cout << "✗ 验证失败: 索引 " << i << " 获取到的值为 " << h_recvbuff[i] << "，应为 " << expected_val << "\n";
            pass = false;
            break;
        }
    }
    
    if (pass) {
        cout << "✓ 全局 AllReduce 同步验证通过。所有设备归约到的结果都是 " << expected_val << "\n";
    }

    // 清理释放
    for (int i = 0; i < nDev; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(d_sendbuffs[i]));
        CUDA_CHECK(cudaFree(d_recvbuffs[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

#else
    cout << "[提示] NCCL 头文件未找到 (<nccl.h>)。\n";
    cout << "       NCCL 通常需要在 Linux 平台上通过包管理器或源代码安装。\n";
    cout << "       请安装 NCCL 后在支持的一台/多台 GPU 上重新编译测试。\n\n";
#endif

    cout << "\n========================================\n";
    return 0;
}
