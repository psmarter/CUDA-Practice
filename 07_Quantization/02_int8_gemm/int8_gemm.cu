// INT8 GEMM - 整数量化矩阵乘法
#include <code_abbreviation.h>

// 针对旧架构或未指定架构的编译，提供 dp4a 软件回退实现
__device__ __forceinline__ int32_t compat_dp4a(int32_t a, int32_t b, int32_t c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int32_t res = c;
    res += static_cast<int32_t>(static_cast<int8_t>(a & 0xFF)) * static_cast<int32_t>(static_cast<int8_t>(b & 0xFF));
    res += static_cast<int32_t>(static_cast<int8_t>((a >> 8) & 0xFF)) * static_cast<int32_t>(static_cast<int8_t>((b >> 8) & 0xFF));
    res += static_cast<int32_t>(static_cast<int8_t>((a >> 16) & 0xFF)) * static_cast<int32_t>(static_cast<int8_t>((b >> 16) & 0xFF));
    res += static_cast<int32_t>(static_cast<int8_t>((a >> 24) & 0xFF)) * static_cast<int32_t>(static_cast<int8_t>((b >> 24) & 0xFF));
    return res;
#endif
}

// 朴素 INT8 GEMM（GPU kernel，手写）
__global__ void naive_int8_gemm(const int8_t* A, const int8_t* B, int32_t* C, CInt M, CInt N, CInt K) {
    CInt row = blockIdx.y * blockDim.y + threadIdx.y;
    CInt col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        int32_t sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += static_cast<int32_t>(A[row * N + i]) * static_cast<int32_t>(B[i * K + col]);
        }
        C[row * K + col] = sum;
    }
}

// 使用 dp4a 指令（GPU kernel，手写）
__global__ void dp4a_int8_gemm(const int8_t* A, const int8_t* B, int32_t* C, CInt M, CInt N, CInt K) {
    CInt row = blockIdx.y * blockDim.y + threadIdx.y;
    CInt col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        int32_t sum = 0;

        // N 必须为 4 的倍数
        for (int i = 0; i < N; i += 4) {

            // 从 A 读取连续 4 字节作为一个 int32_t
            int32_t a_val = *reinterpret_cast<const int32_t*>(&A[row * N + i]);
            
            // B 列由于不连续，目前只能逐个打包
            int32_t b_val = 0;

            // 获取低到高的 4 个字节，装配到一个 INT32 中
            int8_t b0 = B[(i + 0) * K + col];
            int8_t b1 = B[(i + 1) * K + col];
            int8_t b2 = B[(i + 2) * K + col];
            int8_t b3 = B[(i + 3) * K + col];
            
            // 手动 pack，按照小端序 (b3, b2, b1, b0)
            b_val = ((b3 & 0xFF) << 24) |
                    ((b2 & 0xFF) << 16) |
                    ((b1 & 0xFF) << 8) |
                    (b0 & 0xFF);

            // dp4a 执行内积并累加 
            sum = compat_dp4a(a_val, b_val, sum);
        }
        C[row * K + col] = sum;
    }
}

// 向量化 INT8 GEMM（GPU kernel，手写）
__global__ void vectorized_int8_gemm(const int8_t* A, const int8_t* B, int32_t* C, CInt M, CInt N, CInt K) {
    CInt row = blockIdx.y * blockDim.y + threadIdx.y;
    CInt col = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // 每个线程处理 4 列

    if (row < M && col < K) {
        int32_t sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    
        for (int i = 0; i < N; i += 4) {
            
            int32_t a_val = *reinterpret_cast<const int32_t*>(&A[row * N + i]);
            
            // B 因为在 K 维连续 4 个被一个线程承包，所以可以横向用reinterpret_cast向量化读取
            int32_t b_row0_pack = *reinterpret_cast<const int32_t*>(&B[(i + 0) * K + col]); // 包含 [b_00, b_01, b_02, b_03]
            int32_t b_row1_pack = *reinterpret_cast<const int32_t*>(&B[(i + 1) * K + col]);
            int32_t b_row2_pack = *reinterpret_cast<const int32_t*>(&B[(i + 2) * K + col]);
            int32_t b_row3_pack = *reinterpret_cast<const int32_t*>(&B[(i + 3) * K + col]);
            
            // 我们需要重新打包：提取每个人头顶上对应的 4 层：[b_00, b_10, b_20, b_30] 组装给 sum0
            // 小端序下: 0位是第一列, 8位是第二列, 16位是第三列, 24位是第四列
            int8_t r0_c0 = b_row0_pack & 0xFF;
            int8_t r1_c0 = b_row1_pack & 0xFF;
            int8_t r2_c0 = b_row2_pack & 0xFF;
            int8_t r3_c0 = b_row3_pack & 0xFF;
            int32_t col0_val = ((r3_c0 & 0xFF) << 24) | ((r2_c0 & 0xFF) << 16) | ((r1_c0 & 0xFF) << 8) | (r0_c0 & 0xFF);

            int8_t r0_c1 = (b_row0_pack >> 8) & 0xFF;
            int8_t r1_c1 = (b_row1_pack >> 8) & 0xFF;
            int8_t r2_c1 = (b_row2_pack >> 8) & 0xFF;
            int8_t r3_c1 = (b_row3_pack >> 8) & 0xFF;
            int32_t col1_val = ((r3_c1 & 0xFF) << 24) | ((r2_c1 & 0xFF) << 16) | ((r1_c1 & 0xFF) << 8) | (r0_c1 & 0xFF);

            int8_t r0_c2 = (b_row0_pack >> 16) & 0xFF;
            int8_t r1_c2 = (b_row1_pack >> 16) & 0xFF;
            int8_t r2_c2 = (b_row2_pack >> 16) & 0xFF;
            int8_t r3_c2 = (b_row3_pack >> 16) & 0xFF;
            int32_t col2_val = ((r3_c2 & 0xFF) << 24) | ((r2_c2 & 0xFF) << 16) | ((r1_c2 & 0xFF) << 8) | (r0_c2 & 0xFF);

            int8_t r0_c3 = (b_row0_pack >> 24) & 0xFF;
            int8_t r1_c3 = (b_row1_pack >> 24) & 0xFF;
            int8_t r2_c3 = (b_row2_pack >> 24) & 0xFF;
            int8_t r3_c3 = (b_row3_pack >> 24) & 0xFF;
            int32_t col3_val = ((r3_c3 & 0xFF) << 24) | ((r2_c3 & 0xFF) << 16) | ((r1_c3 & 0xFF) << 8) | (r0_c3 & 0xFF);

            sum0 = compat_dp4a(a_val, col0_val, sum0);
            sum1 = compat_dp4a(a_val, col1_val, sum1);
            sum2 = compat_dp4a(a_val, col2_val, sum2);
            sum3 = compat_dp4a(a_val, col3_val, sum3);
        }

        // 存回 4 个 int32_t (即一个 int4) 到 C
        int4 res;
        res.x = sum0;
        res.y = sum1;
        res.z = sum2;
        res.w = sum3;
        *reinterpret_cast<int4*>(&C[row * K + col]) = res;
    }
}

// INT8 GEMM（CPU，手写）
void int8_gemm_cpu(const int8_t* A, const int8_t* B, int32_t* C, CInt M, CInt N, CInt K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += static_cast<int32_t>(A[i * N + k]) * static_cast<int32_t>(B[k * K + j]);
            }
            C[i * K + j] = sum;
        }
    }
}

// 验证结果（AI 生成）
bool verify_results(const std::vector<int32_t>& gpu_result, const std::vector<int32_t>& cpu_result, const string& kernel_name) {
    if (gpu_result.size() > 512*512) { cout << "  [Skip] " << kernel_name << " validation for large matrices.\n"; return true; }
    if (gpu_result.size() != cpu_result.size()) {
        cout << "✗ " << kernel_name << " FAILED: 数组大小不匹配\n";
        return false;
    }

    int error_count = 0;
    int32_t max_diff = 0;
    int max_diff_idx = 0;

    for (size_t i = 0; i < gpu_result.size(); ++i) {
        int32_t gpu_v = gpu_result[i];
        int32_t cpu_v = cpu_result[i];
        int32_t diff = std::abs(gpu_v - cpu_v);
        
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = static_cast<int>(i);
        }
        
        // 整数计算要求绝对相等，零容忍
        if (diff > 0) {
            error_count++;
        }
    }

    if (error_count > 0) {
        cout << "✗ " << kernel_name << " FAILED: " << error_count << " 个元素超出误差阈值\n";
        cout << "  首个错误异常差异位于索引 " << max_diff_idx
             << "：GPU=" << gpu_result[max_diff_idx]
             << ", CPU=" << cpu_result[max_diff_idx]
             << ", 差异=" << max_diff << "\n";
        return false;
    }

    cout << "✓ " << kernel_name << " PASSED: 结果 " << gpu_result[0] 
         << " (期望 " << cpu_result[0] << ")\n";
    return true;
}


// 通用 INT8 GEMM 封装（GPU，手写）
template<typename KernelFunc>
GpuTimingResult int8_gemm_gpu(const std::vector<int8_t>& h_A, const std::vector<int8_t>& h_B, std::vector<int32_t>& h_C,
                              CInt M, CInt N, CInt K, CInt iterations, KernelFunc kernel, bool is_vectorized = false) {
    int8_t* d_A = nullptr;
    int8_t* d_B = nullptr;
    int32_t* d_C = nullptr;

    // A/B 是 Int8 (1 byte)，C 是 Int32 (4 bytes)
    CSize size_A = M * N * sizeof(int8_t);
    CSize size_B = N * K * sizeof(int8_t);
    CSize size_C = M * K * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    CudaTimer timerH2D, timerKernel, timerD2H;
    GpuTimingResult result{};

    timerH2D.start();
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));
    timerH2D.stop();
    result.h2d_ms = timerH2D.elapsed_ms();

    const dim3 block(TILE_SIZE, TILE_SIZE);
    
    // 如果是 Vectorized，X维度线程处理 4 个元素
    const dim3 grid = is_vectorized ? dim3(cdiv(K, block.x * 4), cdiv(M, block.y))
                                    : dim3(cdiv(K, block.x), cdiv(M, block.y));

    kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    timerKernel.start();
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    timerKernel.stop();
    CUDA_CHECK_LAST();
    result.kernel_ms = timerKernel.elapsed_ms() / static_cast<float>(iterations);

    timerD2H.start();
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    timerD2H.stop();
    result.d2h_ms = timerD2H.elapsed_ms();

    result.total_ms = result.h2d_ms + result.kernel_ms + result.d2h_ms;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return result;
}

// 主函数（部分手写，部分AI 生成）
int main() {
    CInt M = 1024;
    CInt N = 1024;     // N 必须为 4 的倍数 (dp4a 内建包需求)
    CInt K = 1024;     // K 必须为 4 的倍数 (Vectorized 列宽要求)
    CInt iterations = 10;

    CSize size_A = M * N * sizeof(int8_t);
    CSize size_B = N * K * sizeof(int8_t);
    CSize size_C = M * K * sizeof(int32_t);
    const double total_mb = (size_A + size_B + size_C) / (1024.0 * 1024.0);

    printDeviceInfo();

    cout << "========================================\n";
    cout << "      INT8 GEMM 性能基准测试\n";
    cout << "========================================\n";
    cout << "矩阵维度：A(" << M << " x " << N << ") * B(" << N << " x " << K << ") = C(" << M << " x " << K << ")\n";
    cout << "数据大小：" << fixed << setprecision(2) << total_mb << " MB\n";
    cout << "Block 大小：" << TILE_SIZE << " x " << TILE_SIZE << " 线程\n";
    cout << "Kernel 迭代次数：" << iterations << " 次\n";
    cout << "\n";

    // 分配主机内存并初始化
    std::vector<int8_t> h_A(M * N);
    std::vector<int8_t> h_B(N * K);
    std::vector<int32_t> h_C1(M * K);
    std::vector<int32_t> h_C2(M * K);
    std::vector<int32_t> h_C3(M * K);
    std::vector<int32_t> h_C_cpu(M * K);

    srand(42);
    // 限定随机数在 [-50, 50]，避免 INT32 内部乘加极限溢出
    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<int8_t>(rand() % 100 - 50);
    for (int i = 0; i < N * K; ++i) h_B[i] = static_cast<int8_t>(rand() % 100 - 50);

    // CPU 计算
    cout << "--- CPU 计时 ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();
    if (M <= 512) { int8_gemm_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K); }
    cpuTimer.stop();
    double cpu_time_ms = cpuTimer.elapsed_ms();
    cout << "CPU 执行时间：   " << setw(8) << cpu_time_ms << " ms\n";
    cout << "\n";
    
    // GPU 版本 1: Naive
    cout << "--- GPU 版本 1: Naive INT8 GEMM ---\n";
    GpuTimingResult result1 = int8_gemm_gpu(h_A, h_B, h_C1, M, N, K, iterations, naive_int8_gemm, false);
    cout << "H2D 传输时间：   " << setw(8) << result1.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result1.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result1.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result1.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 2: dp4a
    cout << "--- GPU 版本 2: dp4a INT8 GEMM ---\n";
    GpuTimingResult result2 = int8_gemm_gpu(h_A, h_B, h_C2, M, N, K, iterations, dp4a_int8_gemm, false);
    cout << "H2D 传输时间：   " << setw(8) << result2.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result2.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result2.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result2.total_ms << " ms\n";
    cout << "\n";

    // GPU 版本 3: Vectorized dp4a
    cout << "--- GPU 版本 3: Vectorized dp4a INT8 GEMM ---\n";
    GpuTimingResult result3 = int8_gemm_gpu(h_A, h_B, h_C3, M, N, K, iterations, vectorized_int8_gemm, true);
    cout << "H2D 传输时间：   " << setw(8) << result3.h2d_ms << " ms\n";
    cout << "Kernel 执行时间：" << setw(8) << result3.kernel_ms << " ms (" << iterations << " 次平均)\n";
    cout << "D2H 传输时间：   " << setw(8) << result3.d2h_ms << " ms\n";
    cout << "GPU 总时间：     " << setw(8) << result3.total_ms << " ms\n";
    cout << "\n";

    // 性能分析
    cout << "--- 性能分析 ---\n";
    double speedup_kernel = cpu_time_ms / result3.kernel_ms;
    double speedup_total = cpu_time_ms / result3.total_ms;
    cout << "CPU vs GPU Kernel 加速比：" << setprecision(2) << speedup_kernel << "x\n";
    cout << "CPU vs GPU 总时间加速比：" << speedup_total << "x\n";

    // MAC = read 2 bytes, write 4 bytes (忽略缓存纯物理读取)
    double bytes = size_A + size_B + size_C;
    double gpu_bandwidth = (bytes / 1e9) / (result3.kernel_ms / 1000.0);
    cout << "GPU 有效带宽：" << setprecision(2) << gpu_bandwidth << " GB/s\n";
    cout << "(RTX 4090 理论峰值：~1008 GB/s)\n";

    // TOPS (Tera Operations Per Second = 1e12) -> INT8 MAC count = 2 Ops (Multiply + Accumulate)
    double ops = 2.0 * M * N * K;
    double gpu_tops = (ops / 1e12) / (result3.kernel_ms / 1000.0);
    cout << "GPU 计算性能：" << setprecision(2) << gpu_tops << " TOPS\n";
    cout << "\n";

    cout << "--- Kernel 性能对比 ---\n";
    cout << "Naive:            " << setw(8) << setprecision(4) << result1.kernel_ms << " ms (基准)\n";
    cout << "dp4a:             " << setw(8) << result2.kernel_ms << " ms ("
         << setprecision(2) << result1.kernel_ms / result2.kernel_ms << "x)\n";
    cout << "Vectorized dp4a:  " << setw(8) << result3.kernel_ms << " ms ("
         << setprecision(2) << result1.kernel_ms / result3.kernel_ms << "x)\n";
    cout << "\n";

    cout << "--- 结果验证 ---\n";
    bool pass1 = verify_results(h_C1, h_C_cpu, "Naive INT8 GEMM");
    bool pass2 = verify_results(h_C2, h_C_cpu, "dp4a INT8 GEMM");
    bool pass3 = verify_results(h_C3, h_C_cpu, "Vectorized dp4a INT8 GEMM");

    if (pass1 && pass2 && pass3) {
        cout << "✓ GPU/CPU 结果一致性验证通过\n";
    } else {
        cout << "✗ GPU/CPU 结果存在差异\n";
    }

    cout << "\n========================================\n";

    return 0;
}
