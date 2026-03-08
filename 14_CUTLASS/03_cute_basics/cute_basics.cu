// CuTe Basics 示例 - 演示 CUTLASS 3.x 核心抽象 CuTe
// 包含 Layout, Tensor 和基础 copy 操作
//
// 注意：编译本文件需要安装 CUTLASS 3.x 头文件库
#include <code_abbreviation.h>
#include <iostream>

#ifdef __has_include
  #if __has_include(<cute/tensor.hpp>)
    #define HAS_CUTE 1
  #else
    #define HAS_CUTE 0
  #endif
#else
  #define HAS_CUTE 0
#endif

#if HAS_CUTE
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

// ========================= CuTe 基础 Kernel =========================

// CuTe Print Kernel（GPU，手写）
__global__ void cute_print_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("--- CuTe Layout 基础演示 ---\n");
        // 定义一个 3x4 的 Layout
        auto layout_2d = make_layout(make_shape(Int<3>{}, Int<4>{}), 
                                     make_stride(Int<4>{}, Int<1>{}));
        
        printf("Layout 2D Shape: (%d, %d)\n", size<0>(layout_2d), size<1>(layout_2d));
        printf("Index(1, 2) 的一维偏移量: %d\n", layout_2d(1, 2));

        printf("\n--- CuTe 循环打印 --- \n");
        // Print 整个 Layout 的映射
        for (int i = 0; i < size(layout_2d); ++i) {
            printf("layout(%d) = %d\n", i, layout_2d(i));
        }
    }
}

// CuTe Tensor Copy Kernel (利用 CuTe 简化共享内存数据搬运)（GPU，手写）
template<class T>
__global__ void cute_copy_kernel(const T* g_in, T* g_out, int M, int N) {
    // 定义全局内存 Tensor Layout
    auto tensor_shape = make_shape(M, N);
    auto tensor_stride = make_stride(N, Int<1>{});
    
    // 使用全局指针创建 Global Tensor
    Tensor tG_in = make_tensor(make_gmem_ptr(g_in), make_layout(tensor_shape, tensor_stride));
    Tensor tG_out = make_tensor(make_gmem_ptr(g_out), make_layout(tensor_shape, tensor_stride));

    // 定义共享内存 Tensor
    // 假设用 16x16 的 Tile
    using smem_shape = Shape<Int<16>, Int<16>>;
    __shared__ T smem_data[size(smem_shape{})];
    Tensor tS = make_tensor(make_smem_ptr(smem_data), make_layout(smem_shape{}));

    // 确定当前 Block 的坐标
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    // 对全局 Tensor 进行 Tiling 划分 (将大 Tensor 切成 16x16 的瓦片)
    Tensor tG_in_tiled = local_tile(tG_in, smem_shape{}, make_coord(bidy, bidx));
    Tensor tG_out_tiled = local_tile(tG_out, smem_shape{}, make_coord(bidy, bidx));
    
    // 当前线程在 Tile 内的坐标
    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;

    if (threadIdx.x < 256) {
        // 全局 -> 共享内存
        tS(ty, tx) = tG_in_tiled(ty, tx);
        __syncthreads();

        // 共享内存 -> 全局内存 (演示基础的数据流动)
        tG_out_tiled(ty, tx) = tS(ty, tx);
    }
}

// 供 Host 端调用的验证函数（AI 生成）
bool verify_copy_results(CRMatrix gpu_result, CRMatrix cpu_result, int n) {
    for (int i = 0; i < n; ++i) {
        if (fabs(gpu_result[i] - cpu_result[i]) > 1e-5f) {
            cout << "✗ CuTe Tensor Copy FAILED: 索引 " << i 
                 << " 结果 " << gpu_result[i] << " (期望 " << cpu_result[i] << ")\n";
            return false;
        }
    }
    cout << "✓ CuTe Tensor Copy 验证通过\n";
    return true;
}

#endif // HAS_CUTE

// ========================= 主函数 =========================

int main() {
    printDeviceInfo();

    cout << "========================================\n";
    cout << "      CuTe (CUTLASS 3.x) 基础演示\n";
    cout << "========================================\n\n";

#if HAS_CUTE
    // 1. 打印 CuTe Layout 的抽象概念
    cout << ">>> 运行 CuTe Print Kernel ... <<<\n";
    cute_print_kernel<<<1, 1>>>();
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();
    
    // 2. 验证 CuTe Tensor Tiling 和 Copy 数据流转
    cout << "\n>>> 测试 CuTe Tensor Copy Kernel ... <<<\n";
    CInt M = 128; // 由于 Block 切块，M 和 N 可以是 16 的倍数
    CInt N = 128;
    CSize num_elements = M * N;

    Matrix h_in(num_elements);
    Matrix h_out(num_elements, 0.0f);
    for (int i = 0; i < num_elements; ++i) h_in[i] = static_cast<float>(i);

    PFloat d_in = nullptr, d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, num_elements * FSIZE));
    CUDA_CHECK(cudaMalloc((void**)&d_out, num_elements * FSIZE));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), num_elements * FSIZE, cudaMemcpyHostToDevice));

    // 使用 16x16 = 256 线程的 Block
    dim3 block(256);
    // Grid大小将原始矩阵划分为 16x16 的块
    dim3 grid(N / 16, M / 16);

    cute_copy_kernel<<<grid, block>>>(d_in, d_out, M, N);
    CUDA_CHECK_LAST();
    CUDA_SYNC_CHECK();

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, num_elements * FSIZE, cudaMemcpyDeviceToHost));

    verify_copy_results(h_out, h_in, num_elements);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

#else
    cout << "[提示] CuTe 头文件未找到 (<cute/tensor.hpp>)。跳过演示。\n";
    cout << "       请设置 CUTLASS_DIR 并使用 CUTLASS v3.0+ 重新编译\n\n";
#endif

    cout << "\n========================================\n";
    return 0;
}
