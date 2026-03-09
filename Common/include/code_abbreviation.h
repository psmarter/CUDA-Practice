#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "cuda_utils.cuh"
#include "timer.cuh"

#define BR 32
#define BC 32
#define MAX_HEAD_DIM 128
#define D_TILE 32 
#define WARPS_PER_BLOCK 4 

// 注意：不在头文件中使用 using namespace std，避免命名空间污染
using std::vector;
using std::cout;
using std::endl;
using std::string;
using std::min;
using std::max;
using std::fixed;
using std::setprecision;
using std::setw;
using CInt = const int;
using CFloat = const float;
using CSize = const size_t;
using PFloat = float*;
using RFloat = float&;
using CPFloat = const float*;
using Matrix = vector<float>;
using RMatrix = vector<float>&;
using CRMatrix = const vector<float>&;

constexpr int BLOCK_SIZE = 1024;
constexpr int BLOCK_SIZE_1D = 256;
constexpr size_t FSIZE = sizeof(float);
constexpr int COARSE_FACTOR = 4;
constexpr int TILE_SIZE = 32;
constexpr int COARSE_X = 4;
constexpr int COARSE_Y = 4;
constexpr float EPSILON = 1e-5f;

// WMMA 常规分块大小 (16x16x16)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARPS_PER_BLOCK_Y = 8; // 一个 Block 中定义 8 个 Warp

constexpr int ASYNC_BLOCK = 256;
constexpr int ASYNC_TILE = 256;
constexpr int STAGES = 3;

// GPU 计时结果结构体（统一定义，避免各 .cu 文件重复定义）
struct GpuTimingResult {
    float h2d_ms;      // Host to Device 传输时间
    float kernel_ms;   // Kernel 执行时间（多次平均）
    float d2h_ms;      // Device to Host 传输时间
    float total_ms;    // 总时间
};