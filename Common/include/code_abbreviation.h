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

using namespace std;
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