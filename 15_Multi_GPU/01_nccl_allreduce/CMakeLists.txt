# Multi-GPU NCCL 示例
add_executable(nccl_allreduce nccl_allreduce.cu)

find_package(CUDA REQUIRED)

# 尝试寻找 NCCL
find_path(NCCL_INCLUDE_DIR NAMES nccl.h)
find_library(NCCL_LIBRARY NAMES nccl)

if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    target_include_directories(nccl_allreduce PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(nccl_allreduce PRIVATE ${NCCL_LIBRARY} ${CUDA_LIBRARIES})
    message(STATUS "NCCL found at: ${NCCL_LIBRARY}")
else()
    message(WARNING "NCCL library or header not found. 
    nccl_allreduce may fail to compile or link. 
    Make sure NCCL is installed and available in environment paths.")
    # 即使没找到也尝试默认编译和链接（依靠系统默认路径），用户可自行解决环境问题
    target_link_libraries(nccl_allreduce PRIVATE nccl)
endif()
