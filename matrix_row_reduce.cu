#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <array>
#include <omp.h>
#include <vector>
#include <cmath>
template <typename T, typename U, int VectorSize, int BlockSize>
__global__ void matrix_row_reduce(const T* a, 
                                  T* b, size_t row_num, size_t col_num, size_t element_num){
    int32_t warp_id = threadIdx.x / 32;
    int32_t lane_id = threadIdx.x % 32;
    __shared__ U shared_data[BlockSize / 32];
    std::array<T, VectorSize> thread_values;
    int4* array_ptr = reinterpret_cast<int4*>(thread_values.data());
    U thread_sum = 0;
    for(size_t col_idx=threadIdx.x * VectorSize; col_idx<row_num; col_idx+=BlockSize * VectorSize){
        size_t lieaner_idx = blockIdx.x * col_num + col_idx;
        *array_ptr = *reinterpret_cast<const int4*>(a + lieaner_idx);
        #pragma unroll
        for(int vector_idx=0; vector_idx<VectorSize; vector_idx++){
            thread_sum += static_cast<U>(thread_values[vector_idx]);
        }
    }
    #pragma unroll
    for(int warp_size = 16; warp_size > 0; warp_size /= 2){
        thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, warp_size, 32);
    }
    if(lane_id == 0){
        shared_data[warp_id] = thread_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        U block_sum = 0;
        if(lane_id < BlockSize / 32){
            block_sum = shared_data[lane_id];
        }
        #pragma unroll
        for(int shfl_size = 16; shfl_size > 0; shfl_size /= 2){
            block_sum += __shfl_xor_sync(0xFFFFFFFF, block_sum, shfl_size, 32);
        }
        if(lane_id == 0){
            b[blockIdx.x] = static_cast<T>(block_sum);
        }
    }
}

void naive_matric_row_reduce(const float* a, float* b, size_t row_num, size_t col_num){
    #pragma omp for
    for(size_t row_idx=0; row_idx<row_num; row_idx++){
        float sum = 0;
        for(size_t col_idx=0; col_idx<col_num; col_idx++){
            sum += a[row_idx * col_num + col_idx];
        }
        b[row_idx] = sum;
    }
}
int main(){
    size_t row_num = 8192;
    size_t col_num = 8192;
    constexpr int VecSize = 128/8/sizeof(__half);
    constexpr int BlockSize = 256;

    std::vector<float> a_float(row_num * col_num);
    std::vector<float> b_float(row_num);
    std::vector<__half> a_half(row_num * col_num);
    void* a_half_device;
    void* b_half_device;
    for(int i = 0; i < row_num * col_num; ++i){
        a_float[i] = static_cast<float>(rand()) / RAND_MAX;
        a_half[i] = __float2half(a_float[i]);
    }
    cudaMalloc(&a_half_device, row_num * col_num * sizeof(__half));
    cudaMalloc(&b_half_device, row_num * sizeof(__half));
    cudaMemcpy(a_half_device, a_half.data(), row_num * col_num * sizeof(__half), cudaMemcpyHostToDevice);
    matrix_row_reduce<__half, float, VecSize, BlockSize><<<row_num, BlockSize>>>(
            reinterpret_cast<const __half*>(a_half_device), 
            reinterpret_cast<__half*>(b_half_device), 
            row_num, col_num, row_num * col_num);
    cudaDeviceSynchronize();


    naive_matric_row_reduce(a_float.data(), b_float.data(), row_num, col_num);
    std::vector<__half> b_half(row_num);
    cudaMemcpy(b_half.data(), b_half_device, row_num * sizeof(__half), cudaMemcpyDeviceToHost);
    for(int i = 0; i < row_num; ++i){
        if(abs( __half2float(b_half[i]) - (b_float[i])) / b_float[i] > 0.1){
            printf("Error: %d %f %f\n", i, __half2float(b_half[i]), b_float[i]);
        } 
    } 

    return 0;
}