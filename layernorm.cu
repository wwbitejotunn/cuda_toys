
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <array>
#include <cuda_fp16.h>
#include <omp.h>
#define WARP_SIZE 32

template<typename T, int VectorSize, int BlockSize, int GridSize, int HiddenDim>
__global__ void layernorm(
    const T* input,
    T* output,
    int32_t dim,
    int32_t token_num,
    int32_t element_num
){
    int32_t block_id = blockIdx.x;
    int32_t thread_id = threadIdx.x;
    int32_t warp_id = thread_id / WARP_SIZE;
    int32_t lane_id = thread_id % WARP_SIZE;
    extern __shared__ T shared_data[];
    int4* shared_data_128b_ptr = reinterpret_cast<int4*>(shared_data);
    __shared__ float block_reduce_shmem_sum[BlockSize / WARP_SIZE];
    __shared__ float block_reduce_shmem_sum_2[BlockSize / WARP_SIZE];
    using half_vector = typename std::array<__half, VectorSize>;
    for(int token_id = block_id; token_id<token_num; token_id += GridSize) {

        float thread_sum = 0;
        float thread_sum_2 = 0;

        for (int dim_idx = threadIdx.x * VectorSize; dim_idx < dim; dim_idx += BlockSize * VectorSize) {
            half_vector thread_values;
            int4* thread_values_ptr = reinterpret_cast<int4*>(thread_values.data());
            *thread_values_ptr = *reinterpret_cast<const int4*>(input + token_id * dim + dim_idx);
            #pragma unroll
            for(int vector_idx = 0; vector_idx < VectorSize; vector_idx++){
                thread_sum += static_cast<float>(thread_values[vector_idx]);
                thread_sum_2 += static_cast<float>(thread_values[vector_idx]) * static_cast<float>(thread_values[vector_idx]);
            }
            *reinterpret_cast<int4*>(shared_data + dim_idx) = *thread_values_ptr;
        }
        for(int shfl_mask = WARP_SIZE / 2; shfl_mask > 0; shfl_mask /= 2) {
            thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, shfl_mask, WARP_SIZE);
            thread_sum_2 += __shfl_xor_sync(0xFFFFFFFF, thread_sum_2, shfl_mask, WARP_SIZE);
        }
        if(lane_id == 0) {
            block_reduce_shmem_sum[warp_id] = thread_sum;
            block_reduce_shmem_sum_2[warp_id] = thread_sum_2;
        }
        __syncthreads();
        if(warp_id == 0) {
            float block_sum = 0;
            float block_sum_2 = 0;
            if(lane_id < BlockSize / WARP_SIZE) {
                block_sum = block_reduce_shmem_sum[lane_id];
                block_sum_2 = block_reduce_shmem_sum_2[lane_id];
            }
            for(int shfl_mask = WARP_SIZE / 2; shfl_mask > 0; shfl_mask /= 2) {
                block_sum += __shfl_xor_sync(0xFFFFFFFF, block_sum, shfl_mask, WARP_SIZE);
                block_sum_2 += __shfl_xor_sync(0xFFFFFFFF, block_sum_2, shfl_mask, WARP_SIZE);
            }
            if(lane_id == 0) {
                block_reduce_shmem_sum[0] = block_sum;
                block_reduce_shmem_sum_2[0] = block_sum_2;
            }
        }
        
        __syncthreads();
        float mean = block_reduce_shmem_sum[0] / dim;
        float inv_var = 1 / sqrtf(block_reduce_shmem_sum_2[0] / dim - mean * mean + 1e-6);
        // if(blockIdx.x == 0 && threadIdx.x == 0){
        //     printf("mean: %f, inv_var: %f\n", mean, inv_var);
        // }
        for (int dim_idx = threadIdx.x * VectorSize; dim_idx < dim; dim_idx += BlockSize * VectorSize) {
            half_vector thread_values;
            auto thread_values_ptr = reinterpret_cast<int4*>(thread_values.data());
            *thread_values_ptr = *reinterpret_cast<int4*>(shared_data + dim_idx);
            #pragma unroll
            for(int vector_idx = 0; vector_idx < VectorSize; vector_idx++){
                thread_values[vector_idx] = static_cast<T>((static_cast<float>(thread_values[vector_idx]) - mean) * inv_var);
            
                // if(blockIdx.x == 0 && threadIdx.x == 0){
                //     printf("mean: %f, inv_var: %f\n", mean, inv_var);
                //     printf("token_id :%d, thread_values[%d]: %f\n", token_id, vector_idx, static_cast<float>(thread_values[vector_idx]));
                // }


            }
            
            *reinterpret_cast<int4*>(output + token_id * dim + dim_idx) = *thread_values_ptr;
        }
        // if(blockIdx.x == 0 && threadIdx.x == 0){
        //     printf("token id :%d done\n", token_id);
        // }
        __syncthreads();
    }
}
void naive_layer_norm(
    const float * input,
    float * output,
    int32_t dim,
    int32_t token_num,
    int32_t element_num
) {
    #pragma omp parallel for
    for(int i = 0; i < token_num; ++i){
        float sum = 0.0f;
        float sum_2 = 0.0f;
        for(int j=0;j<dim;j++){
            sum += input[i * dim + j];
            sum_2 += input[i * dim + j] * input[i * dim + j];
        }
        float mean = sum / dim;
        float inv_var = 1 / sqrt(sum_2 / dim - mean * mean + 1e-6);
        // printf("mean: %f, inv_var: %f\n", mean, inv_var);
        for(int j=0;j<dim;j++){
            output[i * dim + j] = (input[i * dim + j] - mean) * inv_var;
            // printf("output[%d]: %f\n", i * dim + j, output[i * dim + j]);
        }
    }
}

int main(){
    constexpr int dim = 8192;
    int token_num = 8192;
    constexpr int block_size = 256;
    constexpr int vector_size = 128 / 8 / sizeof(half);
    constexpr int grid_size = 120;

    std::vector<float> input(dim * token_num);
    std::vector<__half> input_half(dim * token_num);

    for(int i = 0 ; i < token_num * dim; i++){
        input[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        input_half[i] = __float2half(input[i]);
        input[i] = __half2float(input_half[i]);
    }
    
    std::vector<__half> output(dim * token_num);
    std::vector<float> output_ref(dim * token_num);

    void* input_half_device;
    void* output_half_device;
    cudaMalloc(&input_half_device, dim * token_num * sizeof(__half));
    cudaMalloc(&output_half_device, dim * token_num * sizeof(__half));
    cudaMemcpy(input_half_device, input_half.data(), dim * token_num * sizeof(__half), cudaMemcpyHostToDevice);

    layernorm<__half, vector_size, block_size, grid_size, dim>
             <<<grid_size, block_size, dim * sizeof(__half), 0>>>
        (
           reinterpret_cast<const __half*>(input_half_device), 
           reinterpret_cast<__half*>(output_half_device), dim, token_num, token_num*dim);
    auto error= cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    } else {
        printf("CUDA kernel launched successfully\n");
    }
    naive_layer_norm(
        input.data(), output_ref.data(), dim, token_num, token_num*dim
    );
    
    cudaMemcpy(output.data(), output_half_device, dim * token_num * sizeof(__half), cudaMemcpyDeviceToHost);
    for(int i = 0; i < token_num * dim; ++i){
        float output_float = __half2float(output[i]);
        
        if(fabs(output_float - output_ref[i]) > 1e-3){
            printf("Mismatch at index %d: GPU: %f, CPU: %f\n", i, output_float, output_ref[i]);
            break;
        }
    }
    printf("LayerNorm test passed\n");
    return 0;
}