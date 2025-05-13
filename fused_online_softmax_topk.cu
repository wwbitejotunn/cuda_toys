#include<vector>
#include<array>
#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include<omp.h>
#include<cmath>
#include<stdio.h>
#include<algorithm>
#include<functional>
#include<iostream>
// #define KERNEL_DEBUG
// #undef KERNEL_DEBUG
template<typename T>
__device__ __forceinline__ void my_swap(T& a, T& b) {
    T tmp;
    tmp = a;
    a = b;
    b = tmp;
    // a = a ^ b;
    // b = a ^ b;
    // a = a ^ b;
}


template<typename T, typename U, int VectorSize, int BlockSize, bool ComputeTopK, int TopK>
__global__ void fused_softmax_topk(
    const T* input,
    T* output,
    int32_t* output_topk_idx,
    T* output_topk_value,
    size_t row_num,
    size_t col_num,
    size_t element_num
){
    std::array<T, VectorSize> thread_values;
    int4* thread_values_ptr = reinterpret_cast<int4*>(thread_values.data());
    std::array<T, VectorSize> output_values;
    int4* output_values_ptr = reinterpret_cast<int4*>(output_values.data());
    extern __shared__ T shared_buffer[];
    int4* shared_buffer_vec_ptr = reinterpret_cast<int4*>(shared_buffer);
    __shared__ U shared_buffer_block_reduce[BlockSize / 32];
    __shared__ U shared_buffer_block_reduce_indecs[BlockSize / 32];
    __shared__ U shared_buffer_block_reduce_thread_id[BlockSize / 32];

    U thread_max = -std::numeric_limits<U>::max();
    U thread_exp_sum = 0.0f;

    // topk regs pair
    // [TODO] using heap for topk value
    std::array<T, TopK> topk_values;
    std::array<int32_t, TopK> topk_indices;
    if constexpr (ComputeTopK){
        #pragma unroll
        for(int topk_idx = 0; topk_idx < TopK; topk_idx++){
            topk_values[topk_idx] = static_cast<T>(0.0);
            topk_indices[topk_idx] = -1;
        }
    }

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    for(int idx = threadIdx.x * VectorSize; idx < col_num; idx += BlockSize * VectorSize){
        *thread_values_ptr = *reinterpret_cast<const int4*>(input + blockIdx.x * col_num + idx);
        *reinterpret_cast<int4*>(shared_buffer + idx) = *thread_values_ptr;
        #pragma unroll
        for(int vector_idx = 0; vector_idx < VectorSize; vector_idx++){
            U tmp = thread_values[vector_idx];
            if(tmp > thread_max){
                thread_exp_sum = thread_exp_sum * __expf(thread_max - tmp);
                thread_exp_sum = thread_exp_sum + 1.0f; // thread_value is thread_max, exp(0) = 1
                thread_max = tmp;
            } else {
                thread_exp_sum = thread_exp_sum + __expf(tmp - thread_max);
            }
        }
    }
    #ifdef KERNEL_DEBUG
        {
            if(blockIdx.x == 0 && threadIdx.x == 0){
                printf("111 thread_max: %f, thread_exp_sum: %f\n", thread_max, thread_exp_sum);
            }
        }
    #endif
    U warp_max = thread_max;
    #pragma unroll
    for(int shfl_mask = 16; shfl_mask > 0; shfl_mask /= 2){
        warp_max = max(warp_max, __shfl_xor_sync(0xFFFFFFFF, warp_max, shfl_mask, 32));
    }
    if(lane_id == 0){
        shared_buffer_block_reduce[warp_id] = warp_max;
    }
    __syncthreads();
    if(warp_id == 0){
        U tmp = -std::numeric_limits<U>::max();
        if(lane_id < BlockSize / 32){
            tmp = shared_buffer_block_reduce[lane_id];
        }
        #pragma unroll
        for(int shfl_mask = 16; shfl_mask > 0; shfl_mask /= 2){
            tmp = max(tmp, __shfl_xor_sync(0xFFFFFFFF, tmp, shfl_mask, 32));
        }
        if(lane_id == 0){
            shared_buffer_block_reduce[0] = tmp;
        }
    }
    __syncthreads();
    U global_max = shared_buffer_block_reduce[0];

    thread_exp_sum = thread_exp_sum * __expf(thread_max - global_max);

    U warp_exp_sum = thread_exp_sum;
    #pragma unroll
    for(int shfl_mask = 16; shfl_mask > 0; shfl_mask /= 2){
        warp_exp_sum += __shfl_xor_sync(0xFFFFFFFF, warp_exp_sum, shfl_mask, 32);
    }

    if(lane_id == 0){
        shared_buffer_block_reduce[warp_id] = warp_exp_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        U tmp = 0;
        if(lane_id < BlockSize / 32){
            tmp = shared_buffer_block_reduce[lane_id];
        }
        #pragma unroll
        for(int shfl_mask = 16; shfl_mask > 0; shfl_mask /= 2){
            tmp += __shfl_xor_sync(0xFFFFFFFF, tmp, shfl_mask, 32);
        }
        if(lane_id == 0){
            shared_buffer_block_reduce[0] = tmp;
        }
    }
    __syncthreads();
    U block_exp_sum = shared_buffer_block_reduce[0];
    #ifdef KERNEL_DEBUG
        {
            if(blockIdx.x == 0 && threadIdx.x == 0){
                printf("111 block_exp_sum: %f\n", block_exp_sum);
            }
        }
    #endif
    #pragma unroll
    for(int idx = threadIdx.x * VectorSize; idx < col_num; idx += BlockSize * VectorSize){
        *thread_values_ptr = *reinterpret_cast<int4*>(shared_buffer + idx);
        #pragma unroll
        for(int vector_idx = 0; vector_idx < VectorSize; vector_idx++){
            // {
            //     if(blockIdx.x == 0 && threadIdx.x == 0){
            //         printf("111 thread_values[%d]: %f\n", vector_idx, float(thread_values[vector_idx]));
            //     }
            // }
            output_values[vector_idx] = static_cast<T>(__expf(static_cast<U>(thread_values[vector_idx]) - global_max) / block_exp_sum);
            if constexpr (ComputeTopK){
                // topk_values[TopK] = static_cast<float>(output_values[vector_idx]);
                // topk_indices[TopK] = idx + vector_idx;
                // #pragma unroll
                // for(int topk_idx = TopK - 1; topk_idx >= 0; topk_idx--){
                //     if(topk_values[topk_idx] < topk_values[topk_idx + 1]){
                //         my_swap<float>(topk_values[topk_idx], topk_values[topk_idx + 1]);
                //         my_swap<int32_t>(topk_indices[topk_idx], topk_indices[topk_idx + 1]);
                //     } else {
                //         break;
                //     }
                // }

                int replace_topk_id = -1;
                T replace_topk_values = static_cast<T>(1.0f);
                int replace_topk_indeces = -1;
                #pragma unroll
                for(int topk_idx = 0; topk_idx < TopK; topk_idx++){
                    if(topk_values[topk_idx] < output_values[vector_idx] && topk_values[topk_idx] < replace_topk_values){
                        replace_topk_values = topk_values[topk_idx];
                        replace_topk_id = topk_idx;
                    }
                }
                if(replace_topk_id != -1){
                    topk_values[replace_topk_id] = output_values[vector_idx];
                    topk_indices[replace_topk_id] = idx + vector_idx;
                }
            }
            #ifdef KERNEL_DEBUG
            {
                if(threadIdx.x==0){
                printf("### topk %d %d topk_values:%f->%f, %d->%d\n", blockIdx.x, threadIdx.x, 
                        static_cast<float>(topk_values[0]),
                        static_cast<float>(topk_values[7]),
                        topk_indices[0],
                        topk_indices[7]
                    );
                }
            }
            #endif
        }

        // #pragma unroll
        // for(int vector_idx = 0; vector_idx < VectorSize; vector_idx++){
        //     output[blockIdx.x * col_num + idx + vector_idx] = output_values[vector_idx];
        // }
        *reinterpret_cast<int4*>(output + blockIdx.x * col_num + idx) = *output_values_ptr;
        // *reinterpret_cast<int4*>(output + blockIdx.x * col_num + idx) = *output_values_ptr;
    }
    
    if constexpr (ComputeTopK){
        // warp topk
        // block_reduce max version
        __syncthreads();
        // sort 
        #pragma unroll
        for(int topk_idx = TopK - 2; topk_idx >= 0; topk_idx--){
            if(topk_values[topk_idx] < topk_values[topk_idx + 1]){
                my_swap<T>(topk_values[topk_idx], topk_values[topk_idx + 1]);
                my_swap<int32_t>(topk_indices[topk_idx], topk_indices[topk_idx + 1]);
            }
        }

        //
        int32_t local_topk_max_idx = 0;
        for(int global_topk_id = 0; global_topk_id < TopK; global_topk_id++){
            T max_this_time = topk_values[local_topk_max_idx];
            int local_max_indecs = topk_indices[local_topk_max_idx];

            int32_t max_thread_id = threadIdx.x;
            #pragma unroll
            for(int warp_mask = 32 / 2; warp_mask > 0; warp_mask /=2){
                T shfl_value = __shfl_xor_sync(0xFFFFFFFF, max_this_time, warp_mask, 32);
                int32_t shfl_indecs = __shfl_xor_sync(0xFFFFFFFF, local_max_indecs, warp_mask, 32);
                int32_t shlf_thread_id = __shfl_xor_sync(0xFFFFFFFF, max_thread_id, warp_mask, 32);
                if(shfl_value > max_this_time) {
                    max_this_time = shfl_value;
                    local_max_indecs = shfl_indecs;
                    max_thread_id = shlf_thread_id;
                }
            }
            if(lane_id == 0){
                shared_buffer_block_reduce[warp_id] = max_this_time;
                shared_buffer_block_reduce_indecs[warp_id] = local_max_indecs;
                shared_buffer_block_reduce_thread_id[warp_id] = max_thread_id;
            }
            __syncthreads();
            if(warp_id == 0){
                T block_max = 0.0;
                int32_t block_max_indecs = 0;
                int32_t block_max_thread_id = 0;
                if(lane_id < BlockSize / 32){
                    block_max = shared_buffer_block_reduce[lane_id];
                    block_max_indecs = shared_buffer_block_reduce_indecs[lane_id];
                    block_max_thread_id = shared_buffer_block_reduce_thread_id[lane_id];
                }
    
    
                #pragma unroll
                for(int warp_mask = 32 / 2; warp_mask > 0; warp_mask /=2){
                    T shfl_value = __shfl_xor_sync(0xFFFFFFFF, block_max, warp_mask, 32);
                    int32_t shfl_indecs = __shfl_xor_sync(0xFFFFFFFF, block_max_indecs, warp_mask, 32);
                    int32_t shlf_thread_id = __shfl_xor_sync(0xFFFFFFFF, block_max_thread_id, warp_mask, 32);
                    if(shfl_value > block_max) {
                        block_max = shfl_value;
                        block_max_indecs = shfl_indecs;
                        block_max_thread_id = shlf_thread_id;
                    }
                }
                // {
                //     if(blockIdx.x==0 && lane_id == 0){
                //         printf(" 222 global_topk_id:%d max_this_time:%f, local_max_indecs:%d, max_thread_id:%d\n", 
                //         global_topk_id, static_cast<float>(block_max), block_max_indecs, block_max_thread_id);
                //     }
                // }

                if(lane_id == 0){
                    // shared_buffer_block_reduce[lane_id] = block_max;
                    // shared_buffer_block_reduce_indecs[lane_id] = block_max_indecs;
                    shared_buffer_block_reduce_thread_id[lane_id] = block_max_thread_id;
                    output_topk_idx[blockIdx.x * TopK + global_topk_id] = block_max_indecs;
                    output_topk_value[blockIdx.x * TopK + global_topk_id] = block_max;
                }
            }
            __syncthreads();
            int32_t global_max_thread_id = shared_buffer_block_reduce_thread_id[0];
            
            if(threadIdx.x == global_max_thread_id){
                topk_values[local_topk_max_idx] = -1.0f;
                local_topk_max_idx++;
            }

        }
    }
}
template <typename T>
struct value_indice_pair {
    T value = 0.0;
    int32_t indice = -1;
    value_indice_pair(){}
    value_indice_pair(T value_ , int32_t indice_) : value(value_), indice(indice_) {}
 
};
template<typename T>
bool operator< (const value_indice_pair<T>& x, const value_indice_pair<T>& y){
    if (x.value < y.value) {
        return true;
    } else {
        return false;
    }
}
template<typename T>
bool operator> (const value_indice_pair<T>& x, const value_indice_pair<T>& y){
    if (x.value > y.value) {
        return true;
    } else {
        return false;
    }
}
template <bool ComputeTopK=false, int TopK=8>
void naive_softmax_topk(
    const float* input,
    float* output,
    int32_t* output_topk_indices,
    float* output_topk_value,
    size_t row_num,
    size_t col_num,
    size_t element_num
) {
    #pragma omp parallel for
    for(size_t i = 0; i < row_num; i++) {
        float max_val = -std::numeric_limits<float>::max();
        std::array<value_indice_pair<float>, TopK> topk_values;
        make_heap(topk_values.begin(), topk_values.end(), std::greater<value_indice_pair<float>>());
        for(size_t j = 0; j < col_num; j++) {
            max_val = fmaxf(max_val, input[i * col_num + j]);
        }
        float exp_sum = 0.0f;
        for(size_t j = 0; j < col_num; j++) {
            exp_sum += expf(input[i * col_num + j] - max_val);
        }
        for(size_t j = 0; j < col_num; j++) {
            output[i * col_num + j] = expf(input[i * col_num + j] - max_val) / exp_sum;
            if constexpr (ComputeTopK) {
                if(output[i * col_num + j] > topk_values[0].value){
                    pop_heap(topk_values.begin(), topk_values.end(), std::greater<value_indice_pair<float>>());
                    topk_values[TopK - 1].value = output[i * col_num + j];
                    topk_values[TopK - 1].indice = j;
                    push_heap(topk_values.begin(), topk_values.end(), std::greater<value_indice_pair<float>>());
                }
            }
        }
        if constexpr (ComputeTopK) {
            for(int topk_i = TopK-1; topk_i >=0 ; topk_i--){
                auto heap_min = topk_values[0];
                pop_heap(topk_values.begin(), topk_values.end(), std::greater<value_indice_pair<float>>());
                topk_values[TopK - 1].value = 2;
                output_topk_indices[i*TopK + topk_i] = heap_min.indice;
                output_topk_value[i*TopK + topk_i] = heap_min.value;
            }
        }
    }
}
int main(){
    constexpr size_t row_num = 8192;
    constexpr size_t col_num = 8192;
    constexpr size_t element_num = row_num * col_num;
    constexpr int block_size = 256;
    constexpr int topk_num = 8;
    std::vector<float> input_float(element_num);
    std::vector<__half> input_half(element_num);
    std::vector<__half> output(element_num);
    std::vector<float> output_ref(element_num);

    std::vector<__half> output_topk_value(row_num * topk_num);
    std::vector<int32_t> output_topk_index(row_num * topk_num);

    std::vector<float> output_topk_value_ref(row_num * topk_num);
    std::vector<int32_t> output_topk_index_ref(row_num * topk_num);

    
    #pragma omp parallel for
    for(int i = 0; i < element_num; i++){
        input_float[i] = static_cast<float>(rand()) / RAND_MAX;
        input_half[i] = __float2half(input_float[i]);
        input_float[i] = __half2float(input_half[i]);
        // if(i / col_num == 0){
        //     printf("input[%d]: %f", i, input_float[i]);
        // }
    }
    void * input_half_device;
    void * output_device;
    void * output_topk_value_device;
    void * output_topk_index_device;
    cudaMalloc(&input_half_device, element_num * sizeof(__half));
    cudaMalloc(&output_device, element_num * sizeof(__half));

    cudaMalloc(&output_topk_value_device, row_num * topk_num * sizeof(__half));
    cudaMalloc(&output_topk_index_device, row_num * topk_num * sizeof(int32_t));


    cudaMemcpy(input_half_device, input_half.data(), element_num * sizeof(__half), cudaMemcpyHostToDevice);
    constexpr int vector_size = 128 / 8 / sizeof(__half);

    int share_mem_size = col_num * sizeof(__half); 
    share_mem_size = (share_mem_size + 3) / 4 * 4; // Align to 4 bytes
    printf("Shared memory size: %d\n", share_mem_size);
    // cudaFuncSetAttribute(&fused_softmax_topk, cudaFuncAttributeMaxDynamicSharedMemorySize, share_mem_size);

    fused_softmax_topk<__half, 
                       float, 
                       vector_size, 
                       block_size,
                       false,
                       topk_num><<<row_num, block_size, share_mem_size, 0>>>(
            reinterpret_cast<const __half*>(input_half_device), 
            reinterpret_cast<__half*>(output_device), 
            nullptr,
            nullptr,
            row_num, col_num, element_num);
    auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    } else {
        printf("CUDA kernel executed successfully.\n");
    }

    naive_softmax_topk<true,8>(
            input_float.data(), 
            output_ref.data(), 
            output_topk_index_ref.data(),
            output_topk_value_ref.data(),
            row_num, col_num, element_num);
    cudaMemcpy(output.data(), output_device, element_num * sizeof(__half), cudaMemcpyDeviceToHost);
    bool check = true;
    #pragma omp parallel for
    for(size_t i = 0; i < element_num; i++){
        size_t row = i / col_num;
        size_t col = i % col_num;
        float output_val = __half2float(output[i]);
        if(fabs(output_val-output_ref[i]) > 1e-6){
            printf("Mismatch at index [%d,%d]: GPU: %f, CPU: %f\n", row, col, output_val, output_ref[i]);
            check = false;
            // break; 
        }
    }
    if(check){
        printf("All values match!\n");
    } else {
        printf("Mismatch found!\n");
    }


 
    fused_softmax_topk<__half, 
                    float, 
                    vector_size, 
                    block_size,
                    true,
                    topk_num><<<row_num, block_size, share_mem_size, 0>>>(
        reinterpret_cast<const __half*>(input_half_device), 
        reinterpret_cast<__half*>(output_device), 
        reinterpret_cast<int32_t*>(output_topk_index_device),
        reinterpret_cast<__half*>(output_topk_value_device),
        row_num, col_num, element_num);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    } else {
        printf("CUDA kernel executed successfully.\n");
    }   
    cudaMemcpy(output.data(), output_device, element_num * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_topk_value.data(), output_topk_value_device, row_num * topk_num * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_topk_index.data(), output_topk_index_device, row_num * topk_num * sizeof(int32_t), cudaMemcpyDeviceToHost);
    check = true;
    #pragma omp parallel for
    for(size_t i = 0; i < element_num; i++){
        size_t row = i / col_num;
        size_t col = i % col_num;
        float output_val = __half2float(output[i]);
        if(fabs(output_val-output_ref[i]) > 1e-6){
            printf("Mismatch at index [%d,%d]: GPU: %f, CPU: %f\n", row, col, output_val, output_ref[i]);
            check = false;
            // break; 
        }
    }
    for(size_t i=0;i<row_num;i++){
        printf("device: ");
        for(int topk_i = 0; topk_i < topk_num; topk_i++){
            printf("[%d]:%f | ", output_topk_index[i*topk_num + topk_i], __half2float(output_topk_value[i*topk_num + topk_i]));
        }
        std::cout<<std::endl;
        printf("baseline: ");
        for(int topk_i = 0; topk_i < topk_num; topk_i++){
            printf("[%d]:%f | ", output_topk_index_ref[i*topk_num + topk_i], output_topk_value_ref[i*topk_num + topk_i]);
        }
        std::cout<<std::endl<<std::endl;
    }
    if(check){
        printf("All values match!\n");
    } else {
        printf("Mismatch found!\n");
    }


    cudaFree(input_half_device);
    cudaFree(output_device);

    return 0;
}