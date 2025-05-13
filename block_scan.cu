// inclusive scan

#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <stdlib.h>

#define WARP_SIZE 32
// #define DEBUG_PRINT
template <typename T, int VectorSize, int BlockSize>
__global__ void block_scan_inclusive(const T* scan_list,
                                     T* result_list,
                                     size_t num_elements) {
        int32_t warp_id = threadIdx.x / WARP_SIZE;
        int32_t lane_id = threadIdx.x % WARP_SIZE;
        size_t linear_id = blockIdx.x * BlockSize + threadIdx.x;
        T block_loop_offset = 0;
        __shared__ T shared_data[BlockSize / WARP_SIZE];
        for(size_t element_idx = linear_id * VectorSize; element_idx < num_elements; element_idx += BlockSize * VectorSize) {
            int4 thread_values = *reinterpret_cast<const int4*>(scan_list + element_idx);
            // thread sum in thread_value.w
            thread_values.y += thread_values.x;
            thread_values.z += thread_values.y;
            thread_values.w += thread_values.z;
            T thread_sum = thread_values.w;
            T thread_offset = thread_sum;
            // #ifdef DEBUG_PRINT
            // {
            //     printf("thread_id:%d, v0:%d, v1:%d, v2:%d, v3:%d, thread_sum:%d, thread_offset:%d\n", 
            //             threadIdx.x, thread_values.x, thread_values.y, thread_values.z, thread_values.w, thread_sum, thread_offset);
            // }
            // #endif

            // warp scan
            // warp sum in last lane 
            #pragma unroll
            for(int shfl_size = 1; shfl_size < WARP_SIZE; shfl_size *= 2) {
                int n = __shfl_up_sync(0xFFFFFFFF, thread_offset, shfl_size);
                if(lane_id >= shfl_size) {
                    thread_offset += n;
                }
            }
            #ifdef DEBUG_PRINT
            {
                printf("thread_id:%d, v0:%d, v1:%d, v2:%d, v3:%d, thread_sum:%d, thread_offset:%d\n", 
                        threadIdx.x, thread_values.x, thread_values.y, thread_values.z, thread_values.w, thread_sum, thread_offset);
            }
            #endif
            // add thread offset in warp
            thread_values.x += thread_offset - thread_sum;
            thread_values.y += thread_offset - thread_sum;
            thread_values.z += thread_offset - thread_sum;
            thread_values.w += thread_offset - thread_sum;

            // save warp sum into shared memory
            if(lane_id == WARP_SIZE - 1) {
                shared_data[warp_id] = thread_offset;
            }
            __syncthreads();
            if(warp_id == 0){
                // warp 0 do warp sum scan
                int tmp_warp_sum = 0;
                if(lane_id < BlockSize / WARP_SIZE) {
                    tmp_warp_sum = shared_data[lane_id];
                }
                #pragma unroll
                for(int shfl_size = 1; shfl_size < BlockSize / WARP_SIZE; shfl_size *= 2) {
                    int n = __shfl_up_sync(0xFFFFFFFF, tmp_warp_sum, shfl_size);
                    if(lane_id >= shfl_size) {
                        tmp_warp_sum += n;
                    }
                }
                if(lane_id < BlockSize / WARP_SIZE) {
                    shared_data[lane_id] = tmp_warp_sum;
                }
            }
            __syncthreads();
            T warp_offset = 0;
            if (warp_id>0) {
                warp_offset = shared_data[warp_id - 1];
            };
            // add warp offset in block
            thread_values.x += warp_offset + block_loop_offset;
            thread_values.y += warp_offset + block_loop_offset;
            thread_values.z += warp_offset + block_loop_offset;
            thread_values.w += warp_offset + block_loop_offset;
            // write back to global memory
            *reinterpret_cast<int4*>(result_list + element_idx) = thread_values;

            // add block sum to loop offset
            block_loop_offset += shared_data[BlockSize / WARP_SIZE - 1];
        }
    }
int main(){
    const int32_t max_num_of_scan_list = 4096;
    std::vector<int32_t> scan_list(max_num_of_scan_list);
    std::vector<int32_t> result(max_num_of_scan_list);
    std::vector<int32_t> result_cpu(max_num_of_scan_list);
    // init scan list
    for(int i = 0; i < max_num_of_scan_list; ++i){
        scan_list[i] = rand() % 100;
    }
    // calculate cpu result as baseline;
    for(int i = 0; i < max_num_of_scan_list; ++i){
        if(i == 0) {
            result_cpu[i] = scan_list[0];
        } else {
            result_cpu[i] = result_cpu[i - 1] + scan_list[i];
        }
    }
    void* scan_list_device; 
    void* result_device;

    cudaMalloc(&scan_list_device, max_num_of_scan_list*sizeof(int32_t));
    cudaMalloc(&result_device, max_num_of_scan_list*sizeof(int32_t));
    cudaMemcpy(scan_list_device, scan_list.data(), max_num_of_scan_list*sizeof(int32_t), cudaMemcpyHostToDevice);
    constexpr int32_t VecSize = 128 / 8 / sizeof(int32_t);
    constexpr int32_t BlockSize = 128;
    block_scan_inclusive<int32_t, VecSize, BlockSize><<<1, BlockSize>>>(reinterpret_cast<const int32_t*>(scan_list_device), reinterpret_cast<int32_t*>(result_device), max_num_of_scan_list);
    auto error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    } else {
        printf("success\n");
    }

    cudaMemcpy(result.data(), result_device, max_num_of_scan_list*sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(scan_list_device);
    cudaFree(result_device);
    // check result
    for(int i = 0; i < max_num_of_scan_list; ++i){
        if(result[i] != result_cpu[i]){
            printf("error at %d, expect %d, but get %d\n", i, result_cpu[i], result[i]);
            break;
        }
    }
    printf("result check pass\n");
    return 0;
}