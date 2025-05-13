
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <array>
#include <vector>
#include <omp.h>
#include <cuda_fp16.h>
#define WARPSIZE 32

static inline __device__ void st_flag_release(uint32_t &flag,  // NOLINT
                                              volatile uint32_t *flag_addr) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
#else
  __threadfence_system();
  asm volatile("st.global.volatile.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}


static inline __device__ void ld_flag_acquire(uint32_t &flag,  // NOLINT
                                              volatile uint32_t *flag_addr) {
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
#else
  asm volatile("ld.global.volatile.b32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
#endif
}

template <typename InputT, typename OutputT, int VecSize>
__global__ void outplace_cast_kernel(    
    const InputT * input_data,
    OutputT * output_data,
    size_t numel
){
    size_t linear_idx = threadIdx.x + blockIdx.x * blockDim.x;
    using InputVec = std::array<InputT, VecSize>;
    using OutputVec = std::array<OutputT, VecSize>;
    InputVec input_vec;
    OutputVec output_vec;

    int4* input_vec_data_ptr = reinterpret_cast<int4*>(input_vec.data());
    int2* output_vec_data_ptr = reinterpret_cast<int2*>(output_vec.data());

    for(int idx = linear_idx * VecSize; idx < numel; idx += blockDim.x * gridDim.x * VecSize) {
        *input_vec_data_ptr = *reinterpret_cast<const int4*>(input_data + idx);
        for(int v_i = 0; v_i < VecSize; v_i++) {
            output_vec[v_i] = static_cast<OutputT>(input_vec[v_i]);
        }
        *reinterpret_cast<int2*>(output_data + idx) = *output_vec_data_ptr;
    }
}

template <typename InputT, typename OutputT, int VecSize, int SyncSize>
__global__ void inplace_cast_kernel(
    const InputT * input_data,
    OutputT * output_data,
    int32_t * signal_place,
    size_t numel,
    size_t pad_numel
){
    int32_t block_id = blockIdx.x;
    using InputVec = std::array<InputT, VecSize>;
    using OutputVec = std::array<OutputT, VecSize>;
    size_t linear_idx = block_id * blockDim.x + threadIdx.x;
    InputVec input_vec;
    OutputVec output_vec;
    int4* input_vec_data_ptr = reinterpret_cast<int4*>(input_vec.data());
    int2* output_vec_data_ptr = reinterpret_cast<int2*>(output_vec.data());
    int32_t lane_id = threadIdx.x % WARPSIZE;
    int32_t sync_id = threadIdx.x % SyncSize;
    uint32_t flag = 0;
    for(int64_t idx = linear_idx * VecSize; idx < numel; idx += blockDim.x * VecSize) {
        if (idx >= numel) {
            continue;
        }
        *input_vec_data_ptr = *reinterpret_cast<const int4*>(input_data + idx);
        int32_t read_signal_idx = idx / VecSize / SyncSize;
        int32_t write_signal_idx = read_signal_idx / 2;
        if(sync_id == 0){
            atomicAdd(signal_place + read_signal_idx, 1);
        }
        // __syncwarp();
        __syncthreads();
        #pragma unroll
        for(int v_i = 0; v_i < VecSize; v_i++) {
            output_vec[v_i] = static_cast<OutputT>(input_vec[v_i]);
        }
        // TODO better wait
        uint32_t* write_signal_ptr = reinterpret_cast<uint32_t*>(signal_place + write_signal_idx);
        if(sync_id == 0){
            ld_flag_acquire(flag, write_signal_ptr);
            while(flag < 1){
                ld_flag_acquire(flag, write_signal_ptr);
            }
        }
        // __syncwarp();
        __syncthreads();

        *reinterpret_cast<int2*>(output_data + idx) = *output_vec_data_ptr;
    }
}

template <typename T>
__device__ __forceinline__ T div_up(T a, T b){
    return (a + b - 1) / b;
}

template <typename InputT, typename OutputT, int VecSize, int BlockDim, int BlockElementNumOnePass>
__global__ void inplace_cast_kernel_v2(
    const InputT * input_data,
    OutputT * output_data,
    int32_t * signal_place,
    size_t numel,
    size_t pad_numel
){
    using InputVec = std::array<InputT, VecSize>;
    using OutputVec = std::array<OutputT, VecSize>;
    int GridDim = gridDim.x;
    InputVec input_vec;
    OutputVec output_vec;

    int4* input_vec_data_ptr = reinterpret_cast<int4*>(input_vec.data());
    int2* output_vec_data_ptr = reinterpret_cast<int2*>(output_vec.data());


    extern __shared__ void* shm_buffer[];

    OutputT* shm_buffer_OutputT_ptr = reinterpret_cast<OutputT*>(shm_buffer);
    int32_t loop_num = int(div_up<size_t>(numel , BlockElementNumOnePass * GridDim));
    int32_t grid_dim_num_one_pass = BlockElementNumOnePass * GridDim;
    int32_t thread_num = BlockElementNumOnePass / BlockDim;
    int32_t thread_vec_num = thread_num / VecSize;
    int32_t linear_idx = threadIdx.x;

    for(size_t loop_idx = blockIdx.x * BlockElementNumOnePass; loop_idx < numel; loop_idx += BlockElementNumOnePass * GridDim){
        bool need_wait_signal = true;
        if (loop_idx >= BlockElementNumOnePass * GridDim) {
            need_wait_signal=false;   
        }
        for(size_t idx = linear_idx * VecSize; idx < BlockElementNumOnePass; idx+= BlockDim * VecSize) {
            size_t element_idx = idx + loop_idx;
            *input_vec_data_ptr = *reinterpret_cast<const int4*>(input_data + element_idx);
            // if(threadIdx.x % 32 == 0) {
            //     printf("#### block_idx :%d, warp_id:%d load token_id:%d dim:%d\n", blockIdx.x, threadIdx.x/32, element_idx / 8192 , element_idx % 8192);
            // }
            // if(threadIdx.x == 0 && idx == linear_idx * VecSize){
            //     printf("@@@@ block_idx:%d, "
            //             "loop_idx:%lld, "
            //             "token_id:%d, "
            //             "dim:%d, "
            //             "input:%f \n", 
            //             blockIdx.x, loop_idx, (int) element_idx / 8192, (int) element_idx % 8192, input_vec[0]);
            // }

            #pragma unroll
            for(int v_i = 0; v_i < VecSize; v_i++) {
                output_vec[v_i] = static_cast<OutputT>(input_vec[v_i]);
            }
            *reinterpret_cast<int2*>(shm_buffer_OutputT_ptr + idx) = *output_vec_data_ptr;
        }
        if(need_wait_signal){
            if(threadIdx.x == 0){
                atomicAdd(signal_place + loop_idx / BlockElementNumOnePass, 1);
                // printf("##### block_id: %d, loop_idx: %d, read_signal_offset: %d\n",
                //        blockIdx.x,
                //        (int32_t)loop_idx,
                //        (int32_t)(loop_idx / BlockElementNumOnePass));
            }
            if(threadIdx.x == 0){
                uint32_t flag = 0;
                // printf("##### block_id: %d, loop_idx: %d, write_signal_offset: %d\n",
                //        blockIdx.x,
                //        (int32_t)loop_idx,
                //        (int32_t)(loop_idx/ 2 / BlockElementNumOnePass));

                uint32_t* write_signal_ptr = reinterpret_cast<uint32_t*>(signal_place + loop_idx / 2 / BlockElementNumOnePass);
                ld_flag_acquire(flag, write_signal_ptr);
                while(flag < 1) {
                    ld_flag_acquire(flag, write_signal_ptr);
                }
            }
        }
        __syncthreads();
        for(size_t idx = linear_idx * VecSize * 2; idx<BlockElementNumOnePass; idx+= BlockDim * VecSize * 2) {
            size_t element_idx = idx + loop_idx;
             *reinterpret_cast<int4*>(output_data + element_idx) = *reinterpret_cast<int4*>(shm_buffer_OutputT_ptr + idx);
        }
        __syncthreads();

    }
}

int main(){
    constexpr size_t token_num = 8 * 1024;
    constexpr size_t dim = 8192;
    constexpr int numel = token_num * dim;
    std::vector<float> input_float(numel);
    #pragma omp parallel for
    for(int i = 0; i < numel; i++){
        input_float[i] = float(rand()) / RAND_MAX;
    }
    void* d_input_float;
    void* d_output_half;
    void* d_signal_place;
    cudaMalloc(&d_input_float, numel * sizeof(float));
    cudaMalloc(&d_output_half, numel * sizeof(__half));

    cudaMemcpy(d_input_float, input_float.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
    constexpr int block_size = 1024;
    constexpr int VecSize = 128 / 8 / sizeof(float);

    cudaMalloc(&d_signal_place, numel / VecSize / WARPSIZE * sizeof(int32_t));
    cudaMemset(d_signal_place, 0, numel / VecSize / WARPSIZE * sizeof(int32_t));
    // int32_t grid_size = (numel + block_size * VecSize - 1) / (block_size * VecSize);


    // call v1 
    std::cout << "test v1 " << std::endl;
    const int32_t grid_size = 264;
    inplace_cast_kernel<float, __half, VecSize, block_size>
        <<<grid_size, block_size>>> (
            reinterpret_cast<float*>(d_input_float),
            reinterpret_cast<__half*>(d_output_half),
            reinterpret_cast<int32_t*>(d_signal_place),
            numel,
            (numel + VecSize - 1) / VecSize * VecSize
        );


    auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    } else {
        std::cout << "Success!" << std::endl;
    }
    std::vector<__half> output_half(numel);
    cudaMemcpy(output_half.data(), d_output_half, numel * sizeof(__half), cudaMemcpyDeviceToHost);
    for(int i=0;i<numel;++i){
        float f_output = __half2float(output_half[i]);
        if(fabs(f_output - input_float[i])>0.001) {
            std::cout<<"get error in idx: "<<i<<std::endl;
        }
    }

    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);

    
    constexpr int32_t shm_size_byte = 64 * 1024 ; // use 128k shm
    const int32_t grid_size_for_v2 = device_prop.multiProcessorCount ;
    constexpr int32_t block_element_num = shm_size_byte / sizeof(__half);
    std::cout<<"block_element_num: "<< block_element_num<<std::endl;
    error = cudaDeviceSynchronize();
    cudaMemset(d_signal_place, 0, numel / VecSize / WARPSIZE * sizeof(int32_t));
    cudaMemcpy(d_input_float, input_float.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
    error = cudaDeviceSynchronize();
    cudaFuncSetAttribute(inplace_cast_kernel_v2<float, __half, VecSize, block_size, block_element_num>, 
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size_byte); // 96KB in bytes

    std::cout<<"test v2 kernel"<<std::endl;
    inplace_cast_kernel_v2<float, __half, VecSize, block_size, block_element_num>
        <<<grid_size_for_v2, block_size, shm_size_byte, 0>>> (
            reinterpret_cast<float*>(d_input_float),
            reinterpret_cast<__half*>(d_input_float),
            reinterpret_cast<int32_t*>(d_signal_place),
            numel,
            (numel + VecSize - 1) / VecSize * VecSize
        );
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    } else {
        std::cout << "Success!" << std::endl;
    }
    cudaMemcpy(output_half.data(), d_input_float, numel * sizeof(__half), cudaMemcpyDeviceToHost);
    int error_count = 0;
    for(int i=0;i<numel;++i){
        float f_output = __half2float(output_half[i]);
        
        if(fabs(f_output - input_float[i])>0.001) {
            ++error_count;
            printf("### get error at idx:%d, %f - %f \n", i, f_output, input_float[i]);
        }
        if(error_count > 100) {
            break;
        }
    }

    outplace_cast_kernel<float, __half, VecSize>
        <<<528, 1024>>>(
            reinterpret_cast<float*>(d_input_float),
            reinterpret_cast<__half*>(d_output_half),
            numel);
    return 0;
}

