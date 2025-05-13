// [m,n] -> [m,1]

#define WARP_SIZE 32

template<typename T, typename U, typename Op, int VecSize, int BlockSize>
__global__  void block_reduce(
    const T* input,
    T* output,
    size_t numel,
    size_t m,
    size_t n
) {
    // <<<m, 1024>>>
    int64_t linear_idx = threadIdx.x + blockIdx.x * n;
    // TODO init for different Op
    U thread_reduce_result = 0;
    std::array<T, VecSize> input_vec;
    int32_t lane_id = threadIdx.x % 32;
    int32_t warp_id = threadIdx.x / 32;

    __shared__ U shm[BlockSize/WARP_SIZE];

    const int4* input_vec_ptr = reinterplate_cast<const int4*>(input_vec.data())
    for(int64_t idx=linear_idx * VecSize; idx<numel; idx+=blockDim.x * VecSize) {
        *input_vec_ptr = * (reinterplate_cast<const int4*>(input + idx));
        #pragma unroll
        for(int vec_i = 0; vec_i < VecSize; vec_i ++){
            Op(thread_reduce_result, input_vec[vec_i])
        }
    }

    // warp reduce
    #pragma unorll
    for(int i = WARP_SIZE / 2 ; i > 0; i/=2){
        Op(thread_reduce_result, __shuffl_xor(0xffffffff, thread_reduce_result, WARP_SIZE, i));
    }
    
    if(lane_id == 0){
        shm[warp_id] = thread_reduce_result;
    }
    __syncthread();
    if(warp_id == 0){
        U warp_reduce_result = 0;
        if(lane_id < BlockSize/WARP_SIZE){
            warp_reduce_result = shm[lane_id];
        }
        // warp reduce
        #pragma unorll
        for(int i = WARP_SIZE / 2 ; i > 0; i/=2){
            Op(warp_reduce_result, __shuffl_xor(0xffffffff, warp_reduce_result, WARP_SIZE, i));
        }
    }
    if(threadIdx.x == 0){
        output[blockIdx.x] = static_cast<T>(warp_reduce_result);
    }
}

int main(){

}