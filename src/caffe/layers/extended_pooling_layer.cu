#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/extended_pool_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(
    const int nthreads,
    const int kernel_size, 
    const Dtype*  bottom_data,
    const int* input_offset,
    const int* kernel_offset,
    const int* index_map, 
    int* max_idx_data,
    Dtype*  top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int start_offset = input_offset[index];
        Dtype max = -(~0X0);
        int max_offset  = 0;
        for( int j=0; j<kernel_size; j++ )
        {
            int paded_offset = start_offset + kernel_offset[j];
            int offset = index_map[paded_offset];
            if( offset!=-1)
            {
                max = (max>bottom_data[offset])?max:bottom_data[offset];
                max_offset = (max>bottom_data[offset])?max_offset:offset;
            }
        }
        top_data[index] = max;
        max_idx_data[index] = max_offset;
  }
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads,  Dtype*  bottom_data, 
    const int* offset_data, const Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i = offset_data[index];
    bottom_data[i] = top_data[index];
  }
}

template <typename Dtype>
void ExtendedPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(ouput_ele_size_), CAFFE_CUDA_NUM_THREADS>>>(
          ouput_ele_size_,
          kernel_ele_size_,
          bottom[0]->gpu_data(),
          input_offset_.gpu_data(),
          kernel_offset_.gpu_data(),
          paded_index_map_.gpu_data(),
          max_idx_.mutable_gpu_data(),
          top[0]->mutable_gpu_data()
      );
}

template <typename Dtype>
void ExtendedPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(ouput_ele_size_), CAFFE_CUDA_NUM_THREADS>>>(
          ouput_ele_size_,
          bottom[0]->mutable_gpu_diff(),
          this->max_idx_.gpu_data(),
          top[0]->gpu_diff()
      );
}

INSTANTIATE_LAYER_GPU_FUNCS(ExtendedPoolingLayer);

}