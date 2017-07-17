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
        Dtype max = -0x7fffffff;
        int max_offset  = 0;
        for( int j=0; j<kernel_size; j++ )
        {
            int paded_offset = start_offset + kernel_offset[j];
            int offset = index_map[paded_offset];
            if( offset!=-1)
            {
                max_offset = (max>bottom_data[offset])?max_offset:offset;
                max = (max>bottom_data[offset])?max:bottom_data[offset];               
            }
        }
        max_idx_data[index] = max_offset;
        top_data[index] = max;
        //printf("index:%d, top_data[index]:%f\n", index, top_data[index]);
  }
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads,  Dtype*  bottom_data, 
    const int* offset_data, const Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i = offset_data[index];
    #if 1
    atomicAdd( &bottom_data[i], top_data[index]);
    //printf("index:%d, bottom_data[%d]:%lf\n", index,i, bottom_data[i]);
    #else

    #endif
  }
}

template <typename Dtype>
void ExtendedPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      #if 0
      std::cout << "bottom gpu:"<<std::endl;
      bottom[0]->display();
      #endif
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
      #if 0
      //std::cout << "bottom gpu:"<<std::endl;
      //bottom[0]->display();
      std::cout << "top gpu:"<<std::endl;
      top[0]->display();
      #endif
}

template <typename Dtype>
void ExtendedPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom[0]->mutable_gpu_diff());
      MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(ouput_ele_size_), CAFFE_CUDA_NUM_THREADS>>>(
          ouput_ele_size_,
          bottom[0]->mutable_gpu_diff(),
          this->max_idx_.gpu_data(),
          top[0]->gpu_diff()
      );
      #if 0
      std::cout<<"top gpu diff:"<<std::endl;
      top[0]->display(true);
      std::cout<<"max id offset:"<<std::endl;
      max_idx_.display();
      std::cout<<"bottom gpu diff:"<<std::endl;
      bottom[0]->display(true);
      #endif
}

INSTANTIATE_LAYER_GPU_FUNCS(ExtendedPoolingLayer);

}