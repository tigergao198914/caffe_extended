#include <vector>

#include "caffe/layers/extended_convolution_layer.hpp"

namespace caffe {
template <typename Dtype>
void ExtendedConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    gen_kernel_gpu();
    std::cout<<"feature_cols:"<<std::endl;
    feature_cols_.display();
    const Dtype* weight = feature_cols_.gpu_data();
    Dtype *bottom_data = bottom[0]->mutable_gpu_data();
    Dtype *top_data = top[0]->mutable_gpu_data();
    for ( int n = 0; n < sample_num_; ++n ){
        this->forward_gpu_gemm( bottom_data + n * bottom_sample_dim_, weight,
            top_data + n * top_sample_dim_);
    }
}

template <typename Dtype>
void ExtendedConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    gen_kernel_gpu();
    const Dtype* weight = feature_cols_.gpu_data();
    Dtype* weight_diff = feature_cols_.mutable_gpu_diff();
    
    for (int i = 0; i < top.size(); ++i) {
        Dtype* top_diff = top[i]->mutable_gpu_diff();
        Dtype* bottom_data = bottom[i]->mutable_gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    
        if (this->param_propagate_down_[0] || propagate_down[i]) {
            for (int n = 0; n < sample_num_; ++n) {
                // gradient w.r.t. weight. Note that we will accumulate diffs.
                if (this->param_propagate_down_[0]) {
                this->weight_gpu_gemm(bottom_data + n * bottom_sample_dim_,
                    weight_diff, top_diff + n * top_sample_dim_);
                }
                // gradient w.r.t. bottom data, if necessary.
                if (false) {//propagate_down[i]
                    this->backward_gpu_gemm(top_diff + n * top_sample_dim_, weight,
                        bottom_diff + n * bottom_sample_dim_);
                }
            }
        }
    }

    compute_diff_from_kernel_gpu();
}

INSTANTIATE_LAYER_GPU_FUNCS(ExtendedConvolutionLayer);

}