#include <vector>

#include "caffe/layers/extended_convolution_layer.hpp"

namespace caffe {
template <typename Dtype>
void ExtendedConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //bottom[0]->display();
    gen_weight_col(true);
    const Dtype* weight = weight_cols_.gpu_data();
    const Dtype *bottom_data = bottom[0]->gpu_data();
    Dtype *top_data = top[0]->mutable_gpu_data();
    for ( int n = 0; n < sample_num_; ++n ){
        this->forward_gpu_gemm( bottom_data + n * input_sample_total_size_, weight,
            top_data + n * output_sample_total_size_);
    }
    //top[0]->display();
}

template <typename Dtype>
void ExtendedConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    //top[0]->display(true);
    gen_weight_col(true);
    const Dtype* weight = weight_cols_.gpu_data();
    Dtype* weight_diff = weight_cols_.mutable_gpu_diff();
    caffe_gpu_set( weight_cols_.count(), static_cast<Dtype>(0), weight_diff );   
    for (int i = 0; i < top.size(); ++i) {
        const Dtype* top_diff = top[i]->mutable_gpu_diff();
        const Dtype* bottom_data = bottom[i]->mutable_gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    
        if (this->param_propagate_down_[0] || propagate_down[i]) {
            for (int n = 0; n < sample_num_; ++n) {
                // gradient w.r.t. weight. Note that we will accumulate diffs.
                if (this->param_propagate_down_[0]) {
                this->weight_gpu_gemm(bottom_data + n * input_sample_total_size_,
                    weight_diff, top_diff + n * output_sample_total_size_);
                }
                // gradient w.r.t. bottom data, if necessary.
                if (propagate_down[i]) {
                    this->backward_gpu_gemm(top_diff + n * output_sample_total_size_, weight,
                        bottom_diff + n * input_sample_total_size_);
                }
            }
        }
    }

    gen_weight_diff(true);
}

INSTANTIATE_LAYER_GPU_FUNCS(ExtendedConvolutionLayer);

}