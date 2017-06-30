#ifndef CAFFE_EXTENDED_CONV_LAYER_HPP_
#define CAFFE_EXTENDED_CONV_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class ExtendedConvolutionLayer: public Layer<Dtype> {
public:
    explicit ExtendedConvolutionLayer(const LayerParameter& param)
        : Layer<Dtype>(param){}
    
    virtual inline const char* type() const {return "ExtenedConvolution";}
    Blob<Dtype> * getFeatureCol(){ return &feature_cols_; };
    Blob<Dtype> * getWeight(){
      return  this->blobs_[0].get();
    }
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual inline bool reverse_dimensions() { return false; }
    virtual void compute_output_shape();
    virtual void LayerSetUp( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    std::vector<int> output_shape_;
    int num_spatial_axes_;

    int weight_dim_;
    int feature_num_;
    int kernel_dim_;
    Blob<Dtype> feature_cols_; //get weight from param
    Blob<int> feature_cols_shape_;
    std::vector<int> feature_cols_shape_vector_;
    Blob<int> feature_map_shape_;
    Blob<int> feature_pad_;
    Blob<int> feature_stride_;
    Blob<int> feature_dilation_;
    Blob<int> weight_shape_;
    std::vector<int> weight_shape_vector_;

    Blob<int> kernel_shape_;
    Blob<int> stride_;
    Blob<int> pad_;
    Blob<int> dilation_;
    Blob<int> bottom_sample_shape_;

    int sample_num_;
    int bottom_sample_dim_;
    int top_sample_dim_;
    int input_col_offset_;
    int sample_col_num_;
    Blob<int> data_col_buffer_shape_;
    std::vector<int> data_col_buffer_shape_vector_;
    Blob<Dtype> data_cols_; 

    /*generate kernel map from learnable param*/
    void gen_kernel_cpu()
    {
       Dtype *param = this->blobs_[0]->mutable_cpu_data();
       data2col_cpu(param, num_spatial_axes_, 
          weight_shape_.cpu_data(),  feature_cols_shape_.cpu_data(),
          kernel_shape_.cpu_data(), feature_pad_.cpu_data(), feature_stride_.cpu_data(),
          feature_cols_.mutable_cpu_data());
    }

    void gen_data_col_cpu(const Dtype* input, Dtype* output)
    {
        data2col_cpu(input, num_spatial_axes_,
            bottom_sample_shape_.cpu_data(), data_col_buffer_shape_.cpu_data(),
            kernel_shape_.cpu_data(), pad_.cpu_data(), stride_.cpu_data(),
            output);
    }

    void compute_diff_from_kernel_cpu()
    {
        Dtype *param = this->blobs_[0]->mutable_cpu_diff();
        col2data_cpu(param, num_spatial_axes_,
          weight_shape_.cpu_data(), feature_cols_shape_.cpu_data(),
          kernel_shape_.cpu_data(), feature_pad_.cpu_data(), feature_stride_.cpu_data(),
          feature_cols_.cpu_diff());
    }

    void compute_diff_from_col_data_cpu(Dtype* col_buff, Dtype *data)
    {
        col2data_cpu(data, num_spatial_axes_, 
            bottom_sample_shape_.cpu_data(), data_col_buffer_shape_.cpu_data(),
            kernel_shape_.cpu_data(), pad_.cpu_data(), stride_.cpu_data(),
            col_buff);
    }

    void forward_cpu_gemm(const Dtype* input,const Dtype* weights, Dtype* output)
    {
        Dtype* col_buff = data_cols_.mutable_cpu_data();
        gen_data_col_cpu(input, col_buff);
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, feature_num_,
            sample_col_num_, kernel_dim_,
            (Dtype)1., weights , col_buff ,
            (Dtype)0., output );
    }

    void weight_cpu_gemm(const Dtype* input, Dtype* weights, Dtype* output)
    {
        Dtype* col_buff = data_cols_.mutable_cpu_data();
        gen_data_col_cpu( input, col_buff);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, kernel_dim_,
            feature_num_, sample_col_num_, 
            (Dtype)1., col_buff, output, 
            (Dtype)1., weights);
    }

    void backward_cpu_gemm(const Dtype* output,const Dtype* weights,Dtype* input) 
    {
        Dtype* col_buff = data_cols_.mutable_cpu_data();
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, kernel_dim_,
            sample_col_num_, feature_num_,
            (Dtype)1., weights, output ,
            (Dtype)0., col_buff);
        compute_diff_from_col_data_cpu(col_buff, input);
    }
};
}
#endif