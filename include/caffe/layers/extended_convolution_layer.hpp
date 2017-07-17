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
    
    virtual inline const char* type() const {return "ExtendedConvolution";}
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
    //virtual void compute_output_shape();
    virtual void LayerSetUp( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
#if 0
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
#endif

    int num_spatial_axes_;
    int feature_num_;
    int kernel_total_size_;
    int input_sample_total_size_;
    int output_sample_total_size_;

    vector<int> input_shape_;
    vector<int> kernel_shape_;
    vector<int> stride_;
    vector<int> pad_;
    vector<int> dilation_;
    vector<int> input_paded_shape_; 
    vector<int> output_shape_;
    Blob<int> input_offset_;    //map output to start index of input area
    Blob<int> paded_index_map_; //map paded input index to input index
    Blob<int> kernel_offset_;   //how much index of input moves when the kernel index increase
    Blob<Dtype> data_cols_;
    Blob<Dtype> img2col_map_;


    vector<int> weight_shape_;
    vector<int> kernel_stride_;
    vector<int> kernel_pad_;
    vector<int> weight_paded_shape_;
    vector<int> weight_output_shape_;
 
    Blob<int> weight_input_offset_;
    Blob<int> paded_weight_index_map_;
    Blob<int> weight_offset_;
    Blob<Dtype> weight_cols_;
    Blob<Dtype> weight2col_map_;

    
    /*generate kernel map from learnable param*/
    void gen_kernel_cpu()
    {
       Dtype *weight_data = this->blobs_[0]->mutable_cpu_data();
       data2col_cpu(weight_data, num_spatial_axes_, 
          weight_shape_.cpu_data(),  feature_cols_shape_.cpu_data(),
          kernel_shape_.cpu_data(), feature_pad_.cpu_data(), feature_stride_.cpu_data(),
          feature_cols_.mutable_cpu_data());
       caffe_set( feature_cols_.count(),Dtype(0),feature_cols_.mutable_cpu_diff() );
    }

    void gen_kernel_gpu()
    {
       std::cout<< "weight:"<<std::endl;
       this->blobs_[0]->display();
       Dtype *weight_data = this->blobs_[0]->mutable_gpu_data();
       data2col_gpu(weight_data, num_spatial_axes_, feature_cols_.count(1),
          weight_shape_.gpu_data(),  feature_cols_shape_.gpu_data(),
          kernel_shape_.gpu_data(), feature_pad_.gpu_data(), feature_stride_.gpu_data(),
          feature_cols_.mutable_gpu_data());
       caffe_gpu_set( feature_cols_.count(),Dtype(0),feature_cols_.mutable_gpu_diff() );
    }

    void gen_data_col_cpu(const Dtype* input, Dtype* output)
    {
        data2col_cpu(input, num_spatial_axes_,
            bottom_sample_shape_.cpu_data(), data_col_buffer_shape_.cpu_data(),
            kernel_shape_.cpu_data(), pad_.cpu_data(), stride_.cpu_data(),
            output);
    }

    void gen_data_col_gpu(Dtype* input, Dtype* output)
    {
        data2col_gpu(input, num_spatial_axes_, data_cols_.count(1),
            bottom_sample_shape_.gpu_data(), data_col_buffer_shape_.gpu_data(),
            kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
            output);
    }

    void compute_diff_from_kernel_cpu()
    {
        Dtype *param = this->blobs_[0]->mutable_cpu_diff();
        col2data_cpu(param, num_spatial_axes_,
          weight_shape_.cpu_data(), feature_cols_shape_.cpu_data(),
          kernel_shape_.cpu_data(), feature_pad_.cpu_data(), feature_stride_.cpu_data(),
          feature_cols_.mutable_cpu_diff());
    }

    void compute_diff_from_kernel_gpu()
    {
        Dtype *weight_data = this->blobs_[0]->mutable_gpu_diff();
        col2data_gpu(weight_data, num_spatial_axes_, feature_cols_.count(1),
          weight_shape_.gpu_data(), feature_cols_shape_.gpu_data(),
          kernel_shape_.gpu_data(), feature_pad_.gpu_data(), feature_stride_.gpu_data(),
          feature_cols_.mutable_gpu_diff());
    }

    void compute_diff_from_col_data_cpu(Dtype* col_buff, Dtype *data)
    {
        col2data_cpu(data, num_spatial_axes_, 
            bottom_sample_shape_.cpu_data(), data_col_buffer_shape_.cpu_data(),
            kernel_shape_.cpu_data(), pad_.cpu_data(), stride_.cpu_data(),
            col_buff);
    }

    void compute_diff_from_col_data_gpu(Dtype* col_buff, Dtype *data)
    {
        col2data_gpu(data, num_spatial_axes_, data_cols_.count(1),
            bottom_sample_shape_.gpu_data(), data_col_buffer_shape_.gpu_data(),
            kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
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
        
    void forward_gpu_gemm( Dtype* input, const Dtype* weights, Dtype* output) {
        
        Dtype* col_buff = data_cols_.mutable_gpu_data();
        gen_data_col_gpu(input, col_buff);
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, feature_num_,
            sample_col_num_, kernel_dim_,
            (Dtype)1., weights , col_buff ,
            (Dtype)0., output );
    }

    void weight_cpu_gemm(const Dtype* input, Dtype* weights, const Dtype* output)
    {
        Dtype* col_buff = data_cols_.mutable_cpu_data();
        gen_data_col_cpu( input, col_buff);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, kernel_dim_,
            feature_num_, sample_col_num_, 
            (Dtype)1., col_buff, output, 
            (Dtype)1., weights);
#if 0
       std::cout<<std::endl<<"top diff3:"<<std::endl;
        for( int i=0; i<sample_col_num_*feature_num_ ; i++)
        {
            if(i%feature_num_==0)
            {
                std::cout<< std::endl;
            }
            std::cout<<output[i]<<"  ";
        }

        std::cout<<"col_buff:"<<std::endl;
        for( int i=0; i<data_cols_.count(); i++)
        {
            if( i%data_cols_.count(1) == 0 )
            {
                std::cout<< std::endl;
            }
           std::cout<< col_buff[i]<< "  ";
        }

        std::cout<<"weight diff:"<<std::endl;
        for( int i=0; i<feature_cols_.count(); i++)
        {
            if( i%16 == 0 )
            {
                std::cout<< std::endl;
            }
            std::cout<< weights[i] << "  ";
        }
#endif
    }

    void weight_gpu_gemm( Dtype* input, Dtype* weights, const Dtype* output)
    {
        Dtype* col_buff = data_cols_.mutable_gpu_data();
        gen_data_col_gpu( input, col_buff);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, kernel_dim_,
            feature_num_, sample_col_num_, 
            (Dtype)1., col_buff, output, 
            (Dtype)1., weights);
#if 0
       std::cout<<std::endl<<"top diff3:"<<std::endl;
        for( int i=0; i<sample_col_num_*feature_num_ ; i++)
        {
            if(i%feature_num_==0)
            {
                std::cout<< std::endl;
            }
            std::cout<<output[i]<<"  ";
        }

        std::cout<<"col_buff:"<<std::endl;
        for( int i=0; i<data_cols_.count(); i++)
        {
            if( i%data_cols_.count(1) == 0 )
            {
                std::cout<< std::endl;
            }
           std::cout<< col_buff[i]<< "  ";
        }

        std::cout<<"weight diff:"<<std::endl;
        for( int i=0; i<feature_cols_.count(); i++)
        {
            if( i%16 == 0 )
            {
                std::cout<< std::endl;
            }
            std::cout<< weights[i] << "  ";
        }
#endif
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

    void backward_gpu_gemm(const Dtype* output,const Dtype* weights,Dtype* input) 
    {
        Dtype* col_buff = data_cols_.mutable_gpu_data();
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, kernel_dim_,
            sample_col_num_, feature_num_,
            (Dtype)1., weights, output ,
            (Dtype)0., col_buff);
        compute_diff_from_col_data_gpu(col_buff, input);
    }

};
}
#endif