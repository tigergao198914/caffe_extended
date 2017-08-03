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
    /*
    Blob<Dtype> * getFeatureCol(){ return &feature_cols_; };
    Blob<Dtype> * getWeight(){
      return  this->blobs_[0].get();
    }
    */
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
    int sample_col_num_;
    int input_sample_total_size_;
    int output_sample_total_size_;
    int sample_num_;

    vector<int> input_shape_;
    vector<int> kernel_shape_;
    vector<int> stride_;
    vector<int> pad_;
    vector<int> dilation_;
    vector<int> output_shape_;
    Blob<Dtype> data_cols_;


    vector<int> weight_shape_;
    vector<int> kernel_stride_;
    vector<int> kernel_pad_;
    vector<int> weight_paded_shape_;
    vector<int> weight_output_shape_;
    Blob<Dtype> weight_cols_;

    shared_ptr<Blob<int> > data2col_map_;
    shared_ptr<Blob<int> > col2data_map_;
    shared_ptr<Blob<int> > weight2col_map_;
    shared_ptr<Blob<int> > col2weight_map_;

    void increase(vector<int>& index, vector<int>& boundary);
    int get_offset( vector<int>& col_index, vector<int>& kernel_index, 
                    vector<int>& data_size, vector<int>& pad, vector<int>&  stride );

    shared_ptr<Blob<int> > get_data2col_map_index(
            vector<int>& data_size, 
            vector<int>& kernel_size,
            vector<int>& stride,
            vector<int>& pad);

    shared_ptr<Blob<int> > get_col2data_map_index(vector<int>& data_size, 
                                                    vector<int>& kernel_size,
                                                    vector<int>& stride,
                                                    vector<int>& pad,
                                                    shared_ptr<Blob<int> >& data2col_map );
    void gen_weight_col(bool bGPU)
    {
        const Dtype *data;
        Dtype *data_col;
        const int *data2col_map;
        int data_col_len = weight_cols_.count();
        if( bGPU )
        {
            data = this->blobs_[0]->gpu_data();
            data_col = weight_cols_.mutable_gpu_data();
            data2col_map = weight2col_map_->gpu_data();
            data2col_gpu_v2(  data,  data_col, data2col_map, data_col_len );
        }
        else
        {
            data = this->blobs_[0]->cpu_data();
            data_col = weight_cols_.mutable_cpu_data();
            data2col_map = weight2col_map_->cpu_data();
            data2col_cpu_v2(  data,  data_col, data2col_map, data_col_len );
        }
    }

    void gen_data_col(bool bGPU, const Dtype *data, Dtype *data_col)
    {
        const int *data2col_map;
        int data_col_len = data_cols_.count();
        if( bGPU )
        {
            data2col_map = data2col_map_->gpu_data();
            data2col_gpu_v2(  data,  data_col, data2col_map, data_col_len );
        }
        else
        {
            data2col_map = data2col_map_->cpu_data();
            data2col_cpu_v2(  data,  data_col, data2col_map, data_col_len );
        }
    }

    void gen_weight_diff(bool bGPU)
    {
        int data_ele_len = 1;
        int data_len = this->blobs_[0]->count();
        for( int i=0; i<num_spatial_axes_; i++ )
        {
            data_ele_len *= ((kernel_shape_[i]-1)/kernel_stride_[i] + 1);
        }
        if(bGPU)
        {
            Dtype *weight_diff = this->blobs_[0]->mutable_gpu_diff();
            const Dtype *weight_col_diff = weight_cols_.gpu_diff();
            const int *col2data_map = col2weight_map_->gpu_data();
            col2data_gpu_v2( weight_diff, weight_col_diff, col2data_map, data_len, data_ele_len );
        }
        else
        {
            Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff();
            const Dtype *weight_col_diff = weight_cols_.cpu_diff();
            const int *col2data_map = col2weight_map_->cpu_data();
            col2data_cpu_v2( weight_diff, weight_col_diff, col2data_map, data_len, data_ele_len );
        }
    }

    void gen_data_diff(bool bGPU,  Dtype *data, const Dtype *data_col)
    {
        int data_ele_len = 1;
        int data_len = input_sample_total_size_;
        for( int i=0; i<num_spatial_axes_; i++ )
        {
            data_ele_len *= ((kernel_shape_[i]-1)/stride_[i] + 1);
        }
        if( bGPU )
        {
            const int *col2data_map = col2data_map_->gpu_data();
            col2data_gpu_v2(  data,  data_col, col2data_map, data_len, data_ele_len);
        }
        else
        {
            const int *col2data_map = col2data_map_->cpu_data();
            col2data_cpu_v2(  data,  data_col, col2data_map, data_len,  data_ele_len);
        }
    }

    void forward_cpu_gemm(const Dtype* input,const Dtype* weights, Dtype* output)
    {
        Dtype* data_col_data = data_cols_.mutable_cpu_data();
        gen_data_col( false, input, data_col_data);
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, feature_num_,
            sample_col_num_, kernel_total_size_,
            (Dtype)1., weights , data_col_data ,
            (Dtype)0., output );
    }
        
    void forward_gpu_gemm( const Dtype* input, const Dtype* weights, Dtype* output) 
    {
        
        Dtype* data_col_data = data_cols_.mutable_gpu_data();
        gen_data_col( true, input, data_col_data);
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, feature_num_,
            sample_col_num_, kernel_total_size_,
            (Dtype)1., weights , data_col_data ,
            (Dtype)0., output );
    }


    void weight_cpu_gemm(const Dtype* input, Dtype* weights, const Dtype* output)
    {
        Dtype* data_col_data = data_cols_.mutable_cpu_data();
        gen_data_col( false, input, data_col_data);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, kernel_total_size_,
            feature_num_, sample_col_num_, 
            (Dtype)1., data_col_data, output, 
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

    void weight_gpu_gemm( const Dtype* input, Dtype* weights, const Dtype* output)
    {
        Dtype* data_col_data = data_cols_.mutable_gpu_data();
        gen_data_col( true, input, data_col_data);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, kernel_total_size_,
            feature_num_, sample_col_num_, 
            (Dtype)1., data_col_data, output, 
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
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, kernel_total_size_,
            sample_col_num_, feature_num_,
            (Dtype)1., weights, output ,
            (Dtype)0., col_buff);
        gen_data_diff( false, input,  col_buff);
    }

    void backward_gpu_gemm(const Dtype* output,const Dtype* weights,Dtype* input) 
    {
        Dtype* col_buff = data_cols_.mutable_gpu_data();
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, kernel_total_size_,
            sample_col_num_, feature_num_,
            (Dtype)1., weights, output ,
            (Dtype)0., col_buff);
        gen_data_diff( true, input,  col_buff);
    }

#if 0
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

    


 
#endif
};
}
#endif