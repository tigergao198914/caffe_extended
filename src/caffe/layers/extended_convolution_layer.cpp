#include <vector>

#include "caffe/layers/extended_convolution_layer.hpp"

namespace caffe {
    template <typename Dtype>
    void ExtendedConvolutionLayer<Dtype>::compute_output_shape(){
        const int* kernel_shape_data = this->kernel_shape_.cpu_data();
        const int* stride_data = this->stride_.cpu_data();
        const int* pad_data = this->pad_.cpu_data();
        const int* dilation_data = this->dilation_.cpu_data();
        this->output_shape_.clear();

        for( int i=0; i< weight_dim_; i++ )
        {
            this->output_shape_.push_back( feature_map_shape_.cpu_data()[i] );
        }

        sample_col_num_ = 1;
        for( int i=0; i < this->num_spatial_axes_; ++i)
        {
            const int input_dim = this->bottom_sample_shape_.cpu_data()[i];
            const int kernel_extent = dilation_data[i] * (kernel_shape_data[i]-1) +1;
            const int output_dim =  (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
            this->output_shape_.push_back( output_dim );
            sample_col_num_ *= output_dim;
        }
    }

    template <typename Dtype>
    void ExtendedConvolutionLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
    {
        ExtendedConvolutionParameter conv_param = this->layer_param_.extended_convolution_param();
        const int first_spatial_axis =  1;
        const int num_axes = bottom[0]->num_axes();
        num_spatial_axes_ = num_axes - first_spatial_axis;

        vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_,1));

        kernel_shape_.Reshape(spatial_dim_blob_shape);
        int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
        kernel_dim_ = 1;
        const int num_kernel_dims = conv_param.kernel_size_size();
        for ( int i = 0; i < num_spatial_axes_; ++i )
        {
            kernel_shape_data[i] = 
                conv_param.kernel_size((num_kernel_dims==1)?0:i);
            kernel_dim_ *= kernel_shape_data[i];
        }

        stride_.Reshape(spatial_dim_blob_shape);
        const int num_stride_dims = conv_param.stride_size();
        int* stride_data = stride_.mutable_cpu_data();
        const int kDefaultStride = 1;
        for ( int i=0; i< num_spatial_axes_; i++ )
        {
            stride_data[i] = (num_stride_dims == 0)? kDefaultStride
                :conv_param.stride((num_stride_dims == 1) ? 0 : i);
        }

        pad_.Reshape(spatial_dim_blob_shape);
        int* pad_data = pad_.mutable_cpu_data();
        const int num_pad_dims = conv_param.pad_size();
        const int kDefaultPad = 0;
        for( int i=0; i < num_spatial_axes_; i++ ){
                  pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i); 
        }

        dilation_.Reshape(spatial_dim_blob_shape);
        int* dilation_data = dilation_.mutable_cpu_data();
        const int num_dilation_dims = conv_param.dilation_size();
        const int kDefaultDilation = 1;
        for ( int i = 0; i < num_spatial_axes_; i++ ){
            dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                conv_param.dilation((num_dilation_dims==1)? 0 : i);
        }

        weight_dim_ = conv_param.weight_dim_size();
        vector<int> weight_blob_shape(1, weight_dim_);

        feature_stride_.Reshape(spatial_dim_blob_shape);
        const int num_feature_stride_dims = conv_param.feature_stride_size();
        int* feature_stride_data = feature_stride_.mutable_cpu_data();
        for ( int i=0; i< weight_dim_; i++ )
        {
            feature_stride_data[i] = (num_feature_stride_dims == 0)? kDefaultStride
                :conv_param.feature_stride((num_feature_stride_dims == 1) ? 0 : i);
        }
        for( int i=weight_dim_; i<num_spatial_axes_; i++ )
        {
            feature_stride_data[i] = kDefaultStride;
        }

        feature_pad_.Reshape(spatial_dim_blob_shape);
        int* feature_pad_data = feature_pad_.mutable_cpu_data();
        const int num_feature_pad_dims = conv_param.feature_pad_size();
        for( int i=0; i < weight_dim_; i++ )
        {
            feature_pad_data[i] = (num_feature_pad_dims == 0) ? kDefaultPad :
                conv_param.feature_pad((num_feature_pad_dims == 1) ? 0 : i); 
        }
        for( int i=weight_dim_; i<num_spatial_axes_; i++ )
        {
            feature_pad_data[i] = kDefaultPad;
        }

        feature_dilation_.Reshape(spatial_dim_blob_shape);
        int* feature_dilation_data = feature_dilation_.mutable_cpu_data();
        const int num_feature_dilation_dims = conv_param.feature_dilation_size();
        for( int i=0; i < weight_dim_; i++ )
        {
            feature_dilation_data[i] = (num_feature_dilation_dims == 0) ? kDefaultDilation :
                conv_param.feature_dilation((num_feature_dilation_dims == 1) ? 0 : i); 
        }
        for( int i=weight_dim_; i<num_spatial_axes_; i++ )
        {
            feature_dilation_data[i] = kDefaultDilation;
        }

        weight_shape_.Reshape(spatial_dim_blob_shape);
        int* weight_shape_data = weight_shape_.mutable_cpu_data();
        for ( int i = 0; i < weight_dim_; ++i )
        {
            weight_shape_data[i] = conv_param.weight_dim(i);
            weight_shape_vector_.push_back(weight_shape_data[i]);
        }
        for ( int i=weight_dim_; i<num_spatial_axes_; i++ )
        {
            weight_shape_data[i] = kernel_shape_data[i];
            weight_shape_vector_.push_back(weight_shape_data[i]);
        }

        feature_num_ = 1;
        feature_map_shape_.Reshape(weight_blob_shape);
        int* feature_map_shape_data = feature_map_shape_.mutable_cpu_data();
        for ( int i = 0; i < weight_dim_; ++i )
        {
            //compute feature map size
            feature_map_shape_data[i] =  
                (weight_shape_data[i]+2*feature_pad_data[i]-kernel_shape_data[i])/feature_stride_data[i]+1;
            feature_num_ *= feature_map_shape_data[i];
        }
 
        //feature_cols_shape_.push_back(kernel_dim_);
        vector<int> feature_weight_blob_shape(1, num_spatial_axes_+1);
        feature_cols_shape_.Reshape(feature_weight_blob_shape);
        int *feature_cols_shape_data = feature_cols_shape_.mutable_cpu_data();
        feature_cols_shape_data[0] = kernel_dim_;
        feature_cols_shape_vector_.push_back(kernel_dim_);
        for( int i = 0; i< weight_dim_; ++i )
        {
            feature_cols_shape_data[i+1] = feature_map_shape_data[i];
            feature_cols_shape_vector_.push_back( feature_map_shape_data[i] );
        }
        for( int i = weight_dim_; i<num_spatial_axes_; i++ )
        {
            feature_cols_shape_data[i+1] = 1;
            feature_cols_shape_vector_.push_back( 1 );
        }
        feature_cols_.Reshape(feature_cols_shape_vector_);


        this->blobs_.resize(1);
        //initalize and fill the weight
        this->blobs_[0].reset( new Blob<Dtype>(weight_shape_vector_) );
        shared_ptr<Filler<Dtype> > weight_filter( GetFiller<Dtype>(
            this->layer_param_.extended_convolution_param().weight_filler()) );
        weight_filter->Fill( this->blobs_[0].get() );
        this->param_propagate_down_.resize(this->blobs_.size(), true);
    }

    template <typename Dtype>
    void ExtendedConvolutionLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
    {
        sample_num_ =  bottom[0]->count(0,1);
        //bottom_sample_shape_.resize();
        
        vector<int> bottom_sample_shape(1, bottom[0]->num_axes()-1);
        bottom_sample_shape_.Reshape( bottom_sample_shape );
        int* bottom_sample_shape_data = bottom_sample_shape_.mutable_cpu_data();
        for( int i = 1; i< bottom[0]->num_axes(); i++)
        {
            bottom_sample_shape_data[i-1] = bottom[0]->shape(i);
        }

        compute_output_shape();
        vector<int> top_shape( bottom[0]->shape().begin(), bottom[0]->shape().begin()+1 );
        for( int i = 0; i < this->output_shape_.size(); ++i )
        {
            top_shape.push_back(this->output_shape_[i]);
        }
        for( int top_id=0; top_id < top.size(); ++top_id )
        {
            top[top_id]->Reshape(top_shape);
        }

        bottom_sample_dim_ = bottom[0]->count(1);
        top_sample_dim_ = top[0]->count(1);

        input_col_offset_ = kernel_dim_ * feature_num_;
 
        int data_col_dim = bottom[0]->num_axes();
        vector<int> spatial_dim_blob_shape(1, data_col_dim);
        data_col_buffer_shape_.Reshape(spatial_dim_blob_shape);

        data_col_buffer_shape_vector_.clear();
        int *data_col_buffer_shape_data_ = data_col_buffer_shape_.mutable_cpu_data();
        data_col_buffer_shape_data_[0] = kernel_dim_;
        data_col_buffer_shape_vector_.push_back(kernel_dim_);
        for( int i = 0; i<data_col_dim-1; ++i )
        {
            data_col_buffer_shape_data_[i+1] = output_shape_[weight_dim_+i];
            data_col_buffer_shape_vector_.push_back(data_col_buffer_shape_data_[i+1]);
        }
        data_cols_.Reshape(data_col_buffer_shape_vector_);
    }

    template <typename Dtype>
    void ExtendedConvolutionLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
    {
        gen_kernel_cpu();
        const Dtype* weight = feature_cols_.cpu_data();
        const Dtype *bottom_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        for ( int n = 0; n < sample_num_; ++n ){
            this->forward_cpu_gemm( bottom_data + n * bottom_sample_dim_, weight,
                top_data + n * top_sample_dim_);
        }
    }

    template <typename Dtype>
    void ExtendedConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        gen_kernel_cpu();
        const Dtype* weight = feature_cols_.cpu_data();
        Dtype* weight_diff = feature_cols_.mutable_cpu_diff();
        
        for (int i = 0; i < top.size(); ++i) {
            const Dtype* top_diff = top[i]->cpu_diff();
            const Dtype* bottom_data = bottom[i]->cpu_data();
            Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
        
            if (this->param_propagate_down_[0] || propagate_down[i]) {
                for (int n = 0; n < sample_num_; ++n) {
                    // gradient w.r.t. weight. Note that we will accumulate diffs.
                    if (this->param_propagate_down_[0]) {
                    this->weight_cpu_gemm(bottom_data + n * bottom_sample_dim_,
                        weight_diff, top_diff + n * top_sample_dim_);
                    }
                    // gradient w.r.t. bottom data, if necessary.
                    if (false) {//propagate_down[i]
                        this->backward_cpu_gemm(top_diff + n * top_sample_dim_, weight,
                            bottom_diff + n * bottom_sample_dim_);
                    }
                }
            }
        }

        compute_diff_from_kernel_cpu();
    }

    #ifdef CPU_ONLY
    STUB_GPU(ExtendedConvolutionLayer);
    #endif
    REGISTER_LAYER_CLASS(ExtendedConvolution);
    INSTANTIATE_CLASS(ExtendedConvolutionLayer);
}