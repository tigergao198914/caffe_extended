#include <vector>

#include "caffe/layers/extended_convolution_layer.hpp"

namespace caffe {
#if 0
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
#endif

    template <typename Dtype>
    void ExtendedConvolutionLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
    {
        ExtendedConvolutionParameter conv_param = this->layer_param_.extended_convolution_param();
        const int num_axes = bottom[0]->num_axes();
        num_spatial_axes_  = num_axes - 1;
        const int kDefaultStride = 1;
        const int kDefaultPad = 0;
        const int kDefaultDilation = 1;
        kernel_total_size_ = 1;
        feature_num_ = 1;
        for( int i=0; i<num_spatial_axes_; i++ )
        {
            const int num_kernel_dims = conv_param.kernel_size_size();
            kernel_shape_.push_back( conv_param.kernel_size((num_kernel_dims==1)?0:i) );
            kernel_total_size_ *= kernel_shape_[i];

            const int num_stride_dims = conv_param.stride_size();
            stride_.push_back( (num_stride_dims == 0)? kDefaultStride:
                conv_param.stride((num_stride_dims == 1) ? 0 : i) );
            
            const int num_pad_dims = conv_param.pad_size();
            pad_.push_back( (num_pad_dims == 0) ? kDefaultPad :
                conv_param.pad((num_pad_dims == 1) ? 0 : i) ) ; 

            const int num_dilation_dims = conv_param.dilation_size();
            dilation_.push_back( (num_dilation_dims == 0) ? kDefaultDilation :
                conv_param.dilation((num_dilation_dims==1)? 0 : i) );

            const int num_weight_dims = conv_param.weight_dim_size();
            weight_shape_.push_back( (i>=num_weight_dims)? kernel_shape_[i]:
                conv_param.weight_dim(i));

            const int num_feature_stride_dims = conv_param.feature_stride_size();
            kernel_stride_.push_back( ( i>=num_feature_stride_dims )? kDefaultStride:
                conv_param.feature_stride(i) );

            const int num_kernel_pad_dims = conv_param.feature_pad_size();
            kernel_pad_.push_back( (i>=num_kernel_pad_dims) ? kDefaultPad :
                conv_param.feature_pad(i) ); 

            weight_paded_shape_.push_back( weight_shape_[i]+2*kernel_pad_[i] );
            weight_output_shape_.push_back( (weight_paded_shape_[i]-kernel_shape_[i])/kernel_stride_[i]+1 );
            feature_num_ *= weight_output_shape_[i];

        }

        vector<int> weight_cols_shahpe_;
        weight_cols_shahpe_.push_back(kernel_total_size_);
        weight_cols_shahpe_.push_back(feature_num_);
        weight_cols_.Reshape(weight_cols_shahpe_);

        this->blobs_.resize(1);
        //initalize and fill the weight
        this->blobs_[0].reset( new Blob<Dtype>(weight_shape_) );
        shared_ptr<Filler<Dtype> > weight_filter( GetFiller<Dtype>(
            this->layer_param_.extended_convolution_param().weight_filler()) );
        weight_filter->Fill( this->blobs_[0].get() );
        this->param_propagate_down_.resize(this->blobs_.size(), true);
        //std::cout<<"weight:"<<std::endl;
        //this->blobs_[0]->display();
        weight2col_map_ = get_data2col_map_index(weight_shape_,kernel_shape_,kernel_stride_,kernel_pad_);
        //std::cout<< "weight2col_map:"<<std::endl;
        //weight2col_map_->display();
        col2weight_map_ = get_col2data_map_index(weight_shape_,kernel_shape_,kernel_stride_,kernel_pad_,weight2col_map_);
        //std::cout<< "col2weight_map:"<<std::endl;
        //col2weight_map_->display();
        //std::cout<<"weight:"<<std::endl;
        //this->blobs_[0]->display();
    }

    template <typename Dtype>
    void ExtendedConvolutionLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
    {
        bool input_shape_change = false;
        ExtendedConvolutionParameter conv_param = this->layer_param_.extended_convolution_param();
        for( int i=0; i<num_spatial_axes_; i++ )
        {
            if( input_shape_.size()<=i )
            {
                input_shape_.push_back(bottom[0]->shape(i+1));
                input_shape_change = true;
            }
            else if(  input_shape_[i] != bottom[0]->shape(i+1) )
            {
                input_shape_[i] = bottom[0]->shape(i+1);
                input_shape_change = true;
            }
        }
        
        //if bottom.shape not equal previous shape
        if( input_shape_change )
        {
            output_shape_.clear();
            output_sample_total_size_ = 1;
            input_sample_total_size_  = 1;
            sample_col_num_ = 1;
            output_shape_.push_back(bottom[0]->shape(0));
            //output_shape_.insert(output_shape_.begin()+1,weight_output_shape_.begin(), weight_output_shape_.end());
            for( int i=0; i<conv_param.weight_dim_size(); i++)
            {
                output_shape_.push_back(weight_output_shape_[i]);
                output_sample_total_size_ *= weight_output_shape_[i];
            }

            for( int i=0; i<num_spatial_axes_; i++ )
            {
                int tmp = (input_shape_[i]+2*pad_[i]-kernel_shape_[i])/stride_[i]+1;
                input_sample_total_size_ *= input_shape_[i];
                CHECK_GE( tmp, 0 );
                //if( tmp>=1 )
                {
                    output_shape_.push_back(tmp);
                    output_sample_total_size_ *= tmp;
                    sample_col_num_ *= tmp;
                }
            }
            
            vector<int> data_cols_shape_;
            data_cols_shape_.push_back(kernel_total_size_);
            data_cols_shape_.push_back(sample_col_num_);
            data_cols_.Reshape(data_cols_shape_);
            data2col_map_ = get_data2col_map_index(input_shape_,kernel_shape_,stride_,pad_);
            //std::cout<<"data2col_map:"<<std::endl;
            //data2col_map_->display();
            col2data_map_ = get_col2data_map_index(input_shape_,kernel_shape_,stride_,pad_,data2col_map_);
            //std::cout<<"col2data_map:"<<std::endl;
            //col2data_map_->display();
        }

        output_shape_[0] = bottom[0]->shape(0);
        sample_num_ = output_shape_[0];
        top[0]->Reshape(output_shape_);
    }

    template <typename Dtype>
    void ExtendedConvolutionLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
    {
        gen_weight_col(false);
        const Dtype* weight = weight_cols_.cpu_data();
        //std::cout<<"weight_cols:"<<std::endl;
        //weight_cols_.display();
        const Dtype *bottom_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        for ( int n = 0; n < sample_num_; ++n ){
            this->forward_cpu_gemm( bottom_data + n * input_sample_total_size_, weight,
                top_data + n * output_sample_total_size_);
        }
        //std::cout<< "top data:" << std::endl;
        //top[0]->display();
    }

    template <typename Dtype>
    void ExtendedConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        gen_weight_col(false);
        const Dtype* weight = weight_cols_.cpu_data();
        Dtype* weight_diff = weight_cols_.mutable_cpu_diff();
        //std::cout<<"weight diff before:"<<std::endl;
        //this->blobs_[0]->display(true);
        caffe_set( weight_cols_.count(), static_cast<Dtype>(0), weight_diff );
        for (int i = 0; i < top.size(); ++i) {
            const Dtype* top_diff = top[i]->cpu_diff();
           // std::cout<<"top diff:"<<std::endl;
            //top[i]->display(true);
            const Dtype* bottom_data = bottom[i]->cpu_data();
            //std::cout<<"bottom_data:"<<std::endl;
            //bottom[i]->display();
            Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
        
            if (this->param_propagate_down_[0] || propagate_down[i]) {
                for (int n = 0; n < sample_num_; ++n) {
                    // gradient w.r.t. weight. Note that we will accumulate diffs.
                    if (this->param_propagate_down_[0]) {
                    this->weight_cpu_gemm(bottom_data + n * input_sample_total_size_,
                        weight_diff, top_diff + n * output_sample_total_size_);
                    }
                    // gradient w.r.t. bottom data, if necessary.
                    if (propagate_down[i]) {//
                        this->backward_cpu_gemm(top_diff + n * output_sample_total_size_, weight,
                            bottom_diff + n * input_sample_total_size_);
                    }
                }
            }
            #if 0
            std::cout<<"bottom diff:"<<std::endl;
            bottom[i]->display(true);
            #endif
        }
 
        gen_weight_diff(false);
        #if 0
        std::cout<<"weight:"<<std::endl;
        this->blobs_[0]->display(false);
        std::cout<<"weight col:"<<std::endl;
        weight_cols_.display();

        std::cout<<"weight diff:"<<std::endl;
        this->blobs_[0]->display(true);
        std::cout<<"weight col diff:"<<std::endl;
        weight_cols_.display(true);
        #endif

    }

    template <typename Dtype>
    void ExtendedConvolutionLayer<Dtype>::increase(vector<int>& index, vector<int>& boundary)
    {
        for( int i=index.size()-1; i>=0; i-- )
        {
            int res = index[i]+1;
            index[i] = res%boundary[i]; 
            if( res<boundary[i] )
            {
                break;
            }
        }
    }

    template <typename Dtype>
    int ExtendedConvolutionLayer<Dtype>::get_offset( 
             vector<int>& col_index, 
             vector<int>& kernel_index,
             vector<int>& data_size,
             vector<int>& pad,
             vector<int>& stride )
    {
        CHECK_EQ(col_index.size(), kernel_index.size());
        CHECK_EQ(col_index.size(), data_size.size());
        CHECK_EQ(col_index.size(), pad.size());

        int base = 1;
        int offset = 0;
        for( int i=col_index.size()-1; i>=0; i--)
        {
            int dim_offset = col_index[i]*stride[i] + kernel_index[i];
            if( dim_offset<pad[i] || dim_offset>=data_size[i]+pad[i] )
            {
                //out of boundary
                return -1;
            }
            offset += (dim_offset-pad[i]) * base;
            base *= data_size[i];
        }
        return offset;
    }

    template <typename Dtype>
    shared_ptr<Blob<int> > ExtendedConvolutionLayer<Dtype>::get_data2col_map_index(
        vector<int>& data_size, 
        vector<int>& kernel_size,
        vector<int>& stride,
        vector<int>& pad)
    {
        CHECK_EQ(data_size.size(), kernel_size.size());
        CHECK_EQ(data_size.size(), stride.size());
        CHECK_EQ(data_size.size(), pad.size());
        for( int i=0; i<data_size.size(); i++ )
        {
            CHECK_LE(kernel_size[i], data_size[i]+2*pad[i]);
        }
        
        int kernel_ele_num = 1;
        int col_num = 1;
        vector<int> data2kernel_size;
        vector<int> map_shape;
        vector<int> col_dim_index;
        vector<int> kernel_dim_index;
        shared_ptr<Blob<int> > index_map;
        index_map.reset( new Blob<int>()) ;

        for( int i=0; i<kernel_size.size(); i++ )
        {
            int tmp = (data_size[i]-kernel_size[i]+2*pad[i])/stride[i] + 1;
            kernel_ele_num *= kernel_size[i];
            col_num *= tmp;
            data2kernel_size.push_back(tmp);
            col_dim_index.push_back(0);
            kernel_dim_index.push_back(0);
        }
        //data2kernel_size.insert( data2kernel_size.begin(),kernel_ele_num );

        map_shape.push_back(kernel_ele_num );
        map_shape.push_back(col_num);
        index_map->Reshape(map_shape);

        int *index_map_data = index_map->mutable_cpu_data();
        int *index_map_row_start = index_map_data;
        for( int i=0; i<kernel_ele_num; i++ )
        {
            for( int j=0; j<col_num; j++ )
            {
                int  data_offset = get_offset( col_dim_index, kernel_dim_index, data_size,  pad, stride );
                index_map_row_start[j] = data_offset;
                increase(col_dim_index, data2kernel_size );
            }

            //index_map->display();
            increase(kernel_dim_index, kernel_size );
            col_dim_index.clear();
            for( int k=0; k<kernel_size.size(); k++ )
            {
                col_dim_index.push_back(0);
            }
            index_map_row_start += col_num;
        }
        
        return index_map;
    }

    template <typename Dtype>
    shared_ptr<Blob<int> > ExtendedConvolutionLayer<Dtype>::get_col2data_map_index(vector<int>& data_size, 
                                                 vector<int>& kernel_size,
                                                 vector<int>& stride,
                                                 vector<int>& pad,
                                                 shared_ptr<Blob<int> >& data2col_map )
    {
        CHECK_EQ(data_size.size(), kernel_size.size());
        CHECK_EQ(data_size.size(), stride.size());
        CHECK_EQ(data_size.size(), pad.size());

        vector<int> index_map_size;
        int total_data_size = 1;
        for( int i=0; i<data_size.size(); i++ )
        {
            CHECK_LE(kernel_size[i], data_size[i]+2*pad[i]);
            total_data_size *= data_size[i];
        }
        
        int data_ele_repli_time = 1;
        for( int i=0; i<kernel_size.size(); i++)
        {
            int dim_size = 2*kernel_size[i]-1;
            int dim_repli_time = (dim_size-kernel_size[i])/stride[i] + 1;
            data_ele_repli_time *= dim_repli_time;
            index_map_size.push_back(data_size[i]);
        }
        index_map_size.clear();
        index_map_size.push_back(total_data_size);
        index_map_size.push_back(data_ele_repli_time);

        shared_ptr<Blob<int> > index_map;
        index_map.reset(new Blob<int>());
        shared_ptr<Blob<int> > index_map_count; 
        index_map_count.reset(new Blob<int>());
        index_map->Reshape(index_map_size);
        index_map_count->Reshape(data_size);

        int *index_map_data = index_map->mutable_cpu_data();
        int *index_map_count_data = index_map_count->mutable_cpu_data();
        int *data2col_map_data = data2col_map->mutable_cpu_data();

        for( int i=0; i<index_map->count(); i++ )
        {
            index_map_data[i] = -1;
        }

        for( int i=0; i<data2col_map->count(); i++ )
        {
            int index = data2col_map_data[i]*data_ele_repli_time;
            int curTime = index_map_count_data[data2col_map_data[i]];
            index_map_data[index+curTime] = i;
            index_map_count_data[data2col_map_data[i]] += 1; 
        }
        std::cout<<"data2col_map:"<<std::endl;
        data2col_map->display();
        std::cout<<"index_map:"<<std::endl;
        index_map->display();
        return index_map;
    }

    #ifdef CPU_ONLY
    STUB_GPU(ExtendedConvolutionLayer);
    #endif
    REGISTER_LAYER_CLASS(ExtendedConvolution);
    INSTANTIATE_CLASS(ExtendedConvolutionLayer);
}