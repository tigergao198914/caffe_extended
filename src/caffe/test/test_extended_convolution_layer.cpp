#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/extended_convolution_layer.hpp"

#ifdef USE_CUDNN
//#include "caffe/layers/cudnn_extended_convolution_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#if 0
template <typename Dtype>
Blob<Dtype>* getWeight( Blob<Dtype> &weight, ExtendedConvolutionParameter* conv_param, int spatial_dim)
{
    vector<int> kernel_dims;
    vector<int> weight_dims;
    vector<int> feature_pad;
    vector<int> feature_stride;

    int kernel_dim_size = conv_param->kernel_size_size();
    int weight_dim_size = conv_param->weight_dim_size();
    int feature_pad_size = conv_param->feature_pad_size();
    int feature_stride_size = conv_param->feature_stride_size();

    int kernel_total_size = 1;
    for( int i = 0; i<spatial_dim; i++)
    {
        if(kernel_dim_size==1 )
        {
             kernel_dims.push_back( conv_param->kernel_size(0) ); 
             kernel_total_size *= conv_param->kernel_size(0);
        }
        else
        {
            kernel_dims.push_back( conv_param->kernel_size(i) ); 
            kernel_total_size *= conv_param->kernel_size(i);
        }
    }    

    for( int i = 0; i<weight_dim_size; i++)
    {
        weight_dims.push_back( conv_param->weight_dim(i) ); 
    }

    for( int i = 0; i<weight_dim_size; i++)
    {
        if( feature_pad_size==1 )
        {
            feature_pad.push_back( conv_param->feature_pad(0) );
        }
        else
        {
            feature_pad.push_back( conv_param->feature_pad(i) );
        }
    }

    for( int i=0; i<weight_dim_size; i++)
    {
        if( feature_stride_size==1 )
        {
            feature_stride.push_back( conv_param->feature_stride(0));
        }
        else
        {
            feature_stride.push_back( conv_param->feature_stride(i));
        }
    }

    
    Blob<Dtype> *weight_col = new Blob<Dtype>();
    vector<int> weight_col_dims;
    weight_col_dims.push_back( kernel_total_size );
    int total_feature_num = 1;
    for( int i=0; i<weight_dim_size; i++ )
    {
        int weight_size         = weight_dims[i];
        int feature_size        = kernel_dims[i];
        int feature_pad_size    = feature_pad[i];
        int feature_stride_size = feature_stride[i];
        int weight_num = (weight_size+2*feature_pad_size-feature_size)/feature_stride_size + 1;
        weight_col_dims.push_back(weight_num);
        total_feature_num *= weight_num;
    }

    for( int i=weight_dim_size; i<spatial_dim; i++ )
    {
        weight_col_dims.push_back(1);
    }

    weight_col->Reshape(weight_col_dims);
    Dtype *weight_col_mutable_data = weight_col->mutable_cpu_data();
    for( int i=0; i<total_feature_num; i++ )
    {
        int col_idx = i;
        int tmp_col_idx  = col_idx;
        //get col offset
        vector<int> col_idx_offset;
        col_idx_offset.resize(weight_dim_size);
        for( int k=weight_dim_size-1; k>=0; k-- )
        {
            int tmp_weight_dim = weight_col_dims[k];
            col_idx_offset[k] = (tmp_col_idx % tmp_weight_dim) * feature_stride[k];
            //std::cout<< "col_idx_offset:" << col_idx_offset[k] << std::endl;
            tmp_col_idx /= tmp_weight_dim;
        }

        for( int j=0; j<kernel_total_size; j++)
        {
            int kernel_idx = j;
            int tmp_kernel_idx = kernel_idx;
            //get kernel_offset
            vector<int> kernel_idx_offset;
            kernel_idx_offset.resize(spatial_dim);

            //std::cout<< "feature_idx:" << i << ",kernel_idx:" << j <<std::endl;

            for( int k=spatial_dim-1; k>=0 ; k--)
            {
                int tmp_kernel_dim = kernel_dims[k];
                kernel_idx_offset[k] = tmp_kernel_idx % tmp_kernel_dim;
                tmp_kernel_idx /= tmp_kernel_dim;
                //std::cout<< "kernel_size:"<< kernel_idx_offset[k] <<std::endl;
            }

            for( int t=0; t<weight_dim_size; t++ )
            {
                kernel_idx_offset[t] += col_idx_offset[t];
            }

            weight_col_mutable_data[ i*total_feature_num + j ] = weight.data_at(kernel_idx_offset);
        }

    }

    return weight_col;
}
#endif

template <typename Dtype>
Blob<Dtype>* conv( Blob<Dtype> &weight, Blob<Dtype> &input, ExtendedConvolutionParameter* conv_param, int spatial_dim)
{
    vector<int>  kernel_shape;
    vector<int>  stride;
    vector<int>  pad;
    vector<int>  feature_stride;
    vector<int>  feature_pad;
    vector<int>  weight_shape;
    vector<int>  input_shape;
    int sample_dim = 1;
    for( int i=0; i<spatial_dim; i++ )
    {
        if(i<conv_param->kernel_size_size())
        {
            kernel_shape.push_back( conv_param->kernel_size(i) );
        }
        else
        {
            kernel_shape.push_back( conv_param->kernel_size(0) );
        }

        if(i<conv_param->stride_size())
        {
            stride.push_back( conv_param->stride(i) );
        }
        else if(conv_param->stride_size()==1)
        {
            stride.push_back( conv_param->stride(0) );
        }
        else
        {
            stride.push_back( 1 );
        }

        if(i<conv_param->pad_size())
        {
            pad.push_back( conv_param->pad(i) );
        }
        else if( conv_param->pad_size()==1 )
        {
            pad.push_back( conv_param->pad(0) );
        }
        else
        {
            pad.push_back( 0 );
        }

        if(i<conv_param->feature_stride_size())
        {
            feature_stride.push_back(conv_param->feature_stride(i));
        }
        else
        {
            feature_stride.push_back(1);
        }

        if(i<conv_param->feature_pad_size())
        {
            feature_pad.push_back(conv_param->feature_pad(i));
        }
        else
        {
            feature_pad.push_back(0);
        }

        if( i<conv_param->weight_dim_size() )
        {
            weight_shape.push_back( conv_param->weight_dim(i) );
        }
        else
        {
            weight_shape.push_back( kernel_shape[i] );
        }

        input_shape.push_back( input.shape(i+1) );
        sample_dim*= input.shape(i+1);
    }

    int sample_col_num = 1;
    int feature_num = 1;
    int kernel_dim = 1;

    vector<int> sample_shape;
    vector<int> weight_map_shape;
    for( int i=0; i<spatial_dim; i++ )
    {
        int tmp = ((input_shape[i] + 2*pad[i] - kernel_shape[i])/stride[i] + 1);
        sample_col_num *= tmp;
        sample_shape.push_back(tmp);
    }
    for( int i=0; i<spatial_dim; i++ )
    {
        int tmp = ((weight_shape[i] + 2*feature_pad[i] - kernel_shape[i])/feature_stride[i]+1);
        feature_num *= tmp;
        weight_map_shape.push_back(tmp);
    }
    for( int i=0; i<spatial_dim; i++ )
    {
        kernel_dim *= kernel_shape[i];
    }

    vector<int> output_shape;
    output_shape.push_back(input.shape(0));
    output_shape.insert(output_shape.end(), weight_map_shape.begin(), weight_map_shape.begin()+conv_param->weight_dim_size() );
    output_shape.insert(output_shape.end(), sample_shape.begin(), sample_shape.end() );
    //output_shape.push_back(feature_num);
    //output_shape.push_back(sample_col_num);
    Blob<Dtype> *output = new Blob<Dtype>();
    output->Reshape(output_shape);

    
    const Dtype* weight_data = weight.cpu_data();
    for( int c = 0; c<input.shape(0); c++ )
    {
        Dtype * output_data = output->mutable_cpu_data() + c*feature_num*sample_col_num;
        const Dtype* input_data  = input.cpu_data() + c*sample_dim;
        for( int f=0; f<feature_num; f++ )
        {
            vector<int> feature_offset(spatial_dim);
            int tmp_f = f;
            for( int i=spatial_dim-1; i>=0; i-- )
            {
                feature_offset[i] = tmp_f%weight_map_shape[i];
                feature_offset[i] = feature_offset[i]*feature_stride[i]-feature_pad[i];
                tmp_f /= weight_map_shape[i];
            }

            for( int j=0; j<sample_col_num; j++ )
            {
                vector<int> sample_offset(spatial_dim);
                int tmp_j = j;
                for( int i=spatial_dim-1; i>=0; i-- )
                {
                    sample_offset[i] = tmp_j%sample_shape[i];
                    sample_offset[i] = sample_offset[i]*stride[i]-pad[i];
                    tmp_j /= sample_shape[i];
                }

                Dtype value = 0;
                for( int k=0; k<kernel_dim; k++ )
                {
                    vector<int> kernel_offset(spatial_dim);
                    vector<int> data_offset(spatial_dim);
                    vector<int> weight_offset(spatial_dim);
                    int k_tmp = k;
                    for( int i=spatial_dim-1; i>=0; i-- )
                    {
                        kernel_offset[i] = k_tmp%kernel_shape[i];
                        k_tmp /=kernel_shape[i];
                        data_offset[i]   = sample_offset[i]  + kernel_offset[i];
                        weight_offset[i] = feature_offset[i] + kernel_offset[i];
                    }
                    int weight_idx = 0;
                    int weight_base = 1;
                    int data_idx = 0;
                    int data_base = 1;
                    for( int i=spatial_dim-1; i>=0; i-- )
                    {
                        weight_idx  += weight_offset[i]*weight_base;
                        weight_base *= weight_shape[i];
                        data_idx    += data_offset[i]*data_base;
                        data_base   *= input_shape[i]; 
                    }
                    value += ( weight_data[weight_idx] * input_data[data_idx] );
                }
                output_data[f*sample_col_num+j] = value;
            }
        }
    }
    return output;
}

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, ExtendedConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
        vector<int> kernel_shape;
}

template <typename TypeParam>
class ExtendedConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ExtendedConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    #if 0
    for( int i = 0; i<blob_bottom_->count() ; i++)
    {
       blob_bottom_->mutable_cpu_data()[i] =1;
    }
    #endif
    
  }

  virtual ~ExtendedConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ExtendedConvolutionLayerTest, TestDtypesAndDevices);

#if 0
TYPED_TEST(ExtendedConvolutionLayerTest, TestData2col_cpu) {
    int num_spatial_axes = 3;
    int *im_shape = new int[num_spatial_axes];
    int *kernel_shape = new int[num_spatial_axes];
    int *pad = new int[num_spatial_axes];
    int *stride = new int[num_spatial_axes];
    int *col_shape = new int[num_spatial_axes+1];

    im_shape[0] = 3;
    im_shape[1] = 7;
    im_shape[2] = 5;
    kernel_shape[0] = 3;
    kernel_shape[1] = 4;
    kernel_shape[2] = 2;
    pad[0] = 1;
    pad[1] = 0;    
    pad[2] = 0;
    stride[0] = 2;
    stride[1] = 2;
    stride[2] = 1;
    col_shape[0] = kernel_shape[0]*kernel_shape[1]*kernel_shape[2];
    col_shape[1] = 2;
    col_shape[2] = 2;
    col_shape[3] = 4;

    int data_size     = im_shape[0]*im_shape[1]*im_shape[2];
    int data_col_size = col_shape[0]*col_shape[1]*col_shape[2]*col_shape[3];
    double *data     = new double[data_size];
    double *data_col = new double[data_col_size];

    for( int i=0; i<data_size; i++ )
    {
        data[i] = i+1;
    }

    data2col_cpu<double>( data, num_spatial_axes, im_shape, col_shape, kernel_shape,  pad, stride, data_col );

#if 0
    for( int i=0; i<data_col_size ;i++)
    {
        if( i%(col_shape[1]*col_shape[2]*col_shape[3]) == 0)
        {
            std::cout<<std::endl;
        }
       std::cout<<"  "<<std::setw(3)<< data_col[i];
    }
#endif
}
#endif

#if 0
TYPED_TEST(ExtendedConvolutionLayerTest, TestGetWeight) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ExtendedConvolutionParameter* convolution_param =
        layer_param.mutable_extended_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(2);
    convolution_param->add_weight_dim(9);
    convolution_param->add_weight_dim(9);
    convolution_param->add_feature_pad(0);
    convolution_param->add_feature_pad(0);
    convolution_param->add_feature_stride(2);
    convolution_param->add_feature_stride(2);

   shared_ptr<Layer<Dtype> > layer(
        new ExtendedConvolutionLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    Blob<Dtype>* weight = ((ExtendedConvolutionLayer<Dtype>*)(layer.get()))->getWeight();

    Blob<Dtype>* weight_col_ref = getWeight<Dtype>( *weight, convolution_param, 3 );
    Blob<Dtype>* weight_col = ((ExtendedConvolutionLayer<Dtype>*)(layer.get()))->getFeatureCol();

    const Dtype * weight_col_ref_cpu_data = weight_col_ref->cpu_data();
    const Dtype * weight_col_cpu_data = weight_col->cpu_data();

    EXPECT_EQ( weight_col->num_axes(), weight_col_ref->num_axes() );
    for( int i=0; i<weight_col->count(); i++ )
    {
       EXPECT_EQ(  weight_col_ref_cpu_data[i],weight_col_cpu_data[i] );
    }
}
#endif

TYPED_TEST(ExtendedConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ExtendedConvolutionParameter* convolution_param =
      layer_param.mutable_extended_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);

  convolution_param->add_weight_dim(9);
  convolution_param->add_weight_dim(9);
  convolution_param->add_weight_dim(3);
  convolution_param->add_feature_pad(0);
  convolution_param->add_feature_pad(0);
  convolution_param->add_feature_pad(0);
  convolution_param->add_feature_stride(2);
  convolution_param->add_feature_stride(2);
  convolution_param->add_feature_stride(1);
  
  shared_ptr<Layer<Dtype> > layer(
      new ExtendedConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  //vector<int> top_shape = this->blob_top_->shape();
  EXPECT_EQ(  this->blob_top_->shape()[0], 2 );
  EXPECT_EQ(  this->blob_top_->shape()[1], 4 );
  EXPECT_EQ(  this->blob_top_->shape()[2], 4 );
  EXPECT_EQ(  this->blob_top_->shape()[3], 1 );
  EXPECT_EQ(  this->blob_top_->shape()[4], 1 );
  EXPECT_EQ(  this->blob_top_->shape()[5], 2 );
  EXPECT_EQ(  this->blob_top_->shape()[6], 1 );
}

TYPED_TEST(ExtendedConvolutionLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ExtendedConvolutionParameter* convolution_param =
        layer_param.mutable_extended_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(2);
    convolution_param->add_weight_dim(9);
    convolution_param->add_weight_dim(9);
    convolution_param->add_weight_dim(3);
    convolution_param->add_feature_pad(0);
    convolution_param->add_feature_pad(0);
    convolution_param->add_feature_pad(0);
    convolution_param->add_feature_stride(2);
    convolution_param->add_feature_stride(2);
    convolution_param->add_feature_stride(1);
    convolution_param->mutable_weight_filler()->set_type("gaussian");

    shared_ptr<Layer<Dtype> > layer(
        new ExtendedConvolutionLayer<Dtype>(layer_param));
    layer->SetUp( this->blob_bottom_vec_, this->blob_top_vec_ );

    Blob<Dtype> * weight = layer->blobs()[0].get();

    layer->Forward( this->blob_bottom_vec_, this->blob_top_vec_ );

    Blob<Dtype> *top_ref =  conv( *weight, *(this->blob_bottom_vec_[0]), 
        convolution_param, this->blob_bottom_vec_[0]->num_axes()-1);

    vector<int> top_shape  = this->blob_top_vec_[0]->shape();
    EXPECT_EQ( top_shape.size(), 7 );

    const Dtype *top_data_ref = top_ref->cpu_data();
    const Dtype *top_data = this->blob_top_vec_[0]->cpu_data();

    for( int i=0; i<this->blob_top_vec_[0]->count(); i++ )
    {
        EXPECT_NEAR(top_data[i], top_data_ref[i], 1e-4);
    }
}


TYPED_TEST(ExtendedConvolutionLayerTest, TestForward_mnist) {
    typedef typename TypeParam::Dtype Dtype;

    //init params of extended convolutional network
    LayerParameter layer_param_extended;
    ExtendedConvolutionParameter* extended_convolution_param =
        layer_param_extended.mutable_extended_convolution_param();
    extended_convolution_param->add_kernel_size(1);
    extended_convolution_param->add_kernel_size(5);
    extended_convolution_param->add_kernel_size(5);
    extended_convolution_param->add_stride(1);
    extended_convolution_param->add_stride(1);
    extended_convolution_param->add_stride(1);
    extended_convolution_param->add_weight_dim(1);
    extended_convolution_param->add_weight_dim(25);
    extended_convolution_param->add_weight_dim(25);
    extended_convolution_param->add_feature_pad(0);
    extended_convolution_param->add_feature_pad(0);
    extended_convolution_param->add_feature_pad(0);
    extended_convolution_param->add_feature_stride(1);
    extended_convolution_param->add_feature_stride(5);
    extended_convolution_param->add_feature_stride(5);
    extended_convolution_param->mutable_weight_filler()->set_type("gaussian");
    //extended_convolution_param->mutable_weight_filler()->set_value(1);

    //init params of convolutional network
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(5);
    convolution_param->add_stride(1);
    convolution_param->set_num_output(25);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    //convolution_param->mutable_weight_filler()->set_value(1);



    //init input data, store in blob_bottom_vec
    Blob<Dtype>* const data = new Blob<Dtype>(1,1,28,28);
    vector<Blob<Dtype>*> blob_bottom_vec;
    blob_bottom_vec.push_back(data);
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(data);

    //init output data, store in blob_top_vec
    Blob<Dtype>* const extended_output = new Blob<Dtype>();
    vector<Blob<Dtype>*> extended_blob_top_vec;
    extended_blob_top_vec.push_back(extended_output);

    Blob<Dtype>* const output = new Blob<Dtype>();
    vector<Blob<Dtype>*> blob_top_vec;
    blob_top_vec.push_back(output);

    //setup extended convolutional layer and forword
    shared_ptr<Layer<Dtype> > extended_layer(
        new ExtendedConvolutionLayer<Dtype>(layer_param_extended));
    extended_layer->SetUp(   blob_bottom_vec, extended_blob_top_vec );
    extended_layer->Forward( blob_bottom_vec, extended_blob_top_vec );

    //setup convolutional layer and forword
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayer<Dtype>(layer_param));
    layer->SetUp(   blob_bottom_vec, blob_top_vec );

    const Dtype *extended_weight_data = extended_layer->blobs()[0]->cpu_data();
    Dtype *weight_data = layer->blobs()[0]->mutable_cpu_data();
    for( int i=0; i<25; i++ )
    {
        for( int j=0; j<25; j++ )
        {
            std::cout<< "index:"<< i*25+j << "; extended index:" <<  (i/5*5+j/5)*25+(i%5)*5+j%5 << std::endl;
            weight_data[i*25+j] = extended_weight_data[ (i/5*5+j/5)*25+(i%5)*5+j%5 ];
        }
    }
    //extended_layer->blobs()[0]->display();
    //layer->blobs()[0]->display();
    

    layer->Forward( blob_bottom_vec, blob_top_vec );

    //check two output with same input and param
    const Dtype *extended_output_data  = extended_output->cpu_data();
    const Dtype *output_data = output->cpu_data();
    int count = extended_blob_top_vec[0]->count();
    for( int i=0; i<count; i++ )
    {
        EXPECT_NEAR(extended_output_data[i], output_data[i], 1e-4);
    }

}

TYPED_TEST(ExtendedConvolutionLayerTest, TestBackword) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ExtendedConvolutionParameter* convolution_param =
        layer_param.mutable_extended_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(2);
    convolution_param->add_weight_dim(9);
    convolution_param->add_weight_dim(9);
    convolution_param->add_weight_dim(3);
    convolution_param->add_feature_pad(0);
    convolution_param->add_feature_pad(0);
    convolution_param->add_feature_pad(0);
    convolution_param->add_feature_stride(2);
    convolution_param->add_feature_stride(2);
    convolution_param->add_feature_stride(1);
    convolution_param->mutable_weight_filler()->set_type("gaussian");

    ExtendedConvolutionLayer<Dtype> layer(layer_param);

    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, false);
}

TYPED_TEST(ExtendedConvolutionLayerTest, TestBackword1) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ExtendedConvolutionParameter* convolution_param =
        layer_param.mutable_extended_convolution_param();
    convolution_param->add_kernel_size(1);
    convolution_param->add_kernel_size(3);
    convolution_param->add_kernel_size(2);
    convolution_param->add_stride(1);
    convolution_param->add_stride(1);
    convolution_param->add_stride(1);
    convolution_param->add_weight_dim(1);
    convolution_param->add_weight_dim(9);
    convolution_param->add_weight_dim(9);
    convolution_param->add_feature_pad(0);
    convolution_param->add_feature_stride(2);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    
    ExtendedConvolutionLayer<Dtype> layer(layer_param);

    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, false);
}

};