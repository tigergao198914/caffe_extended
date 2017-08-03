#include <vector>
#include <stdlib.h>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/extended_pool_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ExtendedPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ExtendedPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ExtendedPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ExtendedPoolingLayerTest, TestDtypesAndDevices);
  // Test for 2x 2 square pooling layer
TYPED_TEST(ExtendedPoolingLayerTest, TestSetup)
{     
    typedef typename TypeParam::Dtype Dtype;                                                                             
    LayerParameter layer_param;
    ExtendedPoolingParameter* pooling_param = layer_param.mutable_extended_pooling_param();
    pooling_param->add_kernel_size(2);
    pooling_param->add_pad(0);
    pooling_param->add_stride(1);
    //pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);

    this->blob_bottom_->Reshape(2, 4, 3, 5);

    for (int i = 0; i < this->blob_bottom_->count(); i++) 
    {
      this->blob_bottom_->mutable_cpu_data()[i] = i;
    }

    ExtendedPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->shape(0), 2);
    EXPECT_EQ(this->blob_top_->shape(1), 3);
    EXPECT_EQ(this->blob_top_->shape(2), 2);
    EXPECT_EQ(this->blob_top_->shape(3), 4);
} 


TYPED_TEST(ExtendedPoolingLayerTest, TestForward)
{
    typedef typename TypeParam::Dtype Dtype;                                                                             
    LayerParameter layer_param;
    ExtendedPoolingParameter* pooling_param = layer_param.mutable_extended_pooling_param();
    pooling_param->add_kernel_size(2);
    pooling_param->add_pad(0);
    pooling_param->add_stride(1);
    //pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    this->blob_bottom_->Reshape(2, 4, 3, 5);

    for (int i = 0; i < this->blob_bottom_->count(); i++) 
    {
      this->blob_bottom_->mutable_cpu_data()[i] = i;
    }

    ExtendedPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    
    EXPECT_EQ(this->blob_top_->cpu_data()[0], 21);
    EXPECT_EQ(this->blob_top_->cpu_data()[1], 22);
    EXPECT_EQ(this->blob_top_->cpu_data()[2], 23);
    EXPECT_EQ(this->blob_top_->cpu_data()[3], 24);
    EXPECT_EQ(this->blob_top_->cpu_data()[4], 26);
    EXPECT_EQ(this->blob_top_->cpu_data()[5], 27);
    EXPECT_EQ(this->blob_top_->cpu_data()[6], 28);
    EXPECT_EQ(this->blob_top_->cpu_data()[7], 29);
    EXPECT_EQ(this->blob_top_->cpu_data()[8], 36);
    EXPECT_EQ(this->blob_top_->cpu_data()[9], 37);
    EXPECT_EQ(this->blob_top_->cpu_data()[10], 38);
    EXPECT_EQ(this->blob_top_->cpu_data()[11], 39);
    EXPECT_EQ(this->blob_top_->cpu_data()[12], 41);
    EXPECT_EQ(this->blob_top_->cpu_data()[13], 42);
    EXPECT_EQ(this->blob_top_->cpu_data()[14], 43);
    EXPECT_EQ(this->blob_top_->cpu_data()[15], 44);
    EXPECT_EQ(this->blob_top_->cpu_data()[16], 51);
    EXPECT_EQ(this->blob_top_->cpu_data()[17], 52);
    EXPECT_EQ(this->blob_top_->cpu_data()[18], 53);
    EXPECT_EQ(this->blob_top_->cpu_data()[19], 54);
    EXPECT_EQ(this->blob_top_->cpu_data()[20], 56);
    EXPECT_EQ(this->blob_top_->cpu_data()[21], 57);
    EXPECT_EQ(this->blob_top_->cpu_data()[22], 58);
    EXPECT_EQ(this->blob_top_->cpu_data()[23], 59);
}

TYPED_TEST(ExtendedPoolingLayerTest, TestForward_mnist)
{
    typedef typename TypeParam::Dtype Dtype;     
    //init params of extended pooling layer                                                                        
    LayerParameter extended_layer_param;
    ExtendedPoolingParameter* extended_pooling_param = extended_layer_param.mutable_extended_pooling_param();
    extended_pooling_param->add_kernel_size(1);
    extended_pooling_param->add_kernel_size(1);
    extended_pooling_param->add_kernel_size(1);
    extended_pooling_param->add_kernel_size(1);
    extended_pooling_param->add_kernel_size(2);
    extended_pooling_param->add_kernel_size(2);
    extended_pooling_param->add_pad(0);
    extended_pooling_param->add_stride(1);
    extended_pooling_param->add_stride(1);
    extended_pooling_param->add_stride(1);
    extended_pooling_param->add_stride(1);
    extended_pooling_param->add_stride(2);
    extended_pooling_param->add_stride(2);

    //init params of pooling layer
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    pooling_param->set_pad(0);
    pooling_param->set_stride(2);

    //init input data, store in blob_bottom_vec
    std::vector<int> extended_shape;
    extended_shape.push_back(2);
    extended_shape.push_back(1);
    extended_shape.push_back(5);
    extended_shape.push_back(5);
    extended_shape.push_back(1);
    extended_shape.push_back(24);
    extended_shape.push_back(24);
    Blob<Dtype>* const data = new Blob<Dtype>(extended_shape);
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
        new ExtendedPoolingLayer<Dtype>(extended_layer_param));
    extended_layer->SetUp(   blob_bottom_vec, extended_blob_top_vec );
    extended_layer->Forward( blob_bottom_vec, extended_blob_top_vec );

    //setup convolutional layer and forword
    std::vector<int> shape;
    shape.push_back(2);
    shape.push_back(25);
    shape.push_back(24);
    shape.push_back(24);
    data->Reshape(shape);
    shared_ptr<Layer<Dtype> > layer(
        new PoolingLayer<Dtype>(layer_param));
    layer->SetUp(   blob_bottom_vec, blob_top_vec );
    layer->Forward( blob_bottom_vec, blob_top_vec );

    //check two output with same input and param
    const Dtype *extended_output_data  = extended_output->cpu_data();
    const Dtype *output_data = output->cpu_data();
    for( int i=0; i<extended_blob_top_vec[0]->count(); i++ )
    {
        EXPECT_NEAR(extended_output_data[i], output_data[i], 1e-4);
    }
}

TYPED_TEST(ExtendedPoolingLayerTest, TestBackword) {
    typedef typename TypeParam::Dtype Dtype;                                                                             
    LayerParameter layer_param;
    ExtendedPoolingParameter* pooling_param = layer_param.mutable_extended_pooling_param();
    pooling_param->add_kernel_size(2);
    pooling_param->add_pad(0);
    pooling_param->add_stride(1);
    //pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    this->blob_bottom_->Reshape(2, 4, 3, 5);
    ExtendedPoolingLayer<Dtype> layer(layer_param);

    srand(0xf278932);
    for (int i = 0; i < this->blob_bottom_->count(); i++) 
    {
      this->blob_bottom_->mutable_cpu_data()[i] = random()%5000;
    }

    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, false);
}

TYPED_TEST(ExtendedPoolingLayerTest, TestBackword1) {
    typedef typename TypeParam::Dtype Dtype;                                                                             
    LayerParameter layer_param;
    ExtendedPoolingParameter* pooling_param = layer_param.mutable_extended_pooling_param();
    pooling_param->add_kernel_size(1);
    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(1);
    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(1);
    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(2);
    pooling_param->add_pad(0);
    pooling_param->add_stride(2);
    //pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    vector<int> shape;
    shape.push_back(1);
    shape.push_back(1);
    shape.push_back(2);
    shape.push_back(2);
    shape.push_back(1);
    shape.push_back(3);
    shape.push_back(3);
    shape.push_back(1);
    shape.push_back(5);
    shape.push_back(5);
    this->blob_bottom_->Reshape(shape);
    ExtendedPoolingLayer<Dtype> layer(layer_param);

    srand(0xf278932);
    for (int i = 0; i < this->blob_bottom_->count(); i++) 
    {
      this->blob_bottom_->mutable_cpu_data()[i] = random()%5000;
    }

    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, false);
}

} 
