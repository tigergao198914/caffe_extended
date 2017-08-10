#ifndef EXTENDED_IMAGE_LAYER_HPP_
#define EXTENDED_IMAGE_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"

namespace caffe {
template <typename Dtype>
class ExtendedImageLayer : public Layer<Dtype> {
 public:
  explicit ExtendedImageLayer( const LayerParameter& param)
        : Layer<Dtype>(param) {}

  virtual void LayerSetUp( const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ExtendedImage"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 2; }

  void setLabel( int label ){
        label_ = label;
  }

 protected:
   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){}
   virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}

private:
   unsigned int width_;
   unsigned int height_;
   bool   bcolor_;
   int label_;
   std::vector<int> label_size_;
   Blob<Dtype>  image_;
   TransformationParameter transform_param_;
   shared_ptr<DataTransformer<Dtype> > data_transformer_;
};
};

#endif