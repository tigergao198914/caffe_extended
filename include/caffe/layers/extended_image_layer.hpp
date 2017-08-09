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

  virtual void LayerSetup( const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ExtendedImage"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

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
   Blob<Dtype>  image_;
   TransformationParameter transform_param_;
   shared_ptr<DataTransformer<Dtype> > data_transformer_;
};
};

#endif