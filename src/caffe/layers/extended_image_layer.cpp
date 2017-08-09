#include "caffe/layers/extended_image_layer.hpp"

namespace caffe {
    template <typename Dtype>
    void ExtendedImageLayer<Dtype>::LayerSetup(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
    {
        ExtendedImageParameter extended_image_param = this->layer_param_.extended_image_param();
        width_ = extended_image_param.width();
        height_ = extended_image_param.height();
        bcolor_ = extended_image_param.color();

        transform_param_ = extended_image_param.transformation_param();
    }

    template <typename Dtype>
    void ExtendedImageLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
    {
        image_.Reshape(1, width_,height_, bcolor_?3:1);
        top[0]->ReshapeLike(image_);
    }

    template <typename Dtype>
    void ExtendedImageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
    {
        //random generate image
        FillerParameter filler_param;
        filler_param.set_min(0.);
        filler_param.set_max(255);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(&image_);

        //apply transferlation
        data_transformer_->Transform( &image_, top[0] );
    }

    template <typename Dtype>
    void ExtendedImageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
    {
        double scale_factor = transform_param_.scale();
        //backword according to transfor params
        for( int i=0; i<top.size(); i++ )
        {
            Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();
            const Dtype *top_diff = top[i]->cpu_diff();
            for( int j=0; j<top[i]->count(); j++ )
            {
                bottom_diff[j] = scale_factor * top_diff[j];
            }
        }
    }

    #ifdef CPU_ONLY
    STUB_GPU(ExtendedImageLayer);
    #endif

    INSTANTIATE_CLASS(ExtendedImageLayer);
    REGISTER_LAYER_CLASS(ExtendedImage);
}
