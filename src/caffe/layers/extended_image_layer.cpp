#include "caffe/layers/extended_image_layer.hpp"

namespace caffe {
    template <typename Dtype>
    void ExtendedImageLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
    {
        ExtendedImageParameter extended_image_param = this->layer_param_.extended_image_param();
        int count = 1;
        width_ = extended_image_param.width();
        height_ = extended_image_param.height();
        bcolor_ = extended_image_param.color();
        for( int i=0; i<extended_image_param.label_size_size(); i++ )
        {
            label_size_.push_back(extended_image_param.label_size(i));
            count *= label_size_[i];
        }
        transform_param_ = extended_image_param.transform_param();
        data_transformer_.reset( new DataTransformer<Dtype>(transform_param_, TRAIN) );

        label_ = rand()%count;
      
        image_.reset(new Blob<Dtype>());
        image_->Reshape(1, bcolor_?3:1, width_,height_);
        //random generate image
        FillerParameter filler_param;
        filler_param.set_min(0.);
        filler_param.set_max(255.);
        UniformFiller<Dtype> filler(filler_param);
        //GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(image_.get());
 
        Dtype * image_data = image_->mutable_cpu_data();
 #if 0      
        for( int i=0; i<image_->count()/3; i++ )
        {
            image_data[i] = 1;
        }
        for( int i=image_->count()/3; i<image_->count()*2/3; i++ )
        {
            image_data[i] = 255;
        }

        for( int i=image_->count()*2/3; i<image_->count(); i++ )
        {
            image_data[i] = 125;
        }
#endif     
        for( int i=0; i<image_->count(); i++ )
        {
            image_data[i] = 0;
        }
 

        this->blobs_.resize(1);
        //initalize and fill the weight
        this->blobs_[0] =  image_ ;
        this->param_propagate_down_.resize(1,true);
    }

    template <typename Dtype>
    void ExtendedImageLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
    {

        top[0]->ReshapeLike(*image_.get());
        std::vector<int> s;
        s.push_back(top[0]->shape(0));
        top[1]->Reshape(s);
    }

    template <typename Dtype>
    void ExtendedImageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
    {
        //apply transferlation
        data_transformer_->Transform( image_.get(), top[0] );
        Dtype* label =  top[1]->mutable_cpu_data();
        //caffe_memset( (void*)label, 0x0, top[1]->count());
        for( int i=0; i<top[1]->count(); i++ )
        {
            label[i] = (Dtype)label_;
        }
    }

    template <typename Dtype>
    void ExtendedImageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
    {
        //top[0]->display(true);
        double scale_factor = transform_param_.scale();
        //backword according to transfor params
        Dtype *bottom_diff = image_->mutable_cpu_data();
        const Dtype *top_diff = top[0]->cpu_diff();
        for( int j=0; j<top[0]->count(); j++ )
        {
            bottom_diff[j] -= 0.003* scale_factor * top_diff[j];
        }
    }

    #ifdef CPU_ONLY
    STUB_GPU(ExtendedImageLayer);
    #endif

    INSTANTIATE_CLASS(ExtendedImageLayer);
    REGISTER_LAYER_CLASS(ExtendedImage);
}
