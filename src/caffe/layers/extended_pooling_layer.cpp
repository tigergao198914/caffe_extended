#include <vector>

#include "caffe/layers/extended_pool_layer.hpp"

namespace caffe {
    template <typename Dtype>
    void ExtendedPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
    {
        ExtendedPoolingParameter pool_param = this->layer_param_.extended_pooling_param();
        axes_num_ = bottom[0]->num_axes()-1;
        if( pool_param.kernel_size_size() == 1  )
        {
            for( int i=0; i<axes_num_; i++ )
            {
               kernel_size_.push_back(pool_param.kernel_size(0));
            }
        }   
        else if( pool_param.kernel_size_size()==axes_num_ )
        {
            for( int i=0; i<axes_num_; i++ )
            {
                kernel_size_.push_back(pool_param.kernel_size(i));
            }

        }

        if( pool_param.stride_size() == 0 )
        {
            for( int i=0; i<axes_num_; i++ )
            {
                stride_.push_back(1);
            }
        }
        if( pool_param.stride_size() == 1 )
        {
            for( int i=0; i<axes_num_; i++ )
            {
                stride_.push_back(pool_param.stride(0));
            }
        }
        else if( pool_param.stride_size() == axes_num_ )
        {
            for( int i=0; i<axes_num_; i++ )
            {
                stride_.push_back(pool_param.stride(i));
            }
        }

        if( pool_param.pad_size() == 0 )
        {
            for( int i=0; i<axes_num_; i++ )
            {
                pad_.push_back(0);
            }
        }
        if( pool_param.pad_size() == 1 )
        {
            for( int i=0; i<axes_num_; i++ )
            {
                pad_.push_back(pool_param.pad(0));
            }
        }
        else if( pool_param.pad_size() == axes_num_ )
        {
            for( int i=0; i<axes_num_; i++ )
            {
                pad_.push_back(pool_param.pad(i));
            }
        }

        for( int i=0; i<axes_num_; i++ )
        {
            int input_shape = bottom[0]->shape(i+1);
            input_size_.push_back(input_shape);
            int ouput_shape = (input_shape + 2*pad_[i] - kernel_size_[i])/stride_[i] + 1;
            output_size_.push_back(ouput_shape);
        }
    }

   template <typename Dtype>
   void ExtendedPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
    {
        vector<int> tmp_shape;
        paded_input_size_.clear();
        tmp_shape.push_back(bottom[0]->shape(0));
        for( int i=0; i<output_size_.size(); i++ )
        {
            tmp_shape.push_back(output_size_[i]);
            paded_input_size_.push_back(input_size_[i]+2*pad_[i]);
        }
        top[0]->Reshape(tmp_shape);
        max_idx_.Reshape(tmp_shape);
        paded_index_map_.Reshape(paded_input_size_);
        input_offset_.Reshape(output_size_);
        kernel_offset_.Reshape(kernel_size_);

        int paded_input_ele_size = 1;
        kernel_ele_size_ = 1;
        ouput_ele_size_ = 1;
        for( int i=0; i<axes_num_; i++ )
        {
            ouput_ele_size_ *= output_size_[i];
            kernel_ele_size_ *= kernel_size_[i];
            paded_input_ele_size *= paded_input_size_[i];
        }

        int *input_offset_data = input_offset_.mutable_cpu_data();
        for( int i=0; i<ouput_ele_size_; i++ )
        {
            int tmp_index = i;
            int base = 1;
            int index = 0;
            for( int j=axes_num_-1; j>=0; j-- )
            {
                int tmp = tmp_index%output_size_[j];
                tmp = tmp*stride_[j];
                index += tmp*base;
                base *= paded_input_size_[j];
                tmp_index /= output_size_[j];
            }
            input_offset_data[i] = index;
        }

        int *kernel_offset_data = kernel_offset_.mutable_cpu_data();
        for( int i=0; i<kernel_ele_size_; i++ )
        {
            int tmp_index = i;
            int base = 1;
            int index = 0;
            for( int j=axes_num_-1; j>=0; j-- )
            {
                int tmp = tmp_index%kernel_size_[j];
                index += tmp*base;
                base *= paded_input_size_[j];
                tmp_index /= kernel_size_[j];
            }
            kernel_offset_data[i] = index;
        }

        int* paded_index_map_data = paded_index_map_.mutable_cpu_data();
        //init paded_index_map to index of input_size
        for( int i=0; i<paded_input_ele_size; i++ )
        {
            int tmp_index = i;
            bool out_of_range = false;
            int input_offset = 0;
            int input_base = 1;
            
            for( int j=axes_num_-1; j>=0; j-- )
            {
                int tmp = tmp_index%paded_input_size_[j];
                tmp -= pad_[j];
                if( tmp < 0 )
                {
                    out_of_range = true;
                    break;
                }
                input_offset += tmp*input_base;
                input_base *= input_size_[j];
                tmp_index /= paded_input_size_[j];
            }

            if( out_of_range )
            {
                paded_index_map_data[i] = -1;
            }
            else
            {
                paded_index_map_data[i] = input_offset;
            }
        }
       // std::cout<< "paded_index_map:" << std::endl;
       // paded_index_map_.display();
       // std::cout<< "input_offset:" << std::endl;
       // input_offset_.display();
       // std::cout<< "kernel_offset:" << std::endl;
       // kernel_offset_.display();
    }

    template <typename Dtype>
    void ExtendedPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data   = top[0]->mutable_cpu_data();
        const int* paded_index_map_data = paded_index_map_.cpu_data();
        int * max_idx_data = max_idx_.mutable_cpu_data();

        const int* kernel_offset_data = kernel_offset_.cpu_data();
        const int* input_offset_data = input_offset_.cpu_data();
        if( PoolingParameter_PoolMethod_MAX == this->layer_param_.pooling_param().pool()  )
        {
            for( int i=0; i<ouput_ele_size_; i++ )
            {
                int start_offset = input_offset_data[i];
                Dtype max = -0x7fffffff;
                int max_offset  = 0;
                for( int j=0; j<kernel_ele_size_; j++ )
                {
                    int paded_offset = start_offset + kernel_offset_data[j];
                    int offset = paded_index_map_data[paded_offset];
                    if( offset!=-1)
                    {
                        max_offset = (max>bottom_data[offset])?max_offset:offset;
                        max = (max>bottom_data[offset])?max:bottom_data[offset];
                    }
                }
                max_idx_data[i] = max_offset;
                top_data[i] = max;   
            }
        }
        #if 0
        std::cout<<"bottom:"<<std::endl;
        bottom[0]->display();
        std::cout<<"top:"<<std::endl;
        top[0]->display();
        std::cout<<"max_idx:"<<std::endl;   
        max_idx_.display();   
        #endif
    }

    template <typename Dtype>
    void ExtendedPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
    {
        Dtype * bottom_cpu_diff_data = bottom[0]->mutable_cpu_diff();
        caffe_set(bottom[0]->count(), Dtype(0), bottom_cpu_diff_data);
        const Dtype* top_cpu_diff_data = top[0]->cpu_diff();
        const int * diff_index_data = max_idx_.cpu_data();
        for( int i = 0; i<ouput_ele_size_; i++)
        {
            int index = diff_index_data[i];
            bottom_cpu_diff_data[index] += top_cpu_diff_data[i];
        }
        #if 0
        std::cout<<"top diff:"<<std::endl;
        top[0]->display(true);
        #endif
        std::cout<<"bottom cpu diff:"<<std::endl;
        bottom[0]->display(true);
        
    }

    #ifdef CPU_ONLY
    STUB_GPU(ExtendedPoolingLayer);
    #endif
    REGISTER_LAYER_CLASS(ExtendedPooling);
    INSTANTIATE_CLASS(ExtendedPoolingLayer);
}