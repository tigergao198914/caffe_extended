#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_col);

template <typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
    const int num_spatial_axes, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_output) {
  if (!im2col) {
    int im_size = im_shape[0];
    for (int i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[1 + i];
    }
    caffe_set(im_size, Dtype(0), data_output);
  }
  int kernel_size = 1;
  for (int i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const int channels_col = col_shape[0];
  vector<int> d_offset(num_spatial_axes, 0);
  vector<int> d_iter(num_spatial_axes, 0);
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = c_col;
    for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int index_col = c_col;
      int index_im = c_col / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const int d = d_iter[d_i];
        const int d_im = d * stride[d_i] - pad[d_i] +
            d_offset[d_i] * dilation[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_im;
      }
      if (im2col) {
        if (is_padding) {
          data_output[index_col] = 0;
        } else {
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        const int d_max = col_shape[d_i + 1];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented) {
  }  // for (int c = 0; c < channels_col; ++c) {
}

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
  const bool kIm2Col = true;
  im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                  kernel_shape, pad, stride, dilation, data_col);
}

// Explicit instantiation
template void im2col_nd_cpu<float>(const float* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_cpu<double>(const double* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  const bool kIm2Col = false;
  im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_im);
}

// Explicit instantiation
template void col2im_nd_cpu<float>(const float* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);
template void col2im_nd_cpu<double>(const double* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_im);


void get_offset( const int* shape, const int num_axes, const int idx, vector<int>& offset )
{
    int tmp_idx = idx;
    offset.resize(num_axes);
    for( int dim=num_axes-1; dim>=0; dim-- )
    {
        int tmp_offset = tmp_idx % shape[dim];
        offset[dim] = tmp_offset;
        tmp_idx /= shape[dim];
    }
}

int get_idx( const int *shape, const int num_axes, vector<int>& offset)
{
    int idx = 0;
    int base = 1;
    int max_idx = 1;
    for( int i=0; i<num_axes; i++ )
    {
      max_idx *= shape[i];
    }
    for( int dim=num_axes-1; dim>=0; dim-- )
    {
        if( offset[dim]<0 )
        {
            return -1;
        }
        idx += offset[dim]*base;
        base *= shape[dim];
    }
    if( idx>= max_idx )
    {
      return -1;
    }
    return idx;
}

template <typename Dtype>
Dtype get_data( const Dtype* data, const int *shape, const int num_axes, vector<int>& offset)
{
    int idx = get_idx( shape, num_axes, offset);
    if( idx==-1 )
    {
      return 0;
    } 
    else
    {
      return data[idx];
    }
}

template <typename Dtype>
void data2col_cpu(const Dtype* data, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col) {
    
    int kernel_size = 1;
    for( int i=0; i<num_spatial_axes; i++ )
    {
        kernel_size *= kernel_shape[i];
    }

    int col_num = 1;
    for( int i=0; i<num_spatial_axes; i++ )
    {
       int tmp_dim = (im_shape[i] + 2*pad[i] - kernel_shape[i])/stride[i] + 1;
       col_num *= tmp_dim;
    }

    for( int kel_idx=0; kel_idx<kernel_size; kel_idx++ )
    {
        vector<int> kel_offset;
        get_offset( kernel_shape, num_spatial_axes, kel_idx, kel_offset );

        for( int col_idx=0; col_idx<col_num; col_idx++ )
        {
            vector<int> col_offset;
            vector<int> data_offset;
            get_offset( col_shape+1, num_spatial_axes, col_idx, col_offset );
            for( int dim=0; dim<num_spatial_axes; dim++ )
            {
                data_offset.push_back(col_offset[dim]*stride[dim]);
                data_offset[dim] -= pad[dim];
                data_offset[dim] += kel_offset[dim];
            }
            
            Dtype d = get_data( data, im_shape, num_spatial_axes, data_offset);
            col_offset.insert(col_offset.begin(), kel_idx);
            int idx =  get_idx( col_shape, num_spatial_axes+1, col_offset );
            data_col[idx] = d;
        }
    }
}

template void data2col_cpu<float>(const float* data, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    float* data_col);

template void data2col_cpu<double>(const double* data, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    double* data_col);

template <typename Dtype>
void col2data_cpu( Dtype* data, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const Dtype* data_col)
{
    int kernel_size = 1;
    int image_size = 1;
    for( int i=0; i<num_spatial_axes; i++ )
    {
        kernel_size *= kernel_shape[i];
        image_size *= im_shape[i];
    }

    int col_num = 1;
    for( int i=0; i<num_spatial_axes; i++ )
    {
       int tmp_dim = (im_shape[i] + 2*pad[i] - kernel_shape[i])/stride[i] + 1;
       col_num *= tmp_dim;
    }

    for( int i=0; i<image_size; i++)
    {
      data[i] = 0;
    }

    for( int kel_idx=0; kel_idx<kernel_size; kel_idx++ )
    {
        vector<int> kel_offset;
        get_offset( kernel_shape, num_spatial_axes, kel_idx, kel_offset );

        for( int col_idx=0; col_idx<col_num; col_idx++ )
        {
            vector<int> col_offset;
            vector<int> data_offset;
            get_offset( col_shape+1, num_spatial_axes, col_idx, col_offset );
            for( int dim=0; dim<num_spatial_axes; dim++ )
            {
                data_offset.push_back(col_offset[dim]*stride[dim]);
                data_offset[dim] -= pad[dim];
                data_offset[dim] += kel_offset[dim];
            }

            col_offset.insert(col_offset.begin(), kel_idx);
            Dtype d = get_data( data_col, col_shape, num_spatial_axes+1, col_offset );
            int idx =  get_idx( im_shape, num_spatial_axes, data_offset );
            data[idx] += d;
            #if 0
            Dtype d = get_data( data, im_shape, num_spatial_axes, data_offset);
            col_offset.insert(col_offset.begin(), kel_idx);
            int idx =  get_idx( col_shape, num_spatial_axes+1, col_offset );
            data_col[idx] = d;
            #endif
        }
    }
}
template void col2data_cpu<float>( float* data, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const float* data_col);

template void col2data_cpu<double>( double* data, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const double* data_col);

  template <typename Dtype>
  void data2col_cpu_v2( Dtype* data, const Dtype* data_col, 
            const int input_offset_len, const int* input_offset, 
            const int kernel_offset_len, const int* kernel_offset,
            const int* paded_map )
  {
    Dtype* dst;
    const Dtype* src;
    int paded_start_offset, start_offset;
    dst = data_col;
    src = data;
    for( int i=0; i<input_offset_len; i++ )
    {
      dst++;
      paded_start_offset = input_offset[i];
      start_offset = paded_map[paded_start_offset];
      for( int j = 0; j<kernel_offset_len; j++ )
      {
        if( start_offset == -1 )
        {
          dst[j*input_offset_len] = 0 ;
        }
        else
        {
          dst[j*input_offset_len] = src[start_offset];
        }   
      }
    }
  }

  template <typename Dtype>
  void col2data_cpu_v2( Dtype* data, const Dtype* data_col, 
            const int input_offset_len, const int* input_offset, 
            const int kernel_offset_len, const int* kernel_offset,
            const int* paded_map )
  {
   
  }
}  // namespace caffe
