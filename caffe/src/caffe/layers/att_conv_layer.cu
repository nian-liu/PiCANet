#include <vector>

#include "caffe/layers/att_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void col_att_gpu_kernel(const int col_size, Dtype* col_data,
    const Dtype* att_data, const int att_size) {
  CUDA_KERNEL_LOOP(index, col_size) {
    //const int att_size = kernel_h*kernel_w*height*width;
    const int att_index = index % att_size;
    //const int channel_index = index % att_size;

    col_data[index] *= att_data[att_index];
  }
}

template <typename Dtype>
void AttConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* att_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < this->num_; ++n) {
    //forward_gpu_gemm
    //this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, att_data+bottom[1]->offset(n),
    //    weight, top_data + n * this->top_dim_);
    const Dtype* input = bottom_data + n * this->bottom_dim_;
    const Dtype* att = att_data+bottom[1]->offset(n);
    Dtype* output = top_data + n * this->top_dim_;
    conv_im2col_gpu(input, this->col_buffer_.mutable_gpu_data());
    // attend to col
    const int col_size = col_size_;
    const int att_size = col_size/conv_in_channels_;
    col_att_gpu_kernel<Dtype> <<<CAFFE_GET_BLOCKS(col_size), CAFFE_CUDA_NUM_THREADS>>>(
          col_size, this->col_buffer_.mutable_gpu_data(), att, att_size);
    CUDA_POST_KERNEL_CHECK;

    const Dtype* col_buff = this->col_buffer_.gpu_data();
    for (int g = 0; g < this->group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
          this->group_, conv_out_spatial_dim_, kernel_dim_,
          (Dtype)1., weights + this->weight_offset_ * g, col_buff + col_offset_ * g,
          (Dtype)0., output + output_offset_ * g);
    }
    //forward_gpu_gemm end
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->gpu_data();
      this->forward_gpu_bias(output, bias);
    }
  }
}

template <typename Dtype>
__global__ void col_att_backward_gpu_kernel(Dtype* col_diff, const Dtype* col_data,
    Dtype* att_diff, const Dtype* att_data, const int rdcStart, const int in_channels) {
  const int att_size = gridDim.x;
  int att_index = blockIdx.x;
  extern __shared__ char tmp_char[];
  Dtype *tmp = (Dtype *)tmp_char;

  int tid = threadIdx.x;
  tmp [tid] = 0;
  for (int chIdx = tid; chIdx < in_channels; chIdx += blockDim.x) {
    tmp [tid] += col_diff[chIdx*att_size+att_index] * col_data[chIdx*att_size+att_index];
    col_diff[chIdx*att_size+att_index] *= att_data[att_index];
  }
  __syncthreads();
  // sum w.r.t channels (using reduction)
  int i=rdcStart;
  while (i !=0 ) {
    if (tid < i && tid + i< blockDim.x) 
      tmp[tid] += tmp[tid + i];
    __syncthreads();
    i /= 2;
  }
  if (tid == 0)
    att_diff[att_index] = tmp[0];
}

template <typename Dtype>
void AttConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
    }
  }
  if (this->param_propagate_down_[0] || propagate_down[0] || propagate_down[1]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* att_data = bottom[1]->gpu_data();
    Dtype* att_diff = bottom[1]->mutable_gpu_diff();
    const int rdcStart = pow(2,(ceil(log(conv_in_channels_*1.0)/log(2.0))-1));
    for (int n = 0; n < this->num_; ++n) {
      // gradient w.r.t. bottom data, if necessary.
      const Dtype* input_data = bottom_data + n * this->bottom_dim_;
      const Dtype* col_buff = input_data;
      if (!this->is_1x1_) {
        conv_im2col_gpu(input_data, this->col_buffer_.mutable_gpu_data());
        col_buff = this->col_buffer_.gpu_data();
      }
      if (propagate_down[0]) {
        //backward_gpu_gemm
        //this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
        //    bottom_diff + n * this->bottom_dim_, bottom_data + n * this->bottom_dim_;, att_diff + bottom[1]->offset(n), att_data+bottom[1]->offset(n));
        const Dtype* output = top_diff + n * this->top_dim_;
        Dtype* input = bottom_diff + n * this->bottom_dim_;
        Dtype* col_diff = this->col_buffer_.mutable_gpu_diff();
        if (this->is_1x1_) {
          col_diff = input;
        }
        for (int g = 0; g < this->group_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
              conv_out_spatial_dim_, conv_out_channels_ / this->group_,
              (Dtype)1., weights + this->weight_offset_ * g, output + output_offset_ * g,
              (Dtype)0., col_diff + col_offset_ * g);
        }

        const int threadsPerBlock = conv_in_channels_< 1024? conv_in_channels_:1024;
        const int blocksPerGrid = col_size_/conv_in_channels_;
        col_att_backward_gpu_kernel<Dtype><<<blocksPerGrid,threadsPerBlock, threadsPerBlock * sizeof(Dtype)>>>(
              col_diff, col_buff, att_diff + bottom[1]->offset(n), att_data+bottom[1]->offset(n), rdcStart, conv_in_channels_);

        CUDA_POST_KERNEL_CHECK;
        if (!this->is_1x1_) {
          conv_col2im_gpu(col_diff, input);
        }
        //backward_gpu_gemm end
      }
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
        const int col_size = col_size_;
        const int att_size = col_size/conv_in_channels_;
        col_att_gpu_kernel<Dtype> <<<CAFFE_GET_BLOCKS(col_size), CAFFE_CUDA_NUM_THREADS>>>(
              col_size, this->col_buffer_.mutable_gpu_data(), att_data+bottom[1]->offset(n), att_size);
        CUDA_POST_KERNEL_CHECK;
        //weight_gpu_gemm
        //this->weight_gpu_gemm(col_buffer_.gpu_data(),
        //    top_diff + n * this->top_dim_, weight_diff);
        const Dtype* col_buff = this->col_buffer_.gpu_data();
        const Dtype* output = top_diff + n * this->top_dim_;
        for (int g = 0; g < this->group_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / this->group_,
              kernel_dim_, conv_out_spatial_dim_,
              (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
              (Dtype)1., weight_diff + this->weight_offset_ * g);
        }
        //weight_gpu_gemm end
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AttConvolutionLayer);

}  // namespace caffe
