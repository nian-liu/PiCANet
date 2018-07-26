#include <vector>

#include "caffe/layers/att_deconv_layer.hpp"

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
void AttDeconvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* att_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < this->num_; ++n) {
    //backward_gpu_gemm
    //this->backward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
    //    top_data + n * this->top_dim_);
    const Dtype* output = bottom_data + n * this->bottom_dim_;
    Dtype* input = top_data + n * this->top_dim_;
    Dtype* col_buff = this->col_buffer_.mutable_gpu_data();
    if (this->is_1x1_) {
      col_buff = input;
    }
    for (int g = 0; g < this->group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, this->kernel_dim_,
          this->conv_out_spatial_dim_, this->conv_out_channels_ / this->group_,
          (Dtype)1., weight + this->weight_offset_ * g, output + this->output_offset_ * g,
          (Dtype)0., col_buff + this->col_offset_ * g);
    }
    const Dtype* att = att_data+bottom[1]->offset(n);
    // attend to col
    const int col_size = this->col_size_;
    const int att_size = col_size/this->conv_in_channels_;
    col_att_gpu_kernel<Dtype> <<<CAFFE_GET_BLOCKS(col_size), CAFFE_CUDA_NUM_THREADS>>>(
          col_size, col_buff, att, att_size);
    CUDA_POST_KERNEL_CHECK;
    if (!this->is_1x1_) {
      this->conv_col2im_gpu(col_buff, input);
    }
    //backward_gpu_gemm end
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->gpu_data();
      this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
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
void AttDeconvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
    }
  }
  if (this->param_propagate_down_[0] || propagate_down[0] || propagate_down[1]) {
    const Dtype* att_data = bottom[1]->gpu_data();
    Dtype* att_diff = bottom[1]->mutable_gpu_diff();
    const int rdcStart = pow(2,(ceil(log(this->conv_in_channels_*1.0)/log(2.0))-1));
    for (int n = 0; n < this->num_; ++n) {
      // column diff
      const Dtype* this_top_diff = top_diff + n * this->top_dim_;
      Dtype* col_diff = this->col_buffer_.mutable_gpu_diff();
      this->conv_im2col_gpu(this_top_diff, col_diff);
      // column data
      const Dtype* this_bottom_data = bottom_data + n * this->bottom_dim_;
      Dtype* this_bottom_diff = bottom_diff + n * this->bottom_dim_;
      Dtype* col_buff = this->col_buffer_.mutable_gpu_data();
      for (int g = 0; g < this->group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, this->kernel_dim_,
            this->conv_out_spatial_dim_, this->conv_out_channels_ / this->group_,
            (Dtype)1., weight + this->weight_offset_ * g, this_bottom_data + this->output_offset_ * g,
            (Dtype)0., col_buff + this->col_offset_ * g);
      }
      // gradient w.r.t. att and col_buff
      const int threadsPerBlock = this->conv_in_channels_< 1024? this->conv_in_channels_:1024;
      const int blocksPerGrid = this->col_size_/this->conv_in_channels_;
      col_att_backward_gpu_kernel<Dtype><<<blocksPerGrid,threadsPerBlock, threadsPerBlock * sizeof(Dtype)>>>(
            col_diff, this->col_buffer_.gpu_data(), att_diff + bottom[1]->offset(n), att_data+bottom[1]->offset(n), rdcStart, this->conv_in_channels_);
      CUDA_POST_KERNEL_CHECK;
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[0]) {
        //this->forward_gpu_gemm(top_diff + n * this->top_dim_, weight,
        //    bottom_diff + n * this->bottom_dim_,
        //    this->param_propagate_down_[0]);
        for (int g = 0; g < this->group_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->conv_out_channels_ /
              this->group_, this->conv_out_spatial_dim_, this->kernel_dim_,
              (Dtype)1., weight + this->weight_offset_ * g, col_diff + this->col_offset_ * g,
              (Dtype)0., this_bottom_diff + this->output_offset_ * g);
          }
      }
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
        //this->weight_gpu_gemm(top_diff + n * this->top_dim_,
        //    bottom_data + n * this->bottom_dim_, weight_diff);
        for (int g = 0; g < this->group_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, this->conv_out_channels_ / this->group_,
              this->kernel_dim_, this->conv_out_spatial_dim_,
              (Dtype)1., this_bottom_data + this->output_offset_ * g, col_diff + this->col_offset_ * g,
              (Dtype)1., weight_diff + this->weight_offset_ * g);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AttDeconvolutionLayer);

}  // namespace caffe
