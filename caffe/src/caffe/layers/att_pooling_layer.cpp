#include <vector>

#include "caffe/layers/att_pooling_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AttPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  channels_ = bottom[0]->channels();
  
  AttPoolingParameter att_param = this->layer_param_.att_pooling_param();
  stride_ = att_param.stride();
  pad_ = att_param.pad();
  dilation_ = att_param.dilation();
  attmode_ = att_param.att_mode();
}

template <typename Dtype>
void AttPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  CHECK_EQ(num_, bottom[1]->num()) << "The two bottom blobs must have same batchsizes.";
  height0_ = bottom[0]->height();
  width0_ = bottom[0]->width();
  height1_ = bottom[1]->height();
  width1_ = bottom[1]->width();
  
  AttPoolingParameter att_param = this->layer_param_.att_pooling_param();
  int kernel_extent, height_out, width_out;
  
  if (attmode_ == AttPoolingParameter_AttMode_GLOBAL) {
	  // CHECK_EQ(0, pad_) << "Pad must be 0 for global attention!";
	  CHECK_EQ(1, stride_) << "Stride must be 1 for global attention!";
	  CHECK_EQ(height0_, width0_) << "Only square feature maps are supported for global attention!";
	  if (att_param.has_kernel_size()){
		  kernel_size_ = att_param.kernel_size();
		  kernel_extent = dilation_ * (kernel_size_ - 1) + 1;
		  // CHECK_EQ(kernel_extent, width0_) << "Extended kernel must equal to the feature map size for global attention!";
	  }	else {
		  CHECK_EQ(1, dilation_) << "Dilation should be 1 for global attention without giving kernel_size!";
		  kernel_size_ = width0_;
		  kernel_extent = dilation_ * (kernel_size_ - 1) + 1;
	  }
      height_out = (height0_ + 2 * pad_ - kernel_extent) / stride_ + 1;
      width_out = (width0_ + 2 * pad_ - kernel_extent) / stride_ + 1;
	  CHECK_EQ(height_out, 1) << "Height of convolutional setting does not fit for global attention!";
	  CHECK_EQ(width_out, 1) << "Width of convolutional setting does not fit for global attention!";
  } else if (attmode_ == AttPoolingParameter_AttMode_LOCAL) {
	  CHECK_EQ(1, att_param.has_kernel_size()) << "Kernel size must be given for local attention!";
	  kernel_size_ = att_param.kernel_size();
	  kernel_extent = dilation_ * (kernel_size_ - 1) + 1;
	  height_out = (height0_ + 2 * pad_ - kernel_extent) / stride_ + 1;
	  width_out = (width0_ + 2 * pad_ - kernel_extent) / stride_ + 1;
	  CHECK_EQ(height_out, height1_) << "Height of attention map does not equal to the convolutional setting!";
	  CHECK_EQ(width_out, width1_) << "Width of attention map does not equal to the convolutional setting!";
  } else { LOG(FATAL) << "Unknown attention mode!"; }
  
  CHECK_EQ(kernel_size_ * kernel_size_, bottom[1]->channels()) << "Kernel size is not fit for attention map channels!";
  K_ = kernel_size_ * kernel_size_;
  N_ = height1_ * width1_;
  CHECK_LE(K_, 1024) << "Squared kernel size should not be greater than 1024 for CUDA computation!";
	CHECK_LE(N_, 2147483647) << "Output locations should not be greater than 2147483647 for CUDA computation!";
	//CHECK_LE(channels_, 1024) << "Channel size should not be greater than 1024 for CUDA computation!";
	
  // CHECK_GE(height0_, kernel_extent) << "Feature map height smaller than extended kernel";
  // CHECK_GE(width0_, kernel_extent) << "Feature map width smaller than extended kernel";
  
  // Shape the top.
  top[0]->Reshape(num_, channels_, height1_, width1_);
  
  // The im2col result buffer would only hold one channel of one image at a time to avoid
  // overly large memory usage.
  this->col_buffer_.Reshape( 1, channels_ * kernel_size_ * kernel_size_, height_out, width_out);
}

template <typename Dtype>
void AttPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
		
  NOT_IMPLEMENTED;
  /*
  col_buffer_.Reshape( 1, kernel_size_ * kernel_size_, height_out, width_out);
  
  Dtype* x_data = col_buffer_.mutable_cpu_data();
  const Dtype* fm = bottom[0]->cpu_data();
  const Dtype* attention = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  for (int n = 0; n < num_; n++) {
    for (int m = 0; m < channels_; m++) {
      im2col_cpu(fm + bottom[0]->offset(n, m), 1, height0_,
          width0_, kernel_size_, kernel_size_, pad_, pad_, 
		  stride_, stride_, dilation_, dilation_, x_data);
		  
	  if (attmode_ == AttPoolingParameter_AttMode_GLOBAL) {
		  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_,
              (Dtype)1., x_data, attention+bottom[1]->offset(n),
              (Dtype)0., top_data + top[0]->offset(n, m));
	  } else if (attmode_ == AttPoolingParameter_AttMode_LOCAL) {
		  caffe_mul(K_*N_, x_data, attention+bottom[1]->offset(n),
              intermediate_.mutable_cpu_data());

          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_,
              (Dtype)1., E_.cpu_data(), intermediate_.cpu_data(),
              (Dtype)0., top_data + top[0]->offset(n, m));
	  }
    }
  }*/
}

template <typename Dtype>
void AttPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
  NOT_IMPLEMENTED;
  /*
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* fm = bottom[0]->cpu_data();
  Dtype* fm_diff = bottom[0]->mutable_cpu_diff();
  Dtype* x_data = col_buffer_.mutable_cpu_data();
  Dtype* x_diff = col_buffer_.mutable_cpu_diff();
  const Dtype* attention = bottom[1]->cpu_data();
  Dtype* attention_diff = bottom[1]->mutable_cpu_diff();

  Dtype* intermediate_diff = intermediate_.mutable_cpu_diff();

  caffe_set(bottom[1]->count(), Dtype(0.0), attention_diff);
  for (int n = 0; n < this->num_; n++) {
	for (int m = 0; m < channels_; m++) {
      im2col_cpu(fm + bottom[0]->offset(n, m), 1, height0_,
          width0_, kernel_size_, kernel_size_, pad_, pad_, 
		  stride_, stride_, dilation_, dilation_, x_data);
	  
	  if (attmode_ == AttPoolingParameter_AttMode_GLOBAL) {
		if (propagate_down[1]) {
		  // gradient wrt attention
		  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, 1,
              (Dtype)1., x_data, top_diff + top[0]->offset(n, m),
              (Dtype)1., attention_diff + bottom[1]->offset(n));
		}
		if (propagate_down[0]) {	  
		  // gradient wrt feature map
		  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 1, K_, N_,
              (Dtype)1., top_diff + top[0]->offset(n, m),
			  attention + bottom[1]->offset(n), (Dtype)0., x_diff);
		  
		  // col2im back to the data
          col2im_cpu(x_diff, 1, height0_, width0_, kernel_size_, 
              kernel_size_, pad_, pad_, stride_, stride_, 
              dilation_, dilation_, fm_diff+bottom[0]->offset(n, m));  
		}
	  } else if (attmode_ == AttPoolingParameter_AttMode_LOCAL) {
		if (propagate_down[1]) {
		  // gradient wrt attention
		  for (int k = 0; k < K_; k++) {
            caffe_mul(N_, top_diff+top[0]->offset(n, m),
                x_data+col_buffer_.offset(0, k), intermediate_diff + intermediate_.offset(0, 0, k));
          }
		  caffe_cpu_axpby(K_*N_, Dtype(1.0), intermediate_diff,
              Dtype(1.0), attention_diff + bottom[1]->offset(n));
		}
		if (propagate_down[0]) {	  
		  // gradient wrt feature map
		  for (int k = 0; k < K_; k++) {
            caffe_mul(N_, top_diff+top[0]->offset(n, m),
                attention + bottom[1]->offset(n, k), x_diff + col_buffer_.offset(0, k));
          }
		  // col2im back to the data
          col2im_cpu(x_diff, 1, height0_, width0_, kernel_size_, 
              kernel_size_, pad_, pad_, stride_, stride_, 
              dilation_, dilation_, fm_diff+bottom[0]->offset(n, m));  
		}		
	  }  
	}
  }*/
}

#ifdef CPU_ONLY
STUB_GPU(AttPoolingLayer);
#endif

INSTANTIATE_CLASS(AttPoolingLayer);
REGISTER_LAYER_CLASS(AttPooling);

}  // namespace caffe