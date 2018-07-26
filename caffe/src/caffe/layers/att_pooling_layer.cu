#include <vector>

#include "caffe/layers/att_pooling_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "math.h"

namespace caffe {

template <typename Dtype>
__global__ void colBuf_backward_kernel(const Dtype* topDiff, const Dtype* att, Dtype* colBufDiff, const int channels)
{
	int outDim = gridDim.x;
	int kk = gridDim.y;
	
	int attIdx = blockIdx.y * outDim + blockIdx.x;
	for (int chIdx = threadIdx.x; chIdx < channels; chIdx += blockDim.x) {
		int topDiffIdx = chIdx * outDim + blockIdx.x;
		int colBufDiffIdx = chIdx * (kk * outDim) + attIdx;
		colBufDiff [colBufDiffIdx] = topDiff[topDiffIdx] * att[attIdx];
	}
	__syncthreads();
}


template <typename Dtype>
__global__ void att_backward_kernel(const Dtype* topDiff, const Dtype* col_buffer, Dtype* attDiff, const int rdcStart, const int channels)
{
	int outDim = gridDim.x;
	int kk = gridDim.y;
	extern __shared__ char tmp_char[];
	Dtype *tmp = (Dtype *)tmp_char;
	
	int attDiffIdx = blockIdx.y * outDim + blockIdx.x;
	int tid = threadIdx.x;
	tmp [tid] = 0;
	for (int chIdx = tid; chIdx < channels; chIdx += blockDim.x) {
		int topDiffIdx = chIdx * outDim + blockIdx.x;
		int colBufIdx = chIdx * (kk * outDim) + attDiffIdx;
		tmp [tid] += topDiff[topDiffIdx] * col_buffer[colBufIdx];
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
		attDiff [attDiffIdx] = tmp[0];
}


template <typename Dtype>
__global__ void local_forward_kernel(const Dtype* col_buffer, const Dtype* att, Dtype* top_data, const int rdcStart)
{
	int kk = blockDim.x;
	int outDim = gridDim.x;
	extern __shared__ char tmp_char[];
	Dtype *tmp = (Dtype *)tmp_char;
	
	int kkIdx = threadIdx.x;
	int attIdx = kkIdx * outDim + blockIdx.x;
	int colBufIdx = blockIdx.y * (kk * outDim) + attIdx;
	int topIdx = blockIdx.y * outDim + blockIdx.x;
	tmp [kkIdx] = col_buffer[colBufIdx] * att[attIdx];
	__syncthreads();
	// sum w.r.t kk (using reduction)
	int i=rdcStart;
	while (i !=0 ) {
		if (kkIdx < i && kkIdx + i< kk) 
			tmp[kkIdx] += tmp[kkIdx + i];
		__syncthreads();
		i /= 2;
	}
	if (kkIdx == 0)
		top_data [topIdx] = tmp[0];
}
	
	

/// @brief refer to CPU forward -- the BLAS implementation is the same.
template <typename Dtype>
void AttPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Dtype* x_data = this->col_buffer_.mutable_gpu_data();
  const Dtype* fm = bottom[0]->gpu_data();
  const Dtype* attention = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
	const int rdcStart = pow(2,(ceil(log(K_*1.0)/log(2.0))-1));
	
  for (int n = 0; n < num_; n++) {
	  im2col_gpu(fm + bottom[0]->offset(n), channels_, height0_,
		  width0_, kernel_size_, kernel_size_, pad_, pad_, 
		  stride_, stride_, dilation_, dilation_, x_data);
		  
	  if (attmode_ == AttPoolingParameter_AttMode_GLOBAL) {
		  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, N_, K_,
			  (Dtype)1., x_data, attention+bottom[1]->offset(n),
			  (Dtype)0., top_data + top[0]->offset(n));
	  } else if (attmode_ == AttPoolingParameter_AttMode_LOCAL) {
			
			const int threadsPerBlock = K_;
			dim3 blocksPerGrid(N_, channels_);
			local_forward_kernel<Dtype><<<blocksPerGrid,threadsPerBlock, threadsPerBlock * sizeof(Dtype)>>>
          (x_data, attention+bottom[1]->offset(n), top_data + top[0]->offset(n), rdcStart);
			
			CUDA_POST_KERNEL_CHECK;
	  }
  }
}

/// @brief refer to CPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void AttPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* fm = bottom[0]->gpu_data();
  Dtype* fm_diff = bottom[0]->mutable_gpu_diff();
  Dtype* x_data = this->col_buffer_.mutable_gpu_data();
  Dtype* x_diff = this->col_buffer_.mutable_gpu_diff();
  const Dtype* attention = bottom[1]->gpu_data();
  Dtype* attention_diff = bottom[1]->mutable_gpu_diff();
	const int rdcStart = pow(2,(ceil(log(channels_*1.0)/log(2.0))-1));

  caffe_gpu_set(bottom[1]->count(), Dtype(0.0), attention_diff);
  for (int n = 0; n < this->num_; n++) {
		im2col_gpu(fm + bottom[0]->offset(n), channels_, height0_,
		  width0_, kernel_size_, kernel_size_, pad_, pad_, 
		  stride_, stride_, dilation_, dilation_, x_data);
	  
	  if (attmode_ == AttPoolingParameter_AttMode_GLOBAL) {
			if (propagate_down[1]) {
				// gradient wrt attention
				caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_,channels_,
						(Dtype)1., x_data, top_diff + top[0]->offset(n),
						(Dtype)0., attention_diff + bottom[1]->offset(n));
			}
			if (propagate_down[0]) {	  
				// gradient wrt feature map
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_,
						K_, N_, (Dtype)1., top_diff + top[0]->offset(n),
						attention + bottom[1]->offset(n), (Dtype)0., x_diff);
				
				// col2im back to the data
				col2im_gpu(x_diff, channels_, height0_, width0_, kernel_size_, 
						kernel_size_, pad_, pad_, stride_, stride_, 
						dilation_, dilation_, fm_diff+bottom[0]->offset(n));  
			}
	  } else if (attmode_ == AttPoolingParameter_AttMode_LOCAL) {
			const int threadsPerBlock = channels_< 1024? channels_:1024;
			dim3 blocksPerGrid(N_, K_);
			if (propagate_down[1]) {
				// gradient wrt attention
				att_backward_kernel<Dtype><<<blocksPerGrid,threadsPerBlock, threadsPerBlock * sizeof(Dtype)>>>
						(top_diff + top[0]->offset(n), x_data, attention_diff + bottom[1]->offset(n), rdcStart, channels_);
						
				CUDA_POST_KERNEL_CHECK;
			}
			if (propagate_down[0]) {	  
				// gradient wrt feature map
				colBuf_backward_kernel<Dtype><<<blocksPerGrid,threadsPerBlock>>>
						(top_diff + top[0]->offset(n), attention + bottom[1]->offset(n), x_diff, channels_);
						
				CUDA_POST_KERNEL_CHECK;
				
				// col2im back to the data
				col2im_gpu(x_diff, channels_, height0_, width0_, kernel_size_, 
						kernel_size_, pad_, pad_, stride_, stride_, 
						dilation_, dilation_, fm_diff+bottom[0]->offset(n));  
			}
		}  
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(AttPoolingLayer);

}  // namespace caffe
