#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/global_att_kldiv_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GlobalAttKLDivLossForwardGPU(const int n, const Dtype* FM, const Dtype* SM, const Dtype eps, Dtype* output) {
	CUDA_KERNEL_LOOP(index, n) {
		output[index] = FM [index] * log (FM [index] / (SM [index] + eps) + eps);
	}
}

template <typename Dtype>
__global__ void GlobalAttKLDivLossBackwardGPU(const int n, const Dtype* FM, const Dtype* SM, const Dtype eps, Dtype* output) {
	CUDA_KERNEL_LOOP(index, n) {
	    output [index] = Dtype( -1) * FM [index] / (SM [index] + eps);
	}
}

template <typename Dtype>
void GlobalAttKLDivLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  Dtype* tmp = bottom[1]->mutable_gpu_diff();
  Dtype loss;
  GlobalAttKLDivLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS >>>(
			bottom[0]->count(), target, input_data, epsilon_, tmp);
  caffe_gpu_asum(bottom[0]->count(), tmp, &loss);  //accumulation
  top[0]->mutable_cpu_data()[0] = loss / smp_num_;
}

template <typename Dtype>
void GlobalAttKLDivLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
	const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	
	GlobalAttKLDivLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS >>>(
			bottom[0]->count(), target, input_data, epsilon_, bottom_diff);
	
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(bottom[0]->count(), loss_weight / smp_num_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GlobalAttKLDivLossLayer);

}  // namespace caffe
