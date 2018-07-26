#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/global_att_kldiv_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GlobalAttKLDivLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void GlobalAttKLDivLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  smp_num_ = bottom[0]->count() / bottom[0]->channels();
}

template <typename Dtype>
void GlobalAttKLDivLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < bottom[0]->count(); ++i) {
    loss += target [i] * log (target [i] / (input_data [i] + epsilon_) + epsilon_);
  }
  top[0]->mutable_cpu_data()[0] = loss / smp_num_;
}

template <typename Dtype>
void GlobalAttKLDivLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
	const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	for (int i = 0; i < bottom[0]->count(); ++i) {
	  bottom_diff [i] = Dtype( -1) * target [i] / (input_data [i] + epsilon_);
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(), loss_weight / smp_num_, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(GlobalAttKLDivLossLayer);
#endif

INSTANTIATE_CLASS(GlobalAttKLDivLossLayer);
REGISTER_LAYER_CLASS(GlobalAttKLDivLoss);

}  // namespace caffe
