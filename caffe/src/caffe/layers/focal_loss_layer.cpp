#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FocalLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  has_ignore_label_ =
	  this->layer_param_.loss_param().has_ignore_label();
  if ( has_ignore_label_ ) {
	  ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  valid_num_ = 0;
  alpha_ = this->layer_param().focal_loss_param().alpha();
  gamma_ = this->layer_param().focal_loss_param().gamma();
  if ( this->layer_param_.loss_param().has_normalization() ) {
	  normalization_ = this->layer_param_.loss_param().normalization();
  }
  else if ( this->layer_param_.loss_param().has_normalize() ) {
	  normalization_ = this->layer_param_.loss_param().normalize() ?
	  LossParameter_NormalizationMode_VALID : LossParameter_NormalizationMode_BATCH_SIZE;
  }
  else {
	  normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[ 0 ]->shape(0);  // batch size
  inner_num_ = bottom[ 0 ]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  scaler_.ReshapeLike(*bottom[0]);
}


template <typename Dtype>
Dtype FocalLossLayer<Dtype>::get_normalizer(
	LossParameter_NormalizationMode normalization_mode, int valid_count) {
	Dtype normalizer;
	switch ( normalization_mode ) {
		case LossParameter_NormalizationMode_FULL:
			normalizer = Dtype(outer_num_ * inner_num_);
			break;
		case LossParameter_NormalizationMode_VALID:
			if ( valid_count == -1 ) {
				normalizer = Dtype(outer_num_ * inner_num_);
			}
			else {
				normalizer = Dtype(valid_count);
			}
			break;
		case LossParameter_NormalizationMode_BATCH_SIZE:
			normalizer = Dtype(outer_num_);
			break;
		case LossParameter_NormalizationMode_NONE:
			normalizer = Dtype(1);
			break;
		default:
			LOG(FATAL) << "Unknown normalization mode: "
				<< LossParameter_NormalizationMode_Name(normalization_mode);
	}
	// Some users will have no labels for some examples in order to 'turn off' a
	// particular loss in a multi-task setup. The max prevents NaNs in that case.
	return std::max(Dtype(1.0), normalizer);
}


template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(FocalLossLayer, Backward);
#endif

INSTANTIATE_CLASS(FocalLossLayer);
REGISTER_LAYER_CLASS(FocalLoss);

}  // namespace caffe
