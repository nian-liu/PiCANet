#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/structure_measure.hpp"
#include "caffe/layers/structure_measure_layer.hpp"


namespace caffe {

template <typename Dtype>
void StructureMeasureLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  test_step_ = 0;
  score_ = 0;
  StructureMeasureParameter structure_measure_param = this->layer_param_.structure_measure_param();
}

template <typename Dtype>
void StructureMeasureLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), 1)
    << "The saliency map should have one channel.";
  CHECK_EQ(bottom[1]->channels(), 1)
    << "The label should have one channel.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data should have the same height as label.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data should have the same width as label.";
  GT_ = cv::Mat::zeros(bottom[0]->height(), bottom[0]->width(), CV_32FC1);
  SM_ = cv::Mat::zeros(bottom[0]->height(), bottom[0]->width(), CV_32FC1);
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void StructureMeasureLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  Dtype* bottom_label = bottom[1]->mutable_cpu_data();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  Dtype score_tmp = 0;

  int data_index;

  // remove old predictions if reset() flag is true
  if (this->phase_ == TEST) {
    if (test_step_ > 
      this->layer_param_.structure_measure_param().test_iter()) {
      score_ = 0;
      test_step_ = 1;
      LOG(INFO) << "Resting score at step "<< test_step_;
    } else {
      test_step_ += 1;
    }
  }

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
  data_index = (i * height + h) * width + w;
  SM_.at<float>(h, w) = bottom_data[data_index];
  GT_.at<float>(h, w) = bottom_label[data_index];
      }
    }
  score_tmp += (Dtype)StructureMeasure(GT_, SM_);
  }
  score_ += score_tmp/num;
  top[0]->mutable_cpu_data()[0] = score_/test_step_;
}

INSTANTIATE_CLASS(StructureMeasureLayer);
REGISTER_LAYER_CLASS(StructureMeasure);

}  // namespace caffe
