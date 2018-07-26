#ifndef CAFFE_STRUCTURE_MEASURE_LAYER_HPP_
#define CAFFE_STRUCTURE_MEASURE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "opencv2/core/core.hpp"

namespace caffe {

template <typename Dtype>
class StructureMeasureLayer : public Layer<Dtype> {
 public:
  explicit StructureMeasureLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "StructureMeasure"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- This layer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  
  int test_step_;
  Dtype score_;
  cv::Mat GT_;
  cv::Mat SM_;
};

}  // namespace caffe

#endif  // CAFFE_STRUCTURE_MEASURE_HPP_
