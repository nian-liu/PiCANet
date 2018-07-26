#ifndef CAFFE_GLOBAL_ATT_KLDIV_LOSS_LAYER_HPP_
#define CAFFE_GLOBAL_ATT_KLDIV_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the KL divergence loss for eye fixation prediction 
 */
template <typename Dtype>
class GlobalAttKLDivLossLayer : public LossLayer<Dtype> {
 public:
  explicit GlobalAttKLDivLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GlobalAttKLDivLoss"; }

 protected:
  /// @copydoc GlobalAttKLDivLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  const Dtype epsilon_ = Dtype(1e-6);
  int smp_num_;
};

}  // namespace caffe

#endif  // CAFFE_GLOBAL_ATT_KLDIV_LOSS_LAYER_HPP_
