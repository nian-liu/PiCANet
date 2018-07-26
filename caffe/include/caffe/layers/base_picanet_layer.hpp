#ifndef CAFFE_BASE_PICANET_LAYER_HPP_
#define CAFFE_BASE_PICANET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Attentive Convolution
 */
template <typename Dtype>
class BasePiCANetLayer : public Layer<Dtype> {
 public:
  /**
   */
  explicit BasePiCANetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:

  // N.B. sharing the col buffer reduces memory but interferes with
  // ND conv and parallelism
  static Blob<Dtype> col_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_PICANET_LAYER_HPP_
