#ifndef CAFFE_ATT_POOLING_LAYER_HPP_
#define CAFFE_ATT_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/base_picanet_layer.hpp"

namespace caffe {
	
/**
 * @brief Pixel-wisely attending a feature map (with shape num*channel*height*width) 
 *        with attention (with shape num*(kernel*kernel)*height*width), generating 
 *        a attended feature map (with shape num*channel*height*width)
 */
template <typename Dtype>
class AttPoolingLayer : public BasePiCANetLayer<Dtype> {
 public:
  explicit AttPoolingLayer(const LayerParameter& param)
      : BasePiCANetLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AttPooling"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  AttPoolingParameter_AttMode attmode_;
  int num_, channels_;
  int height0_, width0_;
  int height1_, width1_;
  int kernel_size_, stride_, pad_, dilation_;

  int K_;
  int N_;
};

}  // namespace caffe

#endif  //  CAFFE_ATT_POOLING_LAYER_HPP_