#include "caffe/layers/base_picanet_layer.hpp"

namespace caffe {

template <typename Dtype>
Blob<Dtype> BasePiCANetLayer<Dtype>::col_buffer_;

INSTANTIATE_CLASS(BasePiCANetLayer);

}  // namespace caffe
