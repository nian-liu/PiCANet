#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/global_att_super_sal_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void GlobalAttSuperSalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the saliency map with the attention size
  // bottom[1] supplies the saliency map with the spatial size
  CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
  CHECK_EQ(bottom[0]->channels(), 1) << "Bottom 0 must be a saliency map.";
  CHECK_EQ(bottom[1]->channels(), 1) << "Bottom 1 must be a saliency map.";
}

template <typename Dtype>
void GlobalAttSuperSalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> new_shape(bottom[1]->shape());
  new_shape[1] = bottom[0]->width() * bottom[0]->height();
  top[0]->Reshape(new_shape);
}

template <typename Dtype>
void GlobalAttSuperSalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void GlobalAttSuperSalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(GlobalAttSuperSalLayer);
#endif

INSTANTIATE_CLASS(GlobalAttSuperSalLayer);
REGISTER_LAYER_CLASS(GlobalAttSuperSal);

}  // namespace caffe
