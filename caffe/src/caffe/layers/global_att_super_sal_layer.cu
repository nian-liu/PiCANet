#include <vector>

#include "caffe/layers/global_att_super_sal_layer.hpp"

namespace caffe {

// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
template <typename Dtype>
__global__ void global_att_super_sal_kernel(const int n, const int att_size, const int spatial_size,
    const Dtype* att_super, const Dtype* sm, Dtype* output) {
  CUDA_KERNEL_LOOP(index, n) {
    int num_idx = index / spatial_size;
    int spatial_idx = index % spatial_size;
    Dtype att_sum = 0;
    for (int i = 0; i < att_size; ++i) {
      att_sum += att_super[num_idx * att_size + i];
    }
    if (sm[index] == 1) {
      for (int j = 0; j < att_size; ++j) {
        output[num_idx * (att_size * spatial_size) + j *  spatial_size + spatial_idx] = 
        (1 - att_super[num_idx * att_size + j]) / (Dtype(att_size) - att_sum + 1e-6);
      }
    }
    else {
      for (int j = 0; j < att_size; ++j) {
        output[num_idx * (att_size * spatial_size) + j *  spatial_size + spatial_idx] = 
        att_super[num_idx * att_size + j] / (att_sum + 1e-6);
      }
    }
  }
}

template <typename Dtype>
void GlobalAttSuperSalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* att_super = bottom[0]->gpu_data();
  const Dtype* sm = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int att_size = bottom[0]->width() * bottom[0]->height();
  const int spatial_size = bottom[1]->width() * bottom[1]->height();
  global_att_super_sal_kernel<<<CAFFE_GET_BLOCKS(bottom[1]->count()), CAFFE_CUDA_NUM_THREADS>>>(
          bottom[1]->count(), att_size, spatial_size,
          att_super, sm, top_data);
}

template <typename Dtype>
void GlobalAttSuperSalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(GlobalAttSuperSalLayer);

}  // namespace caffe
