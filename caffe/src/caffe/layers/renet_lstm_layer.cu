#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_math.cuh"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/renet_lstm_layer.hpp"

namespace caffe {
// fill in X_H data
template<typename Dtype>
__global__ void Fill_X_H_Data(ReNetLSTMParameter::Direction dir, int dir_num,
    int patch_ny, int patch_nx, int num, int patch_h, int patch_w,
    int bottom_channels, int num_output, int step_id, int step_start,
    Dtype *X_H_data, const Dtype *hidden_data, const Dtype *bottom_data) {
  int num_RNN = dir == ReNetLSTMParameter_Direction_X_DIR ? patch_ny : patch_nx;
  int bottom_height = patch_ny * patch_h;
  int bottom_width = patch_nx * patch_w;
  int X_H_dim = bottom_channels * patch_h * patch_w + num_output;
  bool not_start = step_id != step_start;
  const Dtype *hidden_data_ptr = NULL;
  CUDA_KERNEL_LOOP(index, num_RNN * num)
  {
    int RNN = index / num;
    int n = index % num;
    Dtype *X_H_data_ptr = X_H_data
        + blob_offset(num_RNN, num, X_H_dim, step_id, RNN, n, 0);
    int X_H_data_idx = 0, hidden_data_idx = 0;
    for (int ch = 0; ch < bottom_channels; ++ch) {
      for (int py = 0; py < patch_h; ++py) {
        for (int px = 0; px < patch_w; ++px) {
          int y = dir == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
          int x = dir == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
          int bottom_idx = blob_offset(bottom_channels, bottom_height,
              bottom_width, n, ch, y * patch_h + py, x * patch_w + px);
          X_H_data_ptr[X_H_data_idx++] = bottom_data[bottom_idx];
        }
      }
    }
    if (not_start) {
      hidden_data_ptr = hidden_data
          + blob_offset(num_RNN, num, num_output, 0, RNN, n, 0);
    }
    for (int d = 0; d < num_output; ++d) {
      if (!not_start) {
        X_H_data_ptr[X_H_data_idx++] = 0;
      } else {
        X_H_data_ptr[X_H_data_idx++] = hidden_data_ptr[hidden_data_idx++];
      }
    }
  }
}

template<typename Dtype>
__global__ void compute_cell_state_part_one(ReNetLSTMParameter::Direction dir,
    int dir_num, int patch_ny, int patch_nx, int num, int patch_h, int patch_w,
    int bottom_channels, int num_output, int step_id, int step_start,
    Dtype *gi_data, Dtype *ci_data, Dtype *go_data, Dtype *gf_data,
    Dtype *cstate_data, const Dtype *cstate_data_prev, Dtype *hidden_data,
    Dtype *top_data) {
  int num_RNN = dir == ReNetLSTMParameter_Direction_X_DIR ? patch_ny : patch_nx;
  CUDA_KERNEL_LOOP(index, num_RNN * num * num_output)
  {
    gi_data[index] = sigmoid_dev<Dtype>(gi_data[index]);
    ci_data[index] = tanh_dev<Dtype>(ci_data[index]);
    gf_data[index] = sigmoid_dev<Dtype>(gf_data[index]);
    cstate_data[index] = ci_data[index] * gi_data[index];
    if (step_id != step_start) {
      cstate_data[index] += gf_data[index] * cstate_data_prev[index];
    }
  }
}

template<typename Dtype>
__global__ void compute_cell_state_part_two(ReNetLSTMParameter::Direction dir,
    int dir_num, int patch_ny, int patch_nx, int num, int patch_h, int patch_w,
    int bottom_channels, int num_output, int step_id, int step_start,
    Dtype *gi_data, Dtype *ci_data, Dtype *go_data, Dtype *gf_data,
    Dtype *cstate_data, const Dtype *cstate_data_prev, Dtype *hidden_data,
    Dtype *top_data) {
  int num_RNN = dir == ReNetLSTMParameter_Direction_X_DIR ? patch_ny : patch_nx;
  CUDA_KERNEL_LOOP(index, num_RNN * num * num_output)
  {
    int RNN = index / (num * num_output);
    int rm = index % (num * num_output);
    int n = rm / num_output;
    int d = rm % num_output;
    go_data[index] = sigmoid_dev<Dtype>(go_data[index]);
    hidden_data[index] = go_data[index] * tanh_dev<Dtype>(cstate_data[index]);
    int y = dir == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
    int x = dir == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
    int top_idx = blob_offset(2 * num_output, patch_ny, patch_nx, n,
        dir_num * num_output + d, y, x);
    top_data[top_idx] = hidden_data[index];
  }
}

// compute cell state and output
template<typename Dtype>
void ReNetLSTMLayer<Dtype>::ComputeCellData_gpu(int dir_num, int step_id,
    int step_start, Blob<Dtype>* top) {
  int step = dir_num == 0 ? 1 : -1;
  bool not_start = step_id != step_start;
  Dtype* top_data = top->mutable_gpu_data();
  const Dtype* X_H_data = X_H_data_[dir_num]->gpu_data()
      + X_H_data_[dir_num]->offset(step_id);
  const Dtype* cstate_prev_data = NULL;
  if (not_start) {
    cstate_prev_data = cstate_data_[dir_num]->gpu_data()
        + cstate_data_[dir_num]->offset(step_id - step);
  }
  const Dtype* param_W_i_data =
      this->blobs_[dir_num * num_blobs_per_dir_]->gpu_data();
  const Dtype* param_W_c_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 1]->gpu_data();
  const Dtype* param_W_o_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 2]->gpu_data();
  const Dtype* param_W_f_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 3]->gpu_data();
  const Dtype* param_W_i_c_data = NULL;
  const Dtype* param_W_o_c_data = NULL;
  const Dtype* param_W_f_c_data = NULL;
  if (peephole_) {
    param_W_i_c_data =
        this->blobs_[dir_num * num_blobs_per_dir_ + 4]->gpu_data();
    param_W_o_c_data =
        this->blobs_[dir_num * num_blobs_per_dir_ + 5]->gpu_data();
    param_W_f_c_data =
        this->blobs_[dir_num * num_blobs_per_dir_ + 6]->gpu_data();
  }

  Dtype* gi_data = gi_data_[dir_num]->mutable_gpu_data()
      + gi_data_[dir_num]->offset(step_id);
  Dtype* ci_data = ci_data_[dir_num]->mutable_gpu_data()
      + ci_data_[dir_num]->offset(step_id);
  Dtype* go_data = go_data_[dir_num]->mutable_gpu_data()
      + go_data_[dir_num]->offset(step_id);
  Dtype* gf_data = gf_data_[dir_num]->mutable_gpu_data()
      + gf_data_[dir_num]->offset(step_id);
  Dtype* cstate_data = cstate_data_[dir_num]->mutable_gpu_data()
      + cstate_data_[dir_num]->offset(step_id);
  Dtype* hidden_data = hidden_data_[dir_num]->mutable_gpu_data();

  // compute gi_data
  // W_{i,x}*X_t + H_i * h_{t-1}
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_i_data,
      (Dtype) 0., gi_data);
  if (not_start && peephole_) {
    // W_{i,c} * s_{t-1}
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_,
        num_output_, num_output_, (Dtype) 1., cstate_prev_data,
        param_W_i_c_data, (Dtype) 1., gi_data);
  }
  // add bias b_i
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.gpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 7]->gpu_data(), (Dtype) 1.,
      gi_data);

  // compute ci_data
  // W_{c,x}*X_t + H_c * h_{t-1}
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_c_data,
      (Dtype) 0., ci_data);
  // add bias b_c
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.gpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 8]->gpu_data(), (Dtype) 1.,
      ci_data);

  // compute go_data
  // W_{o,x}*X_t + H_o * h_{t-1}
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_o_data,
      (Dtype) 0., go_data);
  // add bias b_o
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.gpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 9]->gpu_data(), (Dtype) 1.,
      go_data);

  // compute gf_data
  // W_{f,x}*X_t + H_f * h_{t-1}
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_f_data,
      (Dtype) 0., gf_data);
  if (not_start && peephole_) {
    // W_{f,c} * s_{t-1}
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_,
        num_output_, num_output_, (Dtype) 1., cstate_prev_data,
        param_W_f_c_data, (Dtype) 1., gf_data);
  }
  // add bias b_f
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.gpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 10]->gpu_data(), (Dtype) 1.,
      gf_data);

  Dtype* cstate_data_prev = NULL;
  if (step_id != step_start) {
    cstate_data_prev = cstate_data_[dir_num]->mutable_gpu_data()
        + cstate_data_[dir_num]->offset(step_id - step);
  }

  int num_threads = num_RNN_ * num_ * num_output_;
  compute_cell_state_part_one<Dtype> <<<CAFFE_GET_BLOCKS(num_threads),
      CAFFE_CUDA_NUM_THREADS>>>(dir_, dir_num, patch_ny_, patch_nx_, num_,
      patch_h_, patch_w_, channels_, num_output_, step_id, step_start, gi_data,
      ci_data, go_data, gf_data, cstate_data, cstate_data_prev, hidden_data,
      top_data);
  if (peephole_) {
    // compute go_data
    // W_{o,c} * s_t
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_,
        num_output_, num_output_, (Dtype) 1., cstate_data, param_W_o_c_data,
        (Dtype) 1., go_data);
  }

  compute_cell_state_part_two<Dtype> <<<CAFFE_GET_BLOCKS(num_threads),
      CAFFE_CUDA_NUM_THREADS>>>(dir_, dir_num, patch_ny_, patch_nx_, num_,
      patch_h_, patch_w_, channels_, num_output_, step_id, step_start, gi_data,
      ci_data, go_data, gf_data, cstate_data, cstate_data_prev, hidden_data,
      top_data);
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int dir_num = 0; dir_num < 2; ++dir_num) {
    int step_start, step_end, step_min, step_max, step;
    if (dir_ == ReNetLSTMParameter_Direction_X_DIR) {
      step_start = dir_num == 0 ? 0 : patch_nx_ - 1;
      step_end = dir_num == 0 ? patch_nx_ - 1 : 0;
    } else {
      step_start = dir_num == 0 ? 0 : patch_ny_ - 1;
      step_end = dir_num == 0 ? patch_ny_ - 1 : 0;
    }
    step_min = step_start <= step_end ? step_start : step_end;
    step_max = step_start <= step_end ? step_end : step_start;
    step = dir_num == 0 ? 1 : -1;
    // for each RNN time step, process all the RNNs in parallel
    for (int step_id = step_start; step_id >= step_min && step_id <= step_max;
        step_id += step) {
      Fill_X_H_Data<Dtype> <<<CAFFE_GET_BLOCKS(num_RNN_ * num_),
          CAFFE_CUDA_NUM_THREADS>>>(dir_, dir_num, patch_ny_, patch_nx_, num_,
          patch_h_, patch_w_, channels_, num_output_, step_id, step_start,
          X_H_data_[dir_num]->mutable_gpu_data(),
          hidden_data_[dir_num]->gpu_data(), bottom[0]->gpu_data());
      ComputeCellData_gpu(dir_num, step_id, step_start, top[0]);
    }
  }
  Dtype mean_L1_norm = top[0]->asum_data() / top[0]->count();
  DLOG(INFO)<<"Layer "<<this->layer_param_.name()<<
      " mean_L1_norm "<<mean_L1_norm;
}

template<typename Dtype>
__global__ void FillHiddenDiff(ReNetLSTMParameter::Direction dir, int dir_num,
    int patch_ny, int patch_nx, int num, int patch_h, int patch_w,
    int bottom_channels, int num_output, int step_id, int step_end,
    const Dtype *X_H_diff, Dtype *hidden_diff, const Dtype *top_diff) {
  int num_RNN = dir == ReNetLSTMParameter_Direction_X_DIR ? patch_ny : patch_nx;
  int patch_dim = bottom_channels * patch_h * patch_w;
  int X_H_dim = patch_dim + num_output;
  CUDA_KERNEL_LOOP(index, num_RNN * num * num_output)
  {
    int RNN = index / (num * num_output);
    int rm = index % (num * num_output);
    int n = rm / num_output;
    int d = rm % num_output;
    int y = dir == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
    int x = dir == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
    int top_idx = blob_offset(2 * num_output, patch_ny, patch_nx, n,
        dir_num * num_output + d, y, x);
    hidden_diff[index] = top_diff[top_idx];
    if (step_id != step_end) {
      int X_H_idx = blob_offset(num_RNN, num, X_H_dim, 0, RNN, n,
          patch_dim + d);
      hidden_diff[index] += X_H_diff[X_H_idx];
    }
  }
}

template<typename Dtype>
__global__ void ComputeCellDiffPartOne(ReNetLSTMParameter::Direction dir,
    int dir_num, int patch_ny, int patch_nx, int num, int patch_h, int patch_w,
    int num_output, int step_id, int step_start, int step_end,
    const Dtype *gi_data, const Dtype *ci_data, const Dtype *go_data,
    const Dtype *gf_data, const Dtype *cstate_data, Dtype *gi_diff,
    Dtype *ci_diff, Dtype *go_diff, Dtype *gf_diff, Dtype *cstate_diff,
    Dtype *cstate_next_diff, Dtype *hidden_diff) {
  int step = dir_num == 0 ? 1 : -1;
  int num_RNN = dir == ReNetLSTMParameter_Direction_X_DIR ? patch_ny : patch_nx;

  const Dtype *gf_next_data = NULL;
  if (step_id != step_end) {
    gf_next_data = gf_data + step * num_RNN * num * num_output;
  }

  CUDA_KERNEL_LOOP(index, num_RNN * num * num_output)
  {
    Dtype cstate_val = cstate_data[index];
    Dtype go_val = go_data[index];

    go_diff[index] = hidden_diff[index] * tanh_dev<Dtype>(cstate_val)
        * sigmoid_diff_y_dev<Dtype>(go_val);
    cstate_diff[index] = hidden_diff[index] * go_val
        * tanh_diff_x_dev<Dtype>(cstate_val);
    if (step_id != step_end) {
      cstate_diff[index] += cstate_next_diff[index] * gf_next_data[index];
    }
  }
}

template<typename Dtype>
__global__ void ComputeCellDiffPartTwo(ReNetLSTMParameter::Direction dir,
    int dir_num, int patch_ny, int patch_nx, int num, int patch_h, int patch_w,
    int num_output, int step_id, int step_start, int step_end,
    const Dtype *gi_data, const Dtype *ci_data, const Dtype *go_data,
    const Dtype *gf_data, const Dtype *cstate_data, Dtype *gi_diff,
    Dtype *ci_diff, Dtype *go_diff, Dtype *gf_diff, Dtype *cstate_diff,
    Dtype *gi_next_diff, Dtype *gf_next_diff, Dtype *cstate_next_diff,
    Dtype *hidden_diff) {
  int step = dir_num == 0 ? 1 : -1;
  int num_RNN = dir == ReNetLSTMParameter_Direction_X_DIR ? patch_ny : patch_nx;

  const Dtype *cstate_prev_data = NULL;
  if (step_id != step_start) {
    cstate_prev_data = cstate_data - step * num_RNN * num * num_output;
  }

  CUDA_KERNEL_LOOP(index, num_RNN * num * num_output)
  {
    if (step_id != step_start) {
      gf_diff[index] = cstate_diff[index] * cstate_prev_data[index]
          * sigmoid_diff_y_dev<Dtype>(gf_data[index]);
    } else {
      gf_diff[index] = 0;
    }

    Dtype gi_val = gi_data[index];
    Dtype ci_val = ci_data[index];
    gi_diff[index] = cstate_diff[index] * ci_val
        * sigmoid_diff_y_dev<Dtype>(gi_val);
    ci_diff[index] = cstate_diff[index] * gi_val
        * tanh_diff_y_dev<Dtype>(ci_val);

    gi_next_diff[index] = gi_diff[index];
    gf_next_diff[index] = gf_diff[index];
    cstate_next_diff[index] = cstate_diff[index];
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::ComputeCellDiff_gpu(int dir_num, int step_id,
    int step_start, int step_end) {
  int num_threads = num_RNN_ * num_ * num_output_;
  ComputeCellDiffPartOne<Dtype> <<<CAFFE_GET_BLOCKS(num_threads),
      CAFFE_CUDA_NUM_THREADS>>>(dir_, dir_num, patch_ny_, patch_nx_, num_,
      patch_h_, patch_w_, num_output_, step_id, step_start, step_end,
      gi_data_[dir_num]->gpu_data() + gi_data_[dir_num]->offset(step_id),
      ci_data_[dir_num]->gpu_data() + ci_data_[dir_num]->offset(step_id),
      go_data_[dir_num]->gpu_data() + go_data_[dir_num]->offset(step_id),
      gf_data_[dir_num]->gpu_data() + gf_data_[dir_num]->offset(step_id),
      cstate_data_[dir_num]->gpu_data()
          + cstate_data_[dir_num]->offset(step_id),
      gi_diff_[dir_num]->mutable_gpu_data(),
      ci_diff_[dir_num]->mutable_gpu_data(),
      go_diff_[dir_num]->mutable_gpu_data(),
      gf_diff_[dir_num]->mutable_gpu_data(),
      cstate_diff_[dir_num]->mutable_gpu_data(),
      cstate_next_diff_[dir_num]->mutable_gpu_data(),
      hidden_diff_[dir_num]->mutable_gpu_data());

  bool not_end = step_id != step_end;

  const Dtype* param_W_i_c_data = NULL;
  const Dtype* param_W_o_c_data = NULL;
  const Dtype* param_W_f_c_data = NULL;
  if (peephole_) {
    param_W_i_c_data =
          this->blobs_[dir_num * num_blobs_per_dir_ + 4]->gpu_data();
    param_W_o_c_data =
          this->blobs_[dir_num * num_blobs_per_dir_ + 5]->gpu_data();
    param_W_f_c_data =
          this->blobs_[dir_num * num_blobs_per_dir_ + 6]->gpu_data();
  }

  Dtype *go_diff = go_diff_[dir_num]->mutable_gpu_data();
  Dtype *gi_next_diff = gi_next_diff_[dir_num]->mutable_gpu_data();
  Dtype *gf_next_diff = gf_next_diff_[dir_num]->mutable_gpu_data();
  Dtype *cstate_diff = cstate_diff_[dir_num]->mutable_gpu_data();

  if (peephole_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
        num_output_, num_output_, (Dtype) 1., go_diff, param_W_o_c_data,
        (Dtype) 1., cstate_diff);
    if (not_end) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
          num_output_, num_output_, (Dtype) 1., gf_next_diff, param_W_f_c_data,
          (Dtype) 1., cstate_diff);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
          num_output_, num_output_, (Dtype) 1., gi_next_diff, param_W_i_c_data,
          (Dtype) 1., cstate_diff);
    }
  }

  ComputeCellDiffPartTwo<Dtype> <<<CAFFE_GET_BLOCKS(num_threads),
      CAFFE_CUDA_NUM_THREADS>>>(dir_, dir_num, patch_ny_, patch_nx_, num_,
      patch_h_, patch_w_, num_output_, step_id, step_start, step_end,
      gi_data_[dir_num]->gpu_data() + gi_data_[dir_num]->offset(step_id),
      ci_data_[dir_num]->gpu_data() + ci_data_[dir_num]->offset(step_id),
      go_data_[dir_num]->gpu_data() + go_data_[dir_num]->offset(step_id),
      gf_data_[dir_num]->gpu_data() + gf_data_[dir_num]->offset(step_id),
      cstate_data_[dir_num]->gpu_data()
          + cstate_data_[dir_num]->offset(step_id),
      gi_diff_[dir_num]->mutable_gpu_data(),
      ci_diff_[dir_num]->mutable_gpu_data(),
      go_diff_[dir_num]->mutable_gpu_data(),
      gf_diff_[dir_num]->mutable_gpu_data(),
      cstate_diff_[dir_num]->mutable_gpu_data(),
      gi_next_diff_[dir_num]->mutable_gpu_data(),
      gf_next_diff_[dir_num]->mutable_gpu_data(),
      cstate_next_diff_[dir_num]->mutable_gpu_data(),
      hidden_diff_[dir_num]->mutable_gpu_data());
}

// copy gradients w.r.t. X_H_ into bottom diff
template<typename Dtype>
__global__ void UpdateBottomDiff(ReNetLSTMParameter::Direction dir, int dir_num,
    int patch_ny, int patch_nx, int num, int patch_h, int patch_w,
    int bottom_channels, int num_output, int step_id, const Dtype *X_H_diff,
    Dtype *bottom_diff) {
  int num_RNN = dir == ReNetLSTMParameter_Direction_X_DIR ? patch_ny : patch_nx;
  int patch_dim = bottom_channels * patch_h * patch_w;
  int X_H_dim = patch_dim + num_output;
  int bottom_height = patch_ny * patch_h;
  int bottom_width = patch_nx * patch_w;
  CUDA_KERNEL_LOOP(index, num_RNN * num)
  {
    int RNN = index / num;
    int n = index % num;
    const Dtype *X_H_diff_ptr = X_H_diff
        + blob_offset(num_RNN, num, X_H_dim, 0, RNN, n, 0);
    int X_H_idx = 0;
    for (int ch = 0; ch < bottom_channels; ++ch) {
      for (int py = 0; py < patch_h; ++py) {
        for (int px = 0; px < patch_w; ++px) {
          int y = dir == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
          int x = dir == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
          int bottom_idx = blob_offset(bottom_channels, bottom_height,
              bottom_width, n, ch, y * patch_h + py, x * patch_w + px);
          bottom_diff[bottom_idx] += X_H_diff_ptr[X_H_idx++];
        }
      }
    }
  }
}

// compute gradient w.r.t X_H
// compute gradient w.r.t. bottom
template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Compute_X_H_Diff_gpu(int dir_num, int step_id,
    int step_start, Blob<Dtype>* bottom) {
  int X_H_dim = patch_dim_ + num_output_;
  const Dtype* gi_diff = gi_diff_[dir_num]->gpu_data();
  const Dtype* ci_diff = ci_diff_[dir_num]->gpu_data();
  const Dtype* go_diff = go_diff_[dir_num]->gpu_data();
  const Dtype* gf_diff = gf_diff_[dir_num]->gpu_data();

  const Dtype* param_W_i_data =
      this->blobs_[dir_num * num_blobs_per_dir_]->gpu_data();
  const Dtype* param_W_c_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 1]->gpu_data();
  const Dtype* param_W_o_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 2]->gpu_data();
  const Dtype* param_W_f_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 3]->gpu_data();

  // compute gradients w.r.t. X_H_
  Dtype* X_H_diff = X_H_diff_[dir_num]->mutable_gpu_data();

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
      num_output_, (Dtype) 1., gi_diff, param_W_i_data, (Dtype) 0., X_H_diff);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
      num_output_, (Dtype) 1., ci_diff, param_W_c_data, (Dtype) 1., X_H_diff);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
      num_output_, (Dtype) 1., go_diff, param_W_o_data, (Dtype) 1., X_H_diff);
  if (step_id != step_start) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
        num_output_, (Dtype) 1., gf_diff, param_W_f_data, (Dtype) 1., X_H_diff);
  }
  UpdateBottomDiff<Dtype> <<<CAFFE_GET_BLOCKS(num_RNN_ * num_),
      CAFFE_CUDA_NUM_THREADS>>>(dir_, dir_num, patch_ny_, patch_nx_, num_,
      patch_h_, patch_w_, channels_, num_output_, step_id, X_H_diff,
      bottom->mutable_gpu_diff());
}

// compute gradients w.r.t. parameters and biases
template<typename Dtype>
void ReNetLSTMLayer<Dtype>::ComputeParamDiff_gpu(int dir_num, int step_id,
    int step_start) {
  int step = dir_num == 0 ? 1 : -1;
  int X_H_dim = patch_dim_ + num_output_;
  bool not_start = step_id != step_start;
  const Dtype* X_H_data = X_H_data_[dir_num]->gpu_data()
      + X_H_data_[dir_num]->offset(step_id);
  const Dtype *cstate_data = cstate_data_[dir_num]->gpu_data()
      + cstate_data_[dir_num]->offset(step_id);
  const Dtype *cstate_prev_data =
      not_start ?
          cstate_data_[dir_num]->gpu_data()
              + cstate_data_[dir_num]->offset(step_id - step) :
          NULL;

  const Dtype* gi_diff = gi_diff_[dir_num]->gpu_data();
  const Dtype* ci_diff = ci_diff_[dir_num]->gpu_data();
  const Dtype* go_diff = go_diff_[dir_num]->gpu_data();
  const Dtype* gf_diff = gf_diff_[dir_num]->gpu_data();

  Dtype* param_W_i_diff =
      this->blobs_[dir_num * num_blobs_per_dir_]->mutable_gpu_diff();
  Dtype* param_W_c_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 1]->mutable_gpu_diff();
  Dtype* param_W_o_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 2]->mutable_gpu_diff();
  Dtype* param_W_f_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 3]->mutable_gpu_diff();
  Dtype* param_W_i_c_diff = NULL;
  Dtype* param_W_o_c_diff = NULL;
  Dtype* param_W_f_c_diff = NULL;
  if (peephole_) {
    param_W_i_c_diff =
          this->blobs_[dir_num * num_blobs_per_dir_ + 4]->mutable_gpu_diff();
    param_W_o_c_diff =
          this->blobs_[dir_num * num_blobs_per_dir_ + 5]->mutable_gpu_diff();
    param_W_f_c_diff =
          this->blobs_[dir_num * num_blobs_per_dir_ + 6]->mutable_gpu_diff();
  }

  Dtype* bias_b_i_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 7]->mutable_gpu_diff();
  Dtype* bias_b_c_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 8]->mutable_gpu_diff();
  Dtype* bias_b_o_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 9]->mutable_gpu_diff();
  Dtype* bias_b_f_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 10]->mutable_gpu_diff();

  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    // compute gradients w.r.t. parameters
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim, num_,
        (Dtype) 1., gi_diff, X_H_data, (Dtype) 1., param_W_i_diff);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim, num_,
        (Dtype) 1., ci_diff, X_H_data, (Dtype) 1., param_W_c_diff);
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim, num_,
        (Dtype) 1., go_diff, X_H_data, (Dtype) 1., param_W_o_diff);
    if (not_start) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim,
          num_, (Dtype) 1., gf_diff, X_H_data, (Dtype) 1., param_W_f_diff);
      if (peephole_) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
            num_output_, num_, (Dtype) 1., gi_diff, cstate_prev_data,
            (Dtype) 1., param_W_i_c_diff);
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
            num_output_, num_, (Dtype) 1., gf_diff, cstate_prev_data,
            (Dtype) 1., param_W_f_c_diff);
      }
    }
    if (peephole_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, num_output_,
          num_, (Dtype) 1., go_diff, cstate_data, (Dtype) 1., param_W_o_c_diff);
    }

    // compute gradients w.r.t. biases
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., gi_diff,
        bias_multiplier_.gpu_data(), (Dtype) 1., bias_b_i_diff);
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., ci_diff,
        bias_multiplier_.gpu_data(), (Dtype) 1., bias_b_c_diff);
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., go_diff,
        bias_multiplier_.gpu_data(), (Dtype) 1., bias_b_o_diff);
    if (not_start) {
      caffe_gpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., gf_diff,
          bias_multiplier_.gpu_data(), (Dtype) 1., bias_b_f_diff);
    }
    X_H_data += num_ * X_H_dim;
    cstate_data += num_ * num_output_;
    if (not_start) {
      cstate_prev_data += num_ * num_output_;
    }
    gi_diff += num_ * num_output_;
    ci_diff += num_ * num_output_;
    go_diff += num_ * num_output_;
    gf_diff += num_ * num_output_;
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set<Dtype>(bottom[0]->count(), 0, bottom_diff);

  for (int dir_num = 0; dir_num < 2; ++dir_num) {
    for (int i = 0; i < num_blobs_per_dir_; ++i) {
      Dtype* param_diff =
          this->blobs_[dir_num * num_blobs_per_dir_ + i]->mutable_cpu_diff();
      caffe_set<Dtype>(this->blobs_[dir_num * num_blobs_per_dir_ + i]->count(),
          0, param_diff);
    }
    int step_start, step_end, step_min, step_max, step;

    if (dir_ == ReNetLSTMParameter_Direction_X_DIR) {
      step_start = dir_num == 0 ? 0 : patch_nx_ - 1;
      step_end = dir_num == 0 ? patch_nx_ - 1 : 0;
    } else {
      step_start = dir_num == 0 ? 0 : patch_ny_ - 1;
      step_end = dir_num == 0 ? patch_ny_ - 1 : 0;
    }
    step_min = step_start <= step_end ? step_start : step_end;
    step_max = step_start <= step_end ? step_end : step_start;
    step = dir_num == 0 ? 1 : -1;
    for (int step_id = step_end; step_id >= step_min && step_id <= step_max;
        step_id -= step) {
      FillHiddenDiff<Dtype> <<<CAFFE_GET_BLOCKS(num_RNN_ * num_ * num_output_),
          CAFFE_CUDA_NUM_THREADS>>>(dir_, dir_num, patch_ny_, patch_nx_, num_,
          patch_h_, patch_w_, channels_, num_output_, step_id, step_end,
          X_H_diff_[dir_num]->gpu_data(),
          hidden_diff_[dir_num]->mutable_gpu_data(), top[0]->gpu_diff());
      ComputeCellDiff_gpu(dir_num, step_id, step_start, step_end);
      Compute_X_H_Diff_gpu(dir_num, step_id, step_start, bottom[0]);
      ComputeParamDiff_gpu(dir_num, step_id, step_start);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReNetLSTMLayer);
}  // namespace caffe
