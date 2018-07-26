#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_math.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/renet_lstm_layer.hpp"

namespace caffe {

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  peephole_ = this->layer_param_.renet_lstm_param().peephole();
  LOG(INFO)<< "ReNetLSTMLayer " << this->layer_param_.name()
  << " Peephole connections is enabled ? " << peephole_;
  dir_ = this->layer_param_.renet_lstm_param().direction();
  num_output_ = this->layer_param_.renet_lstm_param().num_output();
  patch_h_ = this->layer_param_.renet_lstm_param().patch_height();
  patch_w_ = this->layer_param_.renet_lstm_param().patch_width();

  CHECK_EQ(bottom[0]->num_axes(), 4);

  channels_ = bottom[0]->shape(1);
  patch_dim_ = channels_ * patch_h_ * patch_w_;
  // two opposite scanning directions
  X_H_data_.resize(2);
  X_H_diff_.resize(2);
  gi_data_.resize(2);
  gi_diff_.resize(2);
  gi_next_diff_.resize(2);
  ci_data_.resize(2);
  ci_diff_.resize(2);
  go_data_.resize(2);
  go_diff_.resize(2);
  gf_data_.resize(2);
  gf_diff_.resize(2);
  gf_next_diff_.resize(2);
  cstate_data_.resize(2);
  cstate_diff_.resize(2);
  cstate_next_diff_.resize(2);
  hidden_data_.resize(2);
  hidden_diff_.resize(2);
  for (int dir_num = 0; dir_num < 2; ++dir_num) {
    X_H_data_[dir_num].reset(new Blob<Dtype>());
    X_H_diff_[dir_num].reset(new Blob<Dtype>());
    gi_data_[dir_num].reset(new Blob<Dtype>());
    gi_diff_[dir_num].reset(new Blob<Dtype>());
    gi_next_diff_[dir_num].reset(new Blob<Dtype>());
    ci_data_[dir_num].reset(new Blob<Dtype>());
    ci_diff_[dir_num].reset(new Blob<Dtype>());
    go_data_[dir_num].reset(new Blob<Dtype>());
    go_diff_[dir_num].reset(new Blob<Dtype>());
    gf_data_[dir_num].reset(new Blob<Dtype>());
    gf_diff_[dir_num].reset(new Blob<Dtype>());
    gf_next_diff_[dir_num].reset(new Blob<Dtype>());
    cstate_data_[dir_num].reset(new Blob<Dtype>());
    cstate_diff_[dir_num].reset(new Blob<Dtype>());
    cstate_next_diff_[dir_num].reset(new Blob<Dtype>());
    hidden_data_[dir_num].reset(new Blob<Dtype>());
    hidden_diff_[dir_num].reset(new Blob<Dtype>());
  }

  // 4 parameter matrices W_i, W_c, W_o, W_f
  // 3 parameter matrices W_{i,c}, W_{o,c}, W_{f,c}
  // 4 bias vectors b_i, b_c, b_o, b_f
  num_blobs_per_dir_ = 11;
  this->blobs_.resize(2 * num_blobs_per_dir_);
  // 4 parameter matrices
  // W_i = [W_{i,x}, H_i]
  // W_c = [W_{c,x}, H_c]
  // W_o = [W_{o,x}, H_o]
  // W_f = [W_{f,x}, H_f]
  vector<int> W_X_H_shape(2);
  W_X_H_shape[0] = num_output_;
  W_X_H_shape[1] = patch_dim_ + num_output_;
  // 3 parameter matrices
  // W_{i,c}, W_{o,c}, W_{f,c}
  vector<int> W_C_shape(2);
  W_C_shape[0] = num_output_;
  W_C_shape[1] = num_output_;
  // four bias vectors b_i, b_c, b_o, b_f
  vector<int> B_shape(1, num_output_);

  int parameter_memory = W_X_H_shape[0] * W_X_H_shape[1] * 4 + num_output_;
  parameter_memory *= 2;
  LOG(INFO)<<"Layer "<<this->layer_param_.name()
  <<" parameter memory footprint "<<parameter_memory * sizeof(Dtype);

  shared_ptr<Filler<Dtype> > general_weight_filler(
      GetFiller<Dtype>(
          this->layer_param_.renet_lstm_param().general_weight_filler()));
  shared_ptr<Filler<Dtype> > general_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.renet_lstm_param().general_bias_filler()));
  shared_ptr<Filler<Dtype> > forget_gate_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.renet_lstm_param().forget_gate_bias_filler()));
  shared_ptr<Filler<Dtype> > input_gate_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.renet_lstm_param().input_gate_bias_filler()));
  shared_ptr<Filler<Dtype> > output_gate_bias_filler(
      GetFiller<Dtype>(
          this->layer_param_.renet_lstm_param().output_gate_bias_filler()));

  for (int dir_num = 0; dir_num < 2; ++dir_num) {
    // 4 parameter matrices, W_i, W_c, W_o, W_f
    for (int p = 0; p < 4; ++p) {
      this->blobs_[dir_num * num_blobs_per_dir_ + p].reset(
          new Blob<Dtype>(W_X_H_shape));
    }
    // 3 parameter matrices, W_{i,c}, W_{o,c}, W_{f,c}
    for (int p = 4; p < 7; ++p) {
      this->blobs_[dir_num * num_blobs_per_dir_ + p].reset(
          new Blob<Dtype>(W_C_shape));
    }
    // 4 bias vectors, b_i, b_c, b_o, b_f
    for (int p = 7; p < 11; ++p) {
      this->blobs_[dir_num * num_blobs_per_dir_ + p].reset(
          new Blob<Dtype>(B_shape));
    }
    // 7 parameter matrices, W_i, W_c, W_o, W_f, W_{i,c}, W_{o,c}, W_{f,c}
    if (peephole_) {
      for (int p = 0; p < 7; ++p) {
        general_weight_filler->Fill(
            this->blobs_[dir_num * num_blobs_per_dir_ + p].get());
      }
    } else {
      for (int p = 0; p < 4; ++p) {
        general_weight_filler->Fill(
            this->blobs_[dir_num * num_blobs_per_dir_ + p].get());
      }
    }
    // input gate bias vector, b_i
    input_gate_bias_filler->Fill(
        this->blobs_[dir_num * num_blobs_per_dir_ + 7].get());
    // cell input bias vector, b_c
    general_bias_filler->Fill(
        this->blobs_[dir_num * num_blobs_per_dir_ + 8].get());
    // output gate bias vector, b_0
    output_gate_bias_filler->Fill(
        this->blobs_[dir_num * num_blobs_per_dir_ + 9].get());
    // forget gate bias vector, b_f
    forget_gate_bias_filler->Fill(
        this->blobs_[dir_num * num_blobs_per_dir_ + 10].get());
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4);
  CHECK_EQ(bottom[0]->shape(1), channels_);
  CHECK_EQ(bottom[0]->shape(2) % patch_h_, 0)<<" bottom height "<<bottom[0]->shape(2);
  CHECK_EQ(bottom[0]->shape(3) % patch_w_, 0)<<" bottom width "<<bottom[0]->shape(3);

  num_ = bottom[0]->shape(0);
  patch_ny_ = bottom[0]->shape(2) / patch_h_;
  patch_nx_ = bottom[0]->shape(3) / patch_w_;

  num_RNN_ = dir_ == ReNetLSTMParameter_Direction_X_DIR ? patch_ny_ : patch_nx_;
  num_steps_ =
      dir_ == ReNetLSTMParameter_Direction_X_DIR ? patch_nx_ : patch_ny_;

  vector<int> X_H_shape_4D(4);
  X_H_shape_4D[0] = num_steps_;
  X_H_shape_4D[1] = num_RNN_;
  X_H_shape_4D[2] = num_;
  X_H_shape_4D[3] = patch_dim_ + num_output_;

  vector<int> X_H_shape_3D(3);
  X_H_shape_3D[0] = num_RNN_;
  X_H_shape_3D[1] = num_;
  X_H_shape_3D[2] = patch_dim_ + num_output_;

  vector<int> cell_shape_4D(4);
  cell_shape_4D[0] = num_steps_;
  cell_shape_4D[1] = num_RNN_;
  cell_shape_4D[2] = num_;
  cell_shape_4D[3] = num_output_;

  vector<int> cell_shape_3D(3);
  cell_shape_3D[0] = num_RNN_;
  cell_shape_3D[1] = num_;
  cell_shape_3D[2] = num_output_;

  for (int dir_num = 0; dir_num < 2; ++dir_num) {
    X_H_data_[dir_num]->Reshape(X_H_shape_4D);
    X_H_diff_[dir_num]->Reshape(X_H_shape_3D);
    X_H_diff_[dir_num]->Reshape(X_H_shape_3D);
    gi_data_[dir_num]->Reshape(cell_shape_4D);
    gi_diff_[dir_num]->Reshape(cell_shape_3D);
    gi_next_diff_[dir_num]->Reshape(cell_shape_3D);
    ci_data_[dir_num]->Reshape(cell_shape_4D);
    ci_diff_[dir_num]->Reshape(cell_shape_3D);
    go_data_[dir_num]->Reshape(cell_shape_4D);
    go_diff_[dir_num]->Reshape(cell_shape_3D);
    gf_data_[dir_num]->Reshape(cell_shape_4D);
    gf_diff_[dir_num]->Reshape(cell_shape_3D);
    gf_next_diff_[dir_num]->Reshape(cell_shape_3D);
    cstate_data_[dir_num]->Reshape(cell_shape_4D);
    cstate_diff_[dir_num]->Reshape(cell_shape_3D);
    cstate_next_diff_[dir_num]->Reshape(cell_shape_3D);
    hidden_data_[dir_num]->Reshape(cell_shape_3D);
    hidden_diff_[dir_num]->Reshape(cell_shape_3D);
  }

  int internal_var_memory = num_steps_ * num_RNN_ * num_
      * (patch_dim_ + num_output_)
      + num_RNN_ * num_ * (patch_dim_ + num_output_)
      + num_steps_ * num_RNN_ * num_ * num_output_ * 5
      + num_RNN_ * num_ * num_output_ * 10;
  internal_var_memory *= 2;
  DLOG(INFO)<<"Layer: "<<this->layer_param_.name()
  <<" Internal variable memory footprint:"<<internal_var_memory*sizeof(Dtype);

  vector<int> top_shape(4);
  top_shape[0] = num_;
  top_shape[1] = 2 * num_output_;
  top_shape[2] = patch_ny_;
  top_shape[3] = patch_nx_;
  top[0]->Reshape(top_shape);

  vector<int> bias_shape(1, num_RNN_ * num_);
  bias_multiplier_.Reshape(bias_shape);
  caffe_set<Dtype>(num_RNN_ * num_, Dtype(1),
      bias_multiplier_.mutable_cpu_data());
}

// fill in X_H data
template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Fill_X_H_Data_cpu(int dir_num, int step_id,
    int step_start, Blob<Dtype>* bottom) {
  bool not_start = step_id != step_start;
  const Dtype *bottom_data = bottom->cpu_data();
  Dtype *X_H_data = X_H_data_[dir_num]->mutable_cpu_data()
      + X_H_data_[dir_num]->offset(step_id);

  const Dtype *hidden_data = hidden_data_[dir_num]->cpu_data();

  int X_H_data_index = 0;
  int hidden_index = 0;
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      for (int ch = 0; ch < channels_; ++ch) {
        for (int py = 0; py < patch_h_; ++py) {
          for (int px = 0; px < patch_w_; ++px) {
            int y = dir_ == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
            int x = dir_ == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
            int bottom_index = bottom->offset(n, ch, y * patch_h_ + py,
                x * patch_w_ + px);
            X_H_data[X_H_data_index++] = bottom_data[bottom_index];
          }
        }
      }
      // fill X_H with previous hidden outputs
      for (int d = 0; d < num_output_; ++d) {
        if (!not_start) {
          X_H_data[X_H_data_index++] = 0;
        } else {
          X_H_data[X_H_data_index++] = hidden_data[hidden_index++];
        }
      }
    }
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::ComputeCellData_cpu(int dir_num, int step_id,
    int step_start, Blob<Dtype>* top) {
  int step = dir_num == 0 ? 1 : -1;
  bool not_start = step_id != step_start;
  Dtype* top_data = top->mutable_cpu_data();
  const Dtype* X_H_data = X_H_data_[dir_num]->cpu_data()
      + X_H_data_[dir_num]->offset(step_id);
  const Dtype* cstate_prev_data = NULL;
  if (not_start) {
    cstate_prev_data = cstate_data_[dir_num]->cpu_data()
        + cstate_data_[dir_num]->offset(step_id - step);
  }
  const Dtype* param_W_i_data =
      this->blobs_[dir_num * num_blobs_per_dir_]->cpu_data();
  const Dtype* param_W_c_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 1]->cpu_data();
  const Dtype* param_W_o_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 2]->cpu_data();
  const Dtype* param_W_f_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 3]->cpu_data();
  const Dtype* param_W_i_c_data = NULL;
  const Dtype* param_W_o_c_data = NULL;
  const Dtype* param_W_f_c_data = NULL;
  if (peephole_) {
    param_W_i_c_data =
        this->blobs_[dir_num * num_blobs_per_dir_ + 4]->cpu_data();
    param_W_o_c_data =
        this->blobs_[dir_num * num_blobs_per_dir_ + 5]->cpu_data();
    param_W_f_c_data =
        this->blobs_[dir_num * num_blobs_per_dir_ + 6]->cpu_data();
  }

  Dtype* gi_data = gi_data_[dir_num]->mutable_cpu_data()
      + gi_data_[dir_num]->offset(step_id);
  Dtype* ci_data = ci_data_[dir_num]->mutable_cpu_data()
      + ci_data_[dir_num]->offset(step_id);
  Dtype* go_data = go_data_[dir_num]->mutable_cpu_data()
      + go_data_[dir_num]->offset(step_id);
  Dtype* gf_data = gf_data_[dir_num]->mutable_cpu_data()
      + gf_data_[dir_num]->offset(step_id);
  Dtype* cstate_data = cstate_data_[dir_num]->mutable_cpu_data()
      + cstate_data_[dir_num]->offset(step_id);
  Dtype* hidden_data = hidden_data_[dir_num]->mutable_cpu_data();

  // compute gi_data
  // W_{i,x}*X_t + H_i * h_{t-1}
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_i_data,
      (Dtype) 0., gi_data);
  if (not_start && peephole_) {
    // W_{i,c} * s_{t-1}
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_,
        num_output_, num_output_, (Dtype) 1., cstate_prev_data,
        param_W_i_c_data, (Dtype) 1., gi_data);
  }
  // add bias b_i
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.cpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 7]->cpu_data(), (Dtype) 1.,
      gi_data);

  // compute ci_data
  // W_{c,x}*X_t + H_c * h_{t-1}
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_c_data,
      (Dtype) 0., ci_data);
  // add bias b_c
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.cpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 8]->cpu_data(), (Dtype) 1.,
      ci_data);

  // compute go_data
  // W_{o,x}*X_t + H_o * h_{t-1}
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_o_data,
      (Dtype) 0., go_data);
  // add bias b_o
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.cpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 9]->cpu_data(), (Dtype) 1.,
      go_data);

  // compute gf_data
  // W_{f,x}*X_t + H_f * h_{t-1}
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_, num_output_,
      patch_dim_ + num_output_, (Dtype) 1., X_H_data, param_W_f_data,
      (Dtype) 0., gf_data);
  if (not_start && peephole_) {
    // W_{f,c} * s_{t-1}
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_,
        num_output_, num_output_, (Dtype) 1., cstate_prev_data,
        param_W_f_c_data, (Dtype) 1., gf_data);
  }
  // add bias b_f
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
      num_output_, 1, (Dtype) 1., bias_multiplier_.cpu_data(),
      this->blobs_[dir_num * num_blobs_per_dir_ + 10]->cpu_data(), (Dtype) 1.,
      gf_data);

  int data_index = 0;
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      for (int d = 0; d < num_output_; ++d) {
        gi_data[data_index] = sigmoid<Dtype>(gi_data[data_index]);
        ci_data[data_index] = tanh<Dtype>(ci_data[data_index]);
        gf_data[data_index] = sigmoid<Dtype>(gf_data[data_index]);
        cstate_data[data_index] = ci_data[data_index] * gi_data[data_index];
        if (not_start) {
          cstate_data[data_index] += gf_data[data_index]
              * cstate_prev_data[data_index];
        }
        data_index++;
      }  // for (int d = 0; d < num_output_; ++d)
    }  // for (int n = 0; n < num_; ++n)
  }
  if (peephole_) {
    // compute go_data
    // W_{o,c} * s_t
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_RNN_ * num_,
        num_output_, num_output_, (Dtype) 1., cstate_data, param_W_o_c_data,
        (Dtype) 1., go_data);
  }

  data_index = 0;
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      for (int d = 0; d < num_output_; ++d) {
        go_data[data_index] = sigmoid<Dtype>(go_data[data_index]);
        hidden_data[data_index] = go_data[data_index]
            * tanh<Dtype>(cstate_data[data_index]);
        // copy hidden output into top data
        int y = dir_ == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
        int x = dir_ == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
        top_data[top->offset(n, dir_num * num_output_ + d, y, x)] =
            hidden_data[data_index];
        data_index++;
      }  // for (int d = 0; d < num_output_; ++d)
    }  // for (int n = 0; n < num_; ++n)
  }
}

/*
 * Forward pass
 * input gate:  i_t = sigmoid(W_{i,x}*X_t + H_i * h_{t-1} +
 *                    W_{i,c} * s_{t-1} + b_i)
 * cell input:  c_t = tanh(W_{c,x}*X_t + H_c * h_{t-1} + b_c)
 * forget gate: f_t = sigmoid(W_{f,x}*X_t + H_f * h_{t-1}
 *                    W_{f,c} * s_{t-1} + b_f)
 * cell state:  s_t = c_t .* i_t + f_t .* s_{t-1}
 * output gate: o_t = sigmoid(W_{o,x}*X_t + H_o * h_{t-1} +
 *                    W_{o,c} * s_t + b_o)
 * cell output: h_t = o_t .* tanh(s_t)
 * */
template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
    for (int step_id = step_start; step_id >= step_min && step_id <= step_max;
        step_id += step) {
      Fill_X_H_Data_cpu(dir_num, step_id, step_start, bottom[0]);
      ComputeCellData_cpu(dir_num, step_id, step_start, top[0]);
    }
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::FillHiddenDiff_cpu(int dir_num, int step_id,
    int step_end, Blob<Dtype>* top) {
  const Dtype* top_diff = top->cpu_diff();
  Dtype* hidden_diff = hidden_diff_[dir_num]->mutable_cpu_data();
  int data_index = 0;
  bool not_end = step_id != step_end;
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      const Dtype* X_H_diff = X_H_diff_[dir_num]->cpu_data()
          + X_H_diff_[dir_num]->offset(RNN, n);
      for (int d = 0; d < num_output_; ++d) {
        // copy top diff into hidden_diff
        int y = dir_ == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
        int x = dir_ == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
        hidden_diff[data_index] = top_diff[top->offset(n,
            dir_num * num_output_ + d, y, x)];
        if (not_end) {
          hidden_diff[data_index] += X_H_diff[patch_dim_ + d];
        }
        data_index++;
      }
    }
  }
}

template<typename Dtype>
void ReNetLSTMLayer<Dtype>::ComputeCellDiff_cpu(int dir_num, int step_id,
    int step_start, int step_end) {
  const int step = dir_num == 0 ? 1 : -1;

  const Dtype *param_W_i_c_data = NULL;
  const Dtype *param_W_o_c_data = NULL;
  const Dtype *param_W_f_c_data = NULL;
  if (peephole_) {
    param_W_i_c_data =
        this->blobs_[dir_num * num_blobs_per_dir_ + 4]->cpu_data();
    param_W_o_c_data =
        this->blobs_[dir_num * num_blobs_per_dir_ + 5]->cpu_data();
    param_W_f_c_data =
        this->blobs_[dir_num * num_blobs_per_dir_ + 6]->cpu_data();
  }

  const int cell_data_offset = gi_data_[dir_num]->offset(step_id);

  const Dtype* hidden_diff = hidden_diff_[dir_num]->cpu_data();

  const Dtype* gi_data = gi_data_[dir_num]->cpu_data() + cell_data_offset;
  Dtype *gi_diff = gi_diff_[dir_num]->mutable_cpu_data();
  Dtype *gi_next_diff = gi_next_diff_[dir_num]->mutable_cpu_data();

  const Dtype* ci_data = ci_data_[dir_num]->cpu_data() + cell_data_offset;
  Dtype* ci_diff = ci_diff_[dir_num]->mutable_cpu_data();

  const Dtype* go_data = go_data_[dir_num]->cpu_data() + cell_data_offset;
  Dtype* go_diff = go_diff_[dir_num]->mutable_cpu_data();

  const Dtype* gf_data = gf_data_[dir_num]->cpu_data() + cell_data_offset;
  Dtype *gf_diff = gf_diff_[dir_num]->mutable_cpu_data();
  Dtype *gf_next_diff = gf_next_diff_[dir_num]->mutable_cpu_data();

  const Dtype* cstate_data = cstate_data_[dir_num]->cpu_data()
      + cell_data_offset;
  Dtype* cstate_diff = cstate_diff_[dir_num]->mutable_cpu_data();
  Dtype* cstate_next_diff = cstate_next_diff_[dir_num]->mutable_cpu_data();

  bool not_start = step_id != step_start;
  bool not_end = step_id != step_end;

  const Dtype* gf_next_data = NULL;
  if (not_end) {
    gf_next_data = gf_data_[dir_num]->cpu_data()
        + gf_data_[dir_num]->offset(step_id + step);
  }

  const Dtype* cstate_prev_data = NULL;
  if (not_start) {
    cstate_prev_data = cstate_data_[dir_num]->cpu_data()
        + cstate_data_[dir_num]->offset(step_id - step);
  }

  int index = 0;
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      for (int d = 0; d < num_output_; ++d) {
        Dtype cstate_val = cstate_data[index];
        Dtype go_val = go_data[index];

        go_diff[index] = hidden_diff[index] * tanh<Dtype>(cstate_val)
            * sigmoid_diff_y<Dtype>(go_val);
        cstate_diff[index] = hidden_diff[index] * go_val
            * tanh_diff_x<Dtype>(cstate_val);
        if (not_end) {
          cstate_diff[index] += cstate_next_diff[index] * gf_next_data[index];
        }
        index++;
      }
    }
  }
  if (peephole_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
        num_output_, num_output_, (Dtype) 1., go_diff, param_W_o_c_data,
        (Dtype) 1., cstate_diff);
    if (not_end) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
          num_output_, num_output_, (Dtype) 1., gf_next_diff, param_W_f_c_data,
          (Dtype) 1., cstate_diff);
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_,
          num_output_, num_output_, (Dtype) 1., gi_next_diff, param_W_i_c_data,
          (Dtype) 1., cstate_diff);
    }
  }

  index = 0;
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      for (int d = 0; d < num_output_; ++d) {
        if (not_start) {
          gf_diff[index] = cstate_diff[index] * cstate_prev_data[index]
              * sigmoid_diff_y<Dtype>(gf_data[index]);
        } else {
          gf_diff[index] = 0;
        }
        Dtype gi_val = gi_data[index];
        Dtype ci_val = ci_data[index];
        gi_diff[index] = cstate_diff[index] * ci_val
            * sigmoid_diff_y<Dtype>(gi_val);
        ci_diff[index] = cstate_diff[index] * gi_val
            * tanh_diff_y<Dtype>(ci_val);
        gi_next_diff[index] = gi_diff[index];
        gf_next_diff[index] = gf_diff[index];
        cstate_next_diff[index] = cstate_diff[index];
        index++;
      }
    }
  }
}

// compute gradient w.r.t X_H
// compute gradient w.r.t. bottom
template<typename Dtype>
void ReNetLSTMLayer<Dtype>::Compute_X_H_Diff_cpu(int dir_num, int step_id,
    int step_start, Blob<Dtype>* bottom) {
  int X_H_dim = patch_dim_ + num_output_;
  const Dtype* gi_diff = gi_diff_[dir_num]->cpu_data();
  const Dtype* ci_diff = ci_diff_[dir_num]->cpu_data();
  const Dtype* go_diff = go_diff_[dir_num]->cpu_data();
  const Dtype* gf_diff = gf_diff_[dir_num]->cpu_data();

  const Dtype* param_W_i_data =
      this->blobs_[dir_num * num_blobs_per_dir_]->cpu_data();
  const Dtype* param_W_c_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 1]->cpu_data();
  const Dtype* param_W_o_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 2]->cpu_data();
  const Dtype* param_W_f_data =
      this->blobs_[dir_num * num_blobs_per_dir_ + 3]->cpu_data();

  // compute gradients w.r.t. X_H_
  Dtype* X_H_diff = X_H_diff_[dir_num]->mutable_cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
      num_output_, (Dtype) 1., gi_diff, param_W_i_data, (Dtype) 0., X_H_diff);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
      num_output_, (Dtype) 1., ci_diff, param_W_c_data, (Dtype) 1., X_H_diff);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
      num_output_, (Dtype) 1., go_diff, param_W_o_data, (Dtype) 1., X_H_diff);
  if (step_id != step_start) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_RNN_ * num_, X_H_dim,
        num_output_, (Dtype) 1., gf_diff, param_W_f_data, (Dtype) 1., X_H_diff);
  }

  // copy gradients w.r.t. X_H_ into bottom diff
  Dtype* bottom_diff = bottom->mutable_cpu_diff();
  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    for (int n = 0; n < num_; ++n) {
      X_H_diff = X_H_diff_[dir_num]->mutable_cpu_data()
          + X_H_diff_[dir_num]->offset(RNN, n);
      int data_index = 0;
      for (int ch = 0; ch < channels_; ++ch) {
        for (int py = 0; py < patch_h_; ++py) {
          for (int px = 0; px < patch_w_; ++px) {
            int y = dir_ == ReNetLSTMParameter_Direction_X_DIR ? RNN : step_id;
            int x = dir_ == ReNetLSTMParameter_Direction_X_DIR ? step_id : RNN;
            bottom_diff[bottom->offset(n, ch, y * patch_h_ + py,
                x * patch_w_ + px)] += X_H_diff[data_index++];
          }
        }
      }
    }
  }
}

// compute gradients w.r.t. parameters and biases
template<typename Dtype>
void ReNetLSTMLayer<Dtype>::ComputeParamDiff_cpu(int dir_num, int step_id,
    int step_start) {
  int step = dir_num == 0 ? 1 : -1;
  int X_H_dim = patch_dim_ + num_output_;
  bool not_start = step_id != step_start;

  const Dtype *X_H_data = X_H_data_[dir_num]->cpu_data()
      + X_H_data_[dir_num]->offset(step_id);
  const Dtype *cstate_data = cstate_data_[dir_num]->cpu_data()
      + cstate_data_[dir_num]->offset(step_id);
  const Dtype *cstate_prev_data =
      not_start ?
          cstate_data_[dir_num]->cpu_data()
              + cstate_data_[dir_num]->offset(step_id - step) :
          NULL;

  const Dtype* gi_diff = gi_diff_[dir_num]->cpu_data();
  const Dtype* ci_diff = ci_diff_[dir_num]->cpu_data();
  const Dtype* go_diff = go_diff_[dir_num]->cpu_data();
  const Dtype* gf_diff = gf_diff_[dir_num]->cpu_data();

  Dtype* param_W_i_diff =
      this->blobs_[dir_num * num_blobs_per_dir_]->mutable_cpu_diff();
  Dtype* param_W_c_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 1]->mutable_cpu_diff();
  Dtype* param_W_o_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 2]->mutable_cpu_diff();
  Dtype* param_W_f_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 3]->mutable_cpu_diff();
  Dtype* param_W_i_c_diff = NULL;
  Dtype* param_W_o_c_diff = NULL;
  Dtype* param_W_f_c_diff = NULL;
  if (peephole_) {
    param_W_i_c_diff =
        this->blobs_[dir_num * num_blobs_per_dir_ + 4]->mutable_cpu_diff();
    param_W_o_c_diff =
        this->blobs_[dir_num * num_blobs_per_dir_ + 5]->mutable_cpu_diff();
    param_W_f_c_diff =
        this->blobs_[dir_num * num_blobs_per_dir_ + 6]->mutable_cpu_diff();
  }

  Dtype* bias_b_i_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 7]->mutable_cpu_diff();
  Dtype* bias_b_c_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 8]->mutable_cpu_diff();
  Dtype* bias_b_o_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 9]->mutable_cpu_diff();
  Dtype* bias_b_f_diff =
      this->blobs_[dir_num * num_blobs_per_dir_ + 10]->mutable_cpu_diff();

  for (int RNN = 0; RNN < num_RNN_; ++RNN) {
    // compute gradients w.r.t. parameters
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim, num_,
        (Dtype) 1., gi_diff, X_H_data, (Dtype) 1., param_W_i_diff);
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim, num_,
        (Dtype) 1., ci_diff, X_H_data, (Dtype) 1., param_W_c_diff);
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim, num_,
        (Dtype) 1., go_diff, X_H_data, (Dtype) 1., param_W_o_diff);
    if (not_start) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, X_H_dim,
          num_, (Dtype) 1., gf_diff, X_H_data, (Dtype) 1., param_W_f_diff);
      if (peephole_) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
            num_output_, num_, (Dtype) 1., gi_diff, cstate_prev_data,
            (Dtype) 1., param_W_i_c_diff);
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_,
            num_output_, num_, (Dtype) 1., gf_diff, cstate_prev_data,
            (Dtype) 1., param_W_f_c_diff);
      }
    }
    if (peephole_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output_, num_output_,
          num_, (Dtype) 1., go_diff, cstate_data, (Dtype) 1., param_W_o_c_diff);
    }

    // compute gradients w.r.t. biases
    caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., gi_diff,
        bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_i_diff);
    caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., ci_diff,
        bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_c_diff);
    caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., go_diff,
        bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_o_diff);
    if (not_start) {
      caffe_cpu_gemv<Dtype>(CblasTrans, num_, num_output_, (Dtype) 1., gf_diff,
          bias_multiplier_.cpu_data(), (Dtype) 1., bias_b_f_diff);
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
void ReNetLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
      FillHiddenDiff_cpu(dir_num, step_id, step_end, top[0]);
      ComputeCellDiff_cpu(dir_num, step_id, step_start, step_end);
      Compute_X_H_Diff_cpu(dir_num, step_id, step_start, bottom[0]);
      ComputeParamDiff_cpu(dir_num, step_id, step_start);
    }
  }
}
#ifdef CPU_ONLY
STUB_GPU(ReNetLSTMLayer);
#endif

INSTANTIATE_CLASS(ReNetLSTMLayer);
REGISTER_LAYER_CLASS(ReNetLSTM);
}  // namespace caffe
