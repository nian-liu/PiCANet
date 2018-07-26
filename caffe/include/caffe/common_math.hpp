#ifndef CAFFE_COMMON_MATH_HPP_
#define CAFFE_COMMON_MATH_HPP_

#include <cmath>

namespace caffe {

template<typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template<typename Dtype>
inline Dtype sigmoid_diff_y(Dtype y) {
  return y * (1.0 - y);
}

template<typename Dtype>
inline Dtype tanh(Dtype x) {
  return 2. * sigmoid<Dtype>(2. * x) - 1.;
}

template<typename Dtype>
inline Dtype tanh_diff_x(Dtype x) {
  Dtype y = tanh<Dtype>(x);
  return 1.0 - y * y;
}

template<typename Dtype>
inline Dtype tanh_diff_y(Dtype y) {
  return 1.0 - y * y;
}

template float sigmoid<float>(float x);
template double sigmoid<double>(double x);

template float sigmoid_diff_y<float>(float y);
template double sigmoid_diff_y<double>(double y);

template float tanh<float>(float x);
template double tanh<double>(double x);

template float tanh_diff_x<float>(float x);
template double tanh_diff_x<double>(double x);

template float tanh_diff_y<float>(float y);
template double tanh_diff_y<double>(double y);

}  //  namespace caffe

#endif  //  #ifndef CAFFE_COMMON_MATH_HPP_
