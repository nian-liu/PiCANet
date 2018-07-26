//  @zyan: define commonly used device functions
#ifndef CAFFE_COMMON_MATH_CUH_
#define CAFFE_COMMON_MATH_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

//namespace caffe {


template<typename Dtype>
static __inline__ __device__ Dtype sigmoid_dev(Dtype x);

template<>
__inline__ __device__ float sigmoid_dev(float x) {
  return 1. / (1. + expf(-x));
}

template<>
__inline__ __device__ double sigmoid_dev(double x) {
  return 1. / (1. + exp(-x));
}


template<typename Dtype>
static __inline__ __device__ Dtype sigmoid_diff_y_dev(Dtype y) {
  return y * (1.0 - y);
}

template<typename Dtype>
static __inline__ __device__ Dtype tanh_dev(Dtype x) {
  return 2. * sigmoid_dev<Dtype>(2. * x) - 1.;
}

template<typename Dtype>
static __inline__ __device__ Dtype tanh_diff_x_dev(Dtype x) {
  Dtype y = tanh_dev<Dtype>(x);
  return 1.0 - y * y;
}

template<typename Dtype>
static __inline__ __device__ Dtype tanh_diff_y_dev(Dtype y) {
  return 1.0 - y * y;
}

static __inline__ __device__ int blob_offset(int channels, int height,
    int width, int n, int ch, int y, int x) {
  return ((n * channels + ch) * height + y) * width + x;
}

template<typename Dtype>
static __inline__ __device__ Dtype atomicAdd_dev(Dtype* address, Dtype val);

template<>
__inline__ __device__ float atomicAdd_dev(float* address, float val) {
  return ::atomicAdd(address, val);
}

template<>
__inline__ __device__ double atomicAdd_dev(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val +
            __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
  return __longlong_as_double(old);
}
//}  //  namespace caffe

#endif  //  #ifndef CAFFE_COMMON_MATH_CUH_
