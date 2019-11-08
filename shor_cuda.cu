#include <thrust/complex.h>

using cudouble = thrust::complex<float>;

__global__ void prepare_state_cuda(cudouble * data, cudouble amp, int period, int n) {
  for (int i = 0; !(i >> n); ++i) {
    data[i] = (i % period == 0 ? amp : 0.0);
  }
}

void gpu_prepare_state(cudouble * data, int n, int period, int a) {
  const int total_period = ((1 << n) - 1) / period + 1;
  const cudouble amp = 1.0 / sqrt(total_period);
  
  prepare_state_cuda<<<1,1>>>(data, amp, period, n);
}

__global__ void hadamard_cuda(cudouble *data, cudouble sqrt_1_2, int mask_q, int n) {
  for (int i = 0; !(i >> n); ++i) {
    if (i & mask_q) continue;
    const int ii = i ^ mask_q;
    const cudouble a = sqrt_1_2 * (data[i] + data[ii]);
    const cudouble b = sqrt_1_2 * (data[i] - data[ii]);
    data[i] = a;
    data[ii] = b;
  }
}

void gpu_hadamard(cudouble * data, int n, int q) {
  static const cudouble sqrt_1_2 = sqrt(0.5);
  const int mask_q = 1 << q;
  
  hadamard_cuda<<<1,1>>>(data, sqrt_1_2, mask_q, n);
}

__global__ void controlled_rz_cuda(cudouble *data, int mask_q, cudouble omega, int n) {
  for (int i = 0; !(i >> n); ++i) {
    if ((~i) & mask_q) continue;
    data[i] *= omega;
  }
}

void gpu_controlled_rz(cudouble * data, int n, int q1, int q2, double ang) {
  static const double PI2 = 2.0 * acos(-1.0);
  const cudouble omega = exp(cudouble(1 * PI2 * ang));
  const int mask_q = (1 << q1) | (1 << q2);
  
  controlled_rz_cuda<<<1,1>>>(data, mask_q, omega, n);
}

void gpuMalloc(void *data, int n) {
  cudaMalloc((void **)&data, n*sizeof(cudouble));
}

void gpuFree(void *data) {
  cudaFree(data);
}

void gpuMemcpy(void * dst, void * src, int size, int dir) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost ); 
}