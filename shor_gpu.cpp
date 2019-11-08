#include "shor_gpu.hpp"
#include "math.hpp"

#include <algorithm>
#include <iostream>
#include <cmath>

extern void gpu_prepare_state(cudouble * data, int n, int period, int a);
extern void gpu_hadamard(cudouble * data, int n, int q);
extern void gpu_controlled_rz(cudouble * data, int n, int q1, int q2, double ang);
extern void gpuMalloc(void *data, int n);
extern void gpuFree(void *data);
extern void gpuMemcpy(void * dst, void * src, int size, int dir) ;

#define cudaMemcpyDeviceToHost  2


// __global__ void prepare_state_cuda(cudouble * data, cudouble amp, int period, int n) {
//   for (int i = 0; !(i >> n); ++i) {
//     data[i] = (i % period == 0 ? amp : 0.0);
//   }
// }

void shor_gpu::prepare_state(int a) {
  const int period = find_period(a, n);
  gpu_prepare_state(data, n, period, a);
}

// __global__ void hadamard_cuda(cudouble *data, cudouble sqrt_1_2, int mask_q, int n) {
//   for (int i = 0; !(i >> n); ++i) {
//     if (i & mask_q) continue;
//     const int ii = i ^ mask_q;
//     const cudouble a = sqrt_1_2 * (data[i] + data[ii]);
//     const cudouble b = sqrt_1_2 * (data[i] - data[ii]);
//     data[i] = a;
//     data[ii] = b;
//   }
// }

void shor_gpu::hadamard(int q) {
  gpu_hadamard(data, n, q);
}

// __global__ void controlled_rz_cuda(cudouble *data, int mask_q, cudouble omega, int n) {
//   for (int i = 0; !(i >> n); ++i) {
//     if ((~i) & mask_q) continue;
//     data[i] *= omega;
//   }
// }

void shor_gpu::controlled_rz(int q1, int q2, double ang) {
  gpu_controlled_rz(data, n, q1, q1, ang);
}

int shor_gpu::measure(void) {
  cudouble * host_data = new cudouble[n];
  // cudaMemcpy(host_data, data, n*sizeof(cudouble), cudaMemcpyDeviceToHost); 
  gpuMemcpy(host_data, data, n*sizeof(cudouble), cudaMemcpyDeviceToHost);

  double rand = (double) std::rand() / RAND_MAX;
  std::complex<float> data_i;
  for (int i = 0; !(i >> n); ++i) {
    data_i = host_data[i];
    const double prob = std::pow(std::abs(data_i), 2.0);
    if (rand <= prob) {
      delete host_data;
      return i;
    }
    rand -= prob;
  }

  delete host_data;
  return -1;
}

void shor_gpu::debug(void) {
  cudouble * host_data = new cudouble[n];
  gpuMemcpy(host_data, data, n*sizeof(cudouble), cudaMemcpyDeviceToHost); 

  std::complex<float> data_i;
  for (int i = 0; !(i >> n); ++i) {
    const int z = i >> n;
    data_i = host_data[i];
    if (std::pow(std::abs(data_i), 2) > 1e-5)
      std::cerr << z << " = " << data_i << std::endl;
  }
}

shor_gpu::shor_gpu(int N)
  : shor_interface(N) {
    // cudaMalloc((void **)&data, N*sizeof(cudouble));
    gpuMalloc((void **)&data, N*sizeof(cudouble));
  }

shor_gpu::~shor_gpu() {
  // delete data;
  // cudaFree(data);
  gpuFree(data);
}

