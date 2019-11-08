#include "shor_cuda.hpp"
#include "math.hpp"

#include <algorithm>
#include <iostream>
#include <cmath>

__global__ prepare_state_cuda(cdouble amp, int period, int n) {
  for (int i = 0; !(i >> n); ++i) {
    data[i] = (i % period == 0 ? amp : 0.0);
  }
}

void shor_cuda::prepare_state(int a) {
  const int period = find_period(a, N);
  const int total_period = ((1 << n) - 1) / period + 1;
  const cdouble amp = 1.0 / sqrt(total_period);
  
  prepare_state_cuda<<<1, 1>>>(amp, period, n);
  // cudaMemcpy(data_cuda, data, N*sizeof(CUCOMPLEX), cudaMemcpyHostToDevice);
}

__global__ hadamard_cuda(cdouble sqrt_1_2, int mask_q) {
  for (int i = 0; !(i >> n); ++i) {
    if (i & mask_q) continue;
    const int ii = i ^ mask_q;
    const cdouble a = sqrt_1_2 * (data[i] + data[ii]);
    const cdouble b = sqrt_1_2 * (data[i] - data[ii]);
    data[i] = a;
    data[ii] = b;
  }
}

void shor_cuda::hadamard(int q) {
  static const cdouble sqrt_1_2 = sqrt(0.5);
  const int mask_q = 1 << q;
  
  hadamard_cuda<<<1,1>>>(sqrt_1_2, mask_q);
}

__global__ controlled_rz_cuda(int mask_q, cdouble omega) {
  for (int i = 0; !(i >> n); ++i) {
    if ((~i) & mask_q) continue;
    data[i] *= omega;
  }
}

void shor_cuda::controlled_rz(int q1, int q2, double ang) {
  static const double PI2 = 2.0 * acos(-1.0);
  const cdouble omega = exp(cdouble(1j * PI2 * ang));
  const int mask_q = (1 << q1) | (1 << q2);
  
  controlled_rz_cuda<<<1,1>>>(mask_q, omega);
}

int shor_cuda::measure(void) {
  cdouble * host_data = new cdouble[n];
  cudaMemcpy(host_data, data, n*sizeof(cdouble), cudaMemcpyDeviceToHost); 

  double rand = (double) std::rand() / RAND_MAX;
  for (int i = 0; !(i >> n); ++i) {
    const double prob = std::pow(std::abs(host_data[i].real()), 2.0);
    if (rand <= prob) {
      free(host_data);
      return i;
    }
    rand -= prob;
  }

  delete host_data;
  return -1;
}

void shor_cuda::debug(void) {
  for (int i = 0; !(i >> n); ++i) {
    const int z = i >> n;
    if (std::pow(std::abs(data[i]), 2) > 1e-5)
      std::cerr << z << " = " << data[i] << std::endl;
  }
}

shor_cuda::shor_cpuda(int N)
  : shor_interface(N) {
    cudaMalloc(&data, N*sizeof(cdouble));
  }

shor_cuda::~shor_cuda() {
  // delete data;
  cudaFree(data);
}

