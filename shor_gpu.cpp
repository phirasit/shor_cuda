#include "shor_gpu.hpp"
#include "math.hpp"

#include <algorithm>
#include <iostream>
#include <cmath>

extern void gpu_prepare_state(int sm, cudouble *data, int n, int period);
extern void gpu_hadamard(int sm, cudouble *data, int n, int q);
extern void gpu_controlled_rz(int sm, cudouble *data, int n, const cudouble omega, const int mask_q);
extern void gpu_init(cudouble **data, int n);
extern void gpu_deinit(cudouble *data);
extern void gpu_memcpy(cudouble *dst, cudouble *src, int n);

void shor_gpu::prepare_state(int a) {
  const int period = find_period(a, N);
  gpu_prepare_state(sm, data, n, period);
}

void shor_gpu::hadamard(int q) {
  gpu_hadamard(sm, data, n, q);
}

void shor_gpu::controlled_rz(int q1, int q2, double ang) {
  static const double PI2 = 2.0 * acos(-1.0);
  const cudouble omega = exp(std::complex<double>(0.0, PI2 * ang));
  const int mask_q = (1 << q1) | (1 << q2);
  gpu_controlled_rz(sm, data, n, omega, mask_q);
}

int shor_gpu::measure(void) {
  double rand = (double) std::rand() / RAND_MAX;
  cudouble * tmp = new cudouble[1 << n];
  gpu_memcpy(tmp, data, n);

  for (int i = 0; !(i >> n); ++i) {
    std::complex<double> data_i = tmp[i];
    const double prob = std::pow(std::abs(data_i), 2.0);
    if (rand <= prob) return i;
    rand -= prob;
  }

  return -1;
}

void shor_gpu::debug(void) {
  cudouble * tmp = new cudouble[1 << n];
  gpu_memcpy(tmp, data, n);

  for (int i = 0; !(i >> n); ++i) {
    const int z = i >> n;
    std::complex<double> data_i = tmp[i];
    if (std::pow(std::abs(data_i), 2) > 1e-5)
      std::cerr << z << " = " << data[i] << std::endl;
  }
}

shor_gpu::shor_gpu(int N, int sm)
 : shor_interface(N), sm(sm) {
    // data(new cudouble[1 << n]) 
    gpu_init(&data, n);
  }

shor_gpu::~shor_gpu() {
  // delete data;
  gpu_deinit(data);
}

