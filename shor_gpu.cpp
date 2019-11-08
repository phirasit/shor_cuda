#include "shor_gpu.hpp"
#include "math.hpp"

#include <algorithm>
#include <iostream>
#include <cmath>

extern void gpu_prepare_state(cudouble *data, int n, int period);
extern void gpu_hadamard(cudouble *data, int n, int q);
extern void gpu_controlled_rz(cudouble *data, int n, const cudouble omega, const int mask_q);

void shor_gpu::prepare_state(int a) {
  const int period = find_period(a, N);
  gpu_prepare_state(data, n, period);
}

void shor_gpu::hadamard(int q) {
  gpu_hadamard(data, n, q);
}

void shor_gpu::controlled_rz(int q1, int q2, double ang) {
  static const double PI2 = 2.0 * acos(-1.0);
  const cudouble omega = exp(std::complex<double>(1j * PI2 * ang));
  const int mask_q = (1 << q1) | (1 << q2);
  gpu_controlled_rz(data, n, omega, mask_q);
}

int shor_gpu::measure(void) {
  double rand = (double) std::rand() / RAND_MAX;

  for (int i = 0; !(i >> n); ++i) {
    std::complex<double> data_i = data[i];
    const double prob = std::pow(std::abs(data_i), 2.0);
    if (rand <= prob) return i;
    rand -= prob;
  }

  return -1;
}

void shor_gpu::debug(void) {
  for (int i = 0; !(i >> n); ++i) {
    const int z = i >> n;
    std::complex<double> data_i = data[i];
    if (std::pow(std::abs(data_i), 2) > 1e-5)
      std::cerr << z << " = " << data[i] << std::endl;
  }
}

shor_gpu::shor_gpu(int N)
  : shor_interface(N), data(new cudouble[1 << n]) {}

shor_gpu::~shor_gpu() {
  delete data;
}

