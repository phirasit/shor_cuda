#include "shor_cpu.hpp"
#include "math.hpp"

#include <cassert>
#include <algorithm>
#include <iostream>
#include <cmath>

void shor_cpu::prepare_state(int a) {
  const int period = find_period(a, N);
  const int total_period = ((1 << n) - 1) / period + 1;
  const cdouble amp = 1.0 / sqrt(total_period);
  for (int i = 0; !(i >> n); ++i) {
    data[i] = (i % period == 0 ? amp : 0.0);
  }
}

void shor_cpu::hadamard(int q) {
  static const cdouble sqrt_1_2 = sqrt(0.5);
  const int mask_q = 1 << q;
  for (int i = 0; !(i >> n); ++i) {
    if (i & mask_q) continue;
    const int ii = i ^ mask_q;
    const cdouble a = sqrt_1_2 * (data[i] + data[ii]);
    const cdouble b = sqrt_1_2 * (data[i] - data[ii]);
    data[i] = a;
    data[ii] = b;
  }
}

void shor_cpu::controlled_rz(int q1, int q2, double ang) {
  static const double PI2 = 2.0 * acos(-1.0);
  const cdouble omega = exp(cdouble(0.0, PI2 * ang));
  const int mask_q = (1 << q1) | (1 << q2);
  for (int i = 0; !(i >> n); ++i) {
    if ((~i) & mask_q) continue;
    data[i] *= omega;
  }
}

int shor_cpu::measure(void) {
  double rand = (double) std::rand() / RAND_MAX;

  for (int i = 0; !(i >> n); ++i) {
    const double prob = std::pow(std::abs(data[i]), 2.0);
    if (rand <= prob) return i;
    rand -= prob;
  }

  assert(false);
  return -1;
}

void shor_cpu::debug(void) {
  for (int i = 0; !(i >> n); ++i) {
    const int z = i >> n;
    if (std::pow(std::abs(data[i]), 2) > 1e-5)
      std::cerr << z << " = " << data[i] << std::endl;
  }
}

shor_cpu::shor_cpu(int N)
  : shor_interface(N), data(new cdouble[1 << n]) {}

shor_cpu::~shor_cpu() {
  delete data;
}

