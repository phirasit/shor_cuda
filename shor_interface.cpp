#include "shor_interface.hpp"
#include "math.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

void shor_interface::debug(void) {
  // do nothing
}

int shor_interface::simulate_quantum_circuit(int a) {
  // power operation
  prepare_state(a);

  // QFT
  /*
  for (int i = n-1; i >= 0; --i) {
    for (int j = i+1; j < n; ++j) {
      controlled_rz(i, j, std::pow(0.5, j-i+1));
    }
    hadamard(i);
  }
  */

  // inverse QFT
  for (int i = 0; i < n; ++i) {
    hadamard(i);
    for (int j = i+1; j < n; ++j) {
      controlled_rz(j, i, -1.0 / std::pow(2.0, j-i+1));
    }
  }

  // debug();

  return measure();
}

int shor_interface::test_period_using_cf(int a, int y) {
  static const int DEPTH = 20;
  std::vector<int> cf;

  // if (y == 0) return 0;

  int z = 1 << n;
  std::cerr << "Fraction is " << y << " / " << z << std::endl;
  while (z && cf.size() < DEPTH) {
    cf.push_back(y / z);
    y %= z;
    std::swap(y, z);

    // test with cf
    long long d = 1LL, s = 0LL;
    for (int j = cf.size()-1; j >= 0; --j) {
      std::swap(d, s);
      d += cf[j] * s;

      const long long g = gcd(d, s);
      d /= g;
      s /= g;
    }

    std::cerr << "Possible period (" << d << " / " << s << ")" << std::endl;
    // if s is period
    if (pow_mod(a, s+1, N) == 1) {
      return s+1;
    }
  }

  return 0;
}

void shor_interface::factorize(void) {
  // static const int THRESHOLD_ROUND = 30;
  std::cerr << "number of qbits = " << n << std::endl;

  // for (int round = 0; round < THRESHOLD_ROUND; ++round) {
  while (true) {
    // get a random value
    const int a = rand() % (N-2) + 2;
    std::cerr << "choose a = " << a << std::endl;
    int _gcd = gcd(a, N);
    if (_gcd != 1) {
      // too easy continue
      std::cerr << " // answer is found accidentally" << std::endl;
      std::cerr << " // Answer: " << N << " = " << _gcd << " x " << (N/_gcd) << std::endl;
      continue;
    }

    const int y = simulate_quantum_circuit(a);
    const int period = test_period_using_cf(a, y);

    if (period) {
      std::cerr << "Solution is found" << std::endl;
      std::cerr << "period = " << period << std::endl;
      if (period % 2 == 1) {
        std::cerr << "Period is odd: try again" << std::endl;
        continue;
      }
      const int check = pow_mod(a, period / 2, N);
      if (check == 1 || check == N-1) {
        std::cerr << "Not the best period" << std::endl;
        continue;
      }

      // find the 2 primes
      const bigint x = bigint(a) ^ bigint(period >> 1);
      const bigint p1 = gcd(x-1, N), p2 = gcd(x+1, N);

      std::cerr << "Shor's algorithm result " << N << " = " << p1 << " x " << p2 << std::endl;

      return;
    }
  }

  // no solution found
  std::cerr << "No solution found" << std::endl;
}

shor_interface::shor_interface(int N)
  : N(N), n(2*(log2(N-1)+1)) {}
