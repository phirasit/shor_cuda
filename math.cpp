#include "math.hpp"

#include <algorithm>
#include <limits>

int log2(int a) {
  return 31 - __builtin_clz(a);
}

int pow_mod(int a, int b, int N) {
  long long ans = 1LL, p = a;
  while (b) {
    if (b & 1) ans = (ans * p) % N;
    p = (p * p) % N;
    b >>= 1;
  }
  return ans;
}

int random_success(double prob) {
  return std::rand() <= (int) (prob * RAND_MAX);
}

int find_period(int a, int N) {
  int p = 1;
  long long x = a;
  while (x != 1LL) {
    x = (a * x) % N;
    ++p;
  }
  return p;
}
