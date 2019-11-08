#include "math.hpp"

#include <algorithm>
#include <limits>

// int log2(int a) {
//   // return 31 - __builtin_clz(a);
//   return log2 (a);
// }

int gcd (int n1, int n2) {
    int tmp;
    while (n2 != 0) {
        tmp = n1;
        n1 = n2;
        n2 = tmp % n2;
    }
    return n1;
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
