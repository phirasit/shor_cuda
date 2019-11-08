#include "factorization.hpp"

static bool prime(int N) {
  if (N == 1) return false;
  for (int i = 2; i*i <= N; ++i) {
    if (N % i == 0) {
      return false;
    }
  }
  return true;
}

std::pair<int, int> factorization(int N) {
  for (int i = 2; i * i <= N; ++i) {
    if (N % i == 0) {
      const int ii = N / i;
      if (prime(i) && prime(ii)) {
        return {i, ii};
      } else {
        return {-1, -1};
      }
    }
  }
  return {-1, -1};
}
