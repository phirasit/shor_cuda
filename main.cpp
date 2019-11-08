#include <iostream>

#include "factorization.hpp"
#include "shor_interface.hpp"
#include "shor_cpu.hpp"
#include "shor_gpu.hpp"

static void show_usage(void);

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "number of arguments (" << argc << ")" << "is invalid" << std::endl;
    show_usage();
    return 0;
  }

  const std::string option(argv[1]);
  if (option != "cpu" && option != "gpu") {
    std::cerr << "option (" << option << ") is invalid" << std::endl;
    show_usage();
    return 0;
  }

  try {
    const int N = std::stoi(std::string(argv[2]));
    std::cerr << "N = " << N << std::endl;

    // factorize with conventional algorithm
    const std::pair<int, int> primes = factorization(N);
    if (primes.first < 0) {
      std::cerr << N << " cannot be factorized to two primes" << std::endl;
      return 0;
    }
    std::cerr << N << " = " << primes.first << " x " << primes.second << std::endl;
    std::cerr << std::endl;

    // factorize with Shor's algorithm
    if (option == "cpu") {
      shor_cpu(N).factorize();
    } else if (option == "gpu") {
      shor_gpu(N).factorize();
    }

  } catch (const std::invalid_argument& arg) {
    std::cerr << "N (" << argv[2] << ") is invalid" << std::endl;
    show_usage();
  }

  return 0;
}

void show_usage(void) {
  std::cerr << "usage: ./shor [cpu|gpu] N" << std::endl;
}
