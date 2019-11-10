#ifndef __SHOR_GPU_HPP__
#define __SHOR_GPU_HPP__

#include "shor_interface.hpp"

#include <complex>
#include <thrust/complex.h>

using cudouble = thrust::complex<double>;

class shor_gpu : public shor_interface {

  private:
    cudouble* const buffer;
    cudouble * data;
    int sm;

  public:
    void prepare_state(int a);
    void hadamard(int q);
    void controlled_rz(int q1, int q2, double ang);
    int measure(void);

    void debug(void);

    shor_gpu(int n, int sm);
    ~shor_gpu();
};

#endif // __SHOR_GPU_HPP__
