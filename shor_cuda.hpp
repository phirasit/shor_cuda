#ifndef __SHOR_CPU_HPP__
#define __SHOR_CPU_HPP__

#include "shor_interface.hpp"

// #include <complex>
#include <thrust/complex.h>

using cdouble = thrust::complex<float>;

class shor_cuda : public shor_interface {

  private:
    cdouble * data;

  public:
    void prepare_state(int a);
    void hadamard(int q);
    void controlled_rz(int q1, int q2, double ang);
    int measure(void);

    void debug(void);

    shor_cuda(int n);
    ~shor_cuda();
};

#endif // __SHOR_CPU_HPP__
