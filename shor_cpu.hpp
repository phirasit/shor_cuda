#ifndef __SHOR_CPU_HPP__
#define __SHOR_CPU_HPP__

#include "shor_interface.hpp"

#include <complex>

using cdouble = std::complex<double>;

class shor_cpu : public shor_interface {

  private:
    cdouble * const data;

  public:
    void prepare_state(int a);
    void hadamard(int q);
    void controlled_rz(int q1, int q2, double ang);
    int measure(void);

    void debug(void);

    shor_cpu(int n);
    ~shor_cpu();
};

#endif // __SHOR_CPU_HPP__
