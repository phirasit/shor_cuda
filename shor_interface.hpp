#ifndef __SHOR_INTERFACE_HPP__
#define __SHOR_INTERFACE_HPP__

class shor_interface {

  protected:
    const int N;
    const int n;

  private:
    int simulate_quantum_circuit(int a);
    int test_period_using_cf(int a, int y);

  public:
    virtual void prepare_state(int a) = 0;
    virtual void hadamard(int q) = 0;
    virtual void controlled_rz(int q1, int q2, double ang) = 0;
    virtual int measure() = 0;

    virtual void debug(void);
    virtual void factorize(void);

    shor_interface(int N);
};

#endif // __SHOR_INTERFACE_HPP__
