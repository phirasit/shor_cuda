#ifndef __MATH_HPP__
#define __MATH_HPP__

#include "bigint.hpp"

// int log2(int a);
int gcd (int n1, int n2);
int pow_mod(int a, int b, int N);
int random_success(double prob);

int find_period(int a, int N);

#endif // __MATH_HPP__
