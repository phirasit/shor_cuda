# Shor's algorithm simulation using CUDA

This is Shor's algorithm period finding simulator which will require 2n qbuits for factorizing n bits number.
The algorithm is implemented to be run on both traditional CPU and Nvidia GPU using CUDA®.

## Build
1. make sure CUDA library is installed in your computer
2. run ```make```

## Run
just run ```./shor [cpu|gpu] N``` where N is the number you wish to factorize
