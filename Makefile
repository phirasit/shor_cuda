all: 
	g++ -o shor *.cpp *.hpp
cuda:
	nvcc -c shor_cuda.cu
cuda_win:
	md ./obj
	md ./bin
	nvcc -c shor_cuda.cu -o obj/shor_cuda.obj
	nvcc -c factorization.cpp -o obj/factorization.obj -I .
	nvcc -c math.cpp -o obj/math.obj -I .
	nvcc -c shor_cpu.cpp -o obj/shor_cpu.obj -I .
	nvcc -c shor_gpu.cpp -o obj/shor_gpu.obj -I .
	nvcc -c shor_interface.cpp -o obj/shor_interface.obj -I .
	nvcc -c main.cpp -o obj/main.obj -I .
	nvcc -o bin/shor obj/shor_cuda.obj obj/factorization.obj obj/math.obj obj/shor_cpu.obj obj/shor_gpu.obj obj/shor_interface.obj obj/main.obj -std c++11

debug:
	g++ *.cpp *.hpp -g -o shor
clean:
	rm shor 
