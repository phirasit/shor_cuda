all:
	g++ *.cpp *.hpp -o shor
cuda:
	nvcc *.cpp *.hpp -o shor_cuda
debug:
	g++ *.cpp *.hpp -g -o shor
clean:
	rm shor 
