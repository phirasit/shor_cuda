PROGRAM = shor
CC 		= g++
CFLAGS 	= -Wall -pedantic -std=c++11 -I .
LDFLAGS = -lcuda -lcudart

C_FILES := $(wildcard *.cpp)
OBJS 	:= $(patsubst %.cpp, %.o, $(C_FILES))


all: $(PROGRAM)
$(PROGRAM): $(OBJS) | cuda
	$(CC) $(CFLAGS) $(OBJS) shor_cuda.o $(LDFLAGS) -o $(PROGRAM)
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
%: %.cpp
	$(CC) $(CFLAGS) -o $@ $<
clean:
	rm -f *.o
	rm shor
.PHONY: clean depend

cuda:
	nvcc -c shor_cuda.cu -o shor_cuda.o
