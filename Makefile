all:
	g++ *.cpp *.hpp -o shor
debug:
	g++ *.cpp *.hpp -g -o shor
clean:
	rm shor 
