CXXFLAGS = -Wall -Wextra -O2 -std=c++20
CXX = g++

all: top7

top7.o: top7.cc

top7: top7.o
	$(CXX) -o $@ top7.o

clean:
	rm -f  top7 top7.o
