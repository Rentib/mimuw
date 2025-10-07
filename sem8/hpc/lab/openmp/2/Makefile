CXX       = g++
CXXFLAGS  = -Wall -Wextra -Werror -pedantic -fopenmp -O2
CXXFLAGS  = -fopenmp -O2
LDFLAGS   = -lm -fopenmp

all: dft

.cpp.o:
	$(CXX) -o $@ -c $(CXXFLAGS) $<

dft: dft.o dft.cpp
	$(CXX) -o $@ dft.o $(LDFLAGS)

clean:
	rm -f *.o dft

.PHONY: all clean
