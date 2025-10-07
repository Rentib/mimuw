CXX = g++
CXXFLAGS  = -Wall -Wextra -Werror -pedantic -fopenmp
LDFLAGS = -lm -lomp

all: mat_mult

.cpp.o:
	$(CXX) -o $@ -c $(CXXFLAGS) $<

mat_mult: mat_mult.o mat_mult.cpp
	$(CXX) -o $@ mat_mult.o $(LDFLAGS)

clean:
	rm -f *.o mat_mult
