.POSIX:

CXXFLAGS = -Wall -Wextra -Werror -pedantic -std=c++17
CXXFLAGS += -O3 -ffast-math -march=native -m64
CXXFLAGS += -DDELTA=20
CXXFLAGS += -DENABLE_IOS=1
CXXFLAGS += -DENABLE_PUSH_PULL_PRUNING=1
CXXFLAGS += -DENABLE_HYBRIDIZATION=1
CXXFLAGS += -DHYBRIDIZATION_TAU=0.9
CXXFLAGS += -DENABLE_LOAD_BALANCING=0
LDFLAGS  = 
LDLIBS   = 
CXX      = CC

all: sssp

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

sssp: sssp.o
	$(CXX) -o $@ $(LDFLAGS) sssp.o $(LDLIBS)

clean:
	rm -f sssp *.o

.PHONY: all debug clean
