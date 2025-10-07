.POSIX:

CXXFLAGS =
LDFLAGS  =
LDLIBS   = -lcurand -lcudart
CXX      = nvcc

all: CXXFLAGS += -O3 -Xptxas -O3,-v
all: acotsp

debug: CXXFLAGS += -g -G
debug: CXXFLAGS += -DDEBUG
debug: acotsp

.cu.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

acotsp: acotsp.o
	$(CXX) -o $@ $(LDFLAGS) acotsp.o $(LDLIBS)

clean:
	rm -f acotsp *.o

.SUFFIXES: .cu .o
.PHONY: all clean
