all: stencil

stencil:
	nvcc -O2 -arch=all-major -o $@ stencil.cu

clean:
	rm -f stencil stencil.o

.PHONY: all clean
