CC := CC     # use mpiCC when not on okeanos
LFLAGS := 
ALL := blas-dmmmult laplace-seq

all : $(ALL)

blas-dmmmult: blas-dmmmult.cpp
	$(CC) -o $@ $<  # -lblas # uncomment this when not on okeanos

laplace-seq: laplace-seq.o laplace-common.o
	$(CC) $(LFLAGS) -o $@ $^

laplace-common.o: laplace-common.cpp Makefile
	$(CC) -c $<

laplace-seq.o: laplace-seq.cpp Makefile
	$(CC) -c $<

clean :
	rm -f $(ALL)
