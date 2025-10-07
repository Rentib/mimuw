#
# A template for the 2016 MPI lab at the University of Warsaw.
# Copyright (C) 2016, Konrad Iwanicki.
#
CC          := mpic++  # or CC on okeanos
# Available flags:
# -DUSE_RANDOM_GRAPH=1   --- generates a random graph
# -DUSE_RANDOM_SEED=123  --- uses a given seed to generate a random graph
CFLAGS      := -O3 -Wall -c -DUSE_RANDOM_GRAPH=1 -DUSE_RANDOM_SEED=123
LFLAGS      := -O3 -Wall
ALL         := \
	generator-seq.exe \
	generator-par.exe \
	hello-world-bcast-par.exe \
	floyd-warshall-seq.exe \
	floyd-warshall-par.exe


all : $(ALL)

%-par.exe : %-par.o graph-utils-par.o graph-base.o
	$(CC) $(LFLAGS) -o $@ $^

%-seq.exe : %-seq.o graph-utils-seq.o graph-base.o
	$(CC) $(LFLAGS) -o $@ $^

%.o : %.cpp graph-base.h graph-utils.h Makefile
	$(CC) $(CFLAGS) $<

clean :
	rm -f *.o *.out *.err $(ALL)
