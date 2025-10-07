#
# A template for the 2016 MPI lab at the University of Warsaw.
# Copyright (C) 2016, Konrad Iwanicki.
# Modified: 2018, Krzysztof Rzadca
#
CC          := CC
# Available flags:
# -DUSE_RANDOM_GRAPH=1   --- generates a random graph
# -DUSE_RANDOM_SEED=123  --- uses a given seed to generate a random graph
CFLAGS      := -c -DUSE_RANDOM_GRAPH=1 -DUSE_RANDOM_SEED=123
LFLAGS      := 
ALL         := floyd-warshall-par.exe


all : $(ALL)

clean :
	rm -f *.o *.out *.err $(ALL) $(ALL)+pat $(ALL)+apa

floyd-warshall-par.exe : floyd-warshall-par.o graph-utils-par.o graph-base.o
	$(CC) $(LFLAGS) -o $@ $^

%.o : %.cpp graph-base.h graph-utils.h Makefile
	$(CC) $(CFLAGS) $<

