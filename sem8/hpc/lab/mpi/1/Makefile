# use cc on okeanos
CC     := mpicc
CFLAGS := -O3 -c -Wall
LFLAGS := -O3
ALL    := bench.exe

all : $(ALL)

%.exe : %.o
	$(CC) $(LFLAGS) -o $@ $<

%.o : %.c
	$(CC) $(CFLAGS) $<

clean :
	rm -f *.o *.out *.err $(ALL)
