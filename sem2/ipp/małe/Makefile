# See LICENSE file for copyright and license details.
# labyrinth - small task for ipp
.POSIX:

include config.mk
REQ = util input stack

all: labyrinth

labyrinth.o: labyrinth.c config.mk $(REQ:=.h)

.c.o:
	$(CC) -o $@ -c $(CFLAGS) $<

labyrinth: labyrinth.o $(REQ:=.o)
	$(CC) -o $@ $(REQ:=.o) labyrinth.o

clean:
	rm -f labyrinth labyrinth.o $(REQ:=.o)
