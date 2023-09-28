# See LICENSE file for copyright and license details.

# radio - zadanie zaliczeniowe na SIK

CFLAGS = -std=c17 -pedantic -Wall -Wextra -Werror -O2 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE
LDFLAGS = -pthread

CC = cc

SRC = bitset.c cyclicbuffer.c printer.c queue.c receiver.c util.c
OBJ = $(SRC:.c=.o)

all: receiver sender

.c.o:
	$(CC) -c $(LDFLAGS) $(CFLAGS) $<

receiver: $(OBJ) control.o
	$(CC) $(LDFLAGS) -o sikradio-receiver $(OBJ) control.o

sender: $(OBJ) sender.o
	$(CC) $(LDFLAGS) -o sikradio-sender $(OBJ) sender.o

clean:
	rm -f sikradio-receiver sikradio-sender control.o sender.o $(OBJ)
