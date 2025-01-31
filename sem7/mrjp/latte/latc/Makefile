all: latc runtime

src/Parser: lang/Latte.cf
	env --chdir=./lang/ sh setup.sh

latc: src/Parser runtime src/*
	cabal build -O2 -j

runtime: lib/runtime.c
	clang -c -Os -o ./lib/runtime.o ./lib/runtime.c

clean:
	cabal clean
	rm -f ./lib/runtime.o

.PHONY: all clean
