CC = cc
CFLAGS = -pedantic -Wall -Wextra -Werror -std=gnu99 -Os -ffast-math \
		 -Wsign-conversion -Wno-unused-parameter -Wno-unused-function -Wno-unneeded-internal-declaration

CXX = c++
CXXFLAGS = -pedantic -Wall -Wextra -Werror -std=c++20 -O2 -ffast-math -march=native

FLEX = flex
FLEXFLAGS = -Pinstant_

BISON = bison
BISONFLAGS = -t -pinstant_

OBJS = ./src/Absyn.o ./src/Buffer.o ./src/Lexer.o ./src/Parser.o ./src/Printer.o

all : insc_jvm insc_llvm

clean :
	rm -f insc_jvm insc_llvm ./src/*.o instant.aux instant.log instant.pdf instant.dvi instant.ps instant

insc_jvm : ${OBJS} ./src/insc_jvm.o
	${CXX} ${CXXFLAGS} ${OBJS} ./src/insc_jvm.o -o insc_jvm

insc_jvm.o : ./src/insc_jvm.cpp
	${CXX} ${CXXFLAGS} -c insc_jvm.cpp

insc_llvm : ${OBJS} ./src/insc_llvm.o
	${CXX} ${CXXFLAGS} ${OBJS} ./src/insc_llvm.o -o insc_llvm

insc_llvm.o : ./src/insc_llvm.cpp
	${CXX} ${CXXFLAGS} -c insc_llvm.cpp

Absyn.o : ./src/Absyn.c ./src/Absyn.h
	${CC} ${CFLAGS} -c Absyn.c

Buffer.o : ./src/Buffer.c ./src/Buffer.h
	${CC} ${CFLAGS} -c Buffer.c

Lexer.c : instant.l
	${FLEX} ${FLEXFLAGS} -oLexer.c instant.l

Parser.c Bison.h : instant.y
	${BISON} ${BISONFLAGS} instant.y -o Parser.c

Lexer.o : CFLAGS+=-Wno-sign-conversion

Lexer.o : Lexer.c Bison.h
	${CC} ${CFLAGS} -c Lexer.c

Parser.o : Parser.c Absyn.h Bison.h
	${CC} ${CFLAGS} -c Parser.c

Printer.o : Printer.c Printer.h Absyn.h
	${CC} ${CFLAGS} -c Printer.c
