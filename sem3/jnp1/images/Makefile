CXX = g++
CXXFLAGS = -std=c++20 -pedantic -Wall -Wextra -O2

REQ = bmp color coordinate

all: example

example.o: example.cc $(REQ:=.h) images.h
images_test2.o: images_test2.cc $(REQ:=.h) images.h functional.h test.h

.cc.o:
	$(CXX) -o $@ -c $(CXXFLAGS) $<

example: example.o $(REQ:=.o) images.h
	$(CXX) -o $@ $(REQ:=.o) example.o

clean:
	rm -f example example.o $(REQ:=.o) images.o
