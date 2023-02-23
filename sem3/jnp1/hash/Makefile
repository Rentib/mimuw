FLAGS    = -Wall -Wextra -O2
CXXFLAGS = $(FLAGS) -std=c++17
CFLAGS 	 = $(FLAGS) -std=c17

all:
	g++ $(CXXFLAGS) -c hash.cc -o hash.o
	gcc $(CFLAGS) -c hash_test1.c -o hash_test1.o
	g++ $(CXXFLAGS) -c hash_test2.cc -o hash_test2.o
	g++ hash_test1.o hash.o -o hash_test1
	g++ hash_test2.o hash.o -o hash_test21
	g++ hash.o hash_test2.o -o hash_test22

clean:
	rm -f *.o hash_test1 hash_test21 hash_test22
