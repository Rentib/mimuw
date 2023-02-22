labirynt - small task for ipp
=============================
labirynt is a simple program for calculating distance between two vertices in 
a k-dimensional graph written in <code>C</code>.

Algorithm
---------
Labirynt uses simple bfs as its main algorithm, it does so layer by layer which allows to avoid passing distance to queue and makes calculating
it on the fly possible. What's more, we can use stacks instead of queues.
s[0] stores vertices that have distance 0 modulo 2 from the source, 
s[1] stores vertices that have distance 1 modulo 2 from the source.
Generating adjaccent vertices uses the fact that when going from (x_1, ..., x_i ..., x_k) to (x_1, ..., x_i +/- 1, ..., x_k), 
we change vertex v by pref[i].
It is possible that some of the coordinates are 1 or n_i, then we need to check if we stay in the cube (look for /* overflow */ in bfs function).

Calculating cube numbers
------------------------
First we calculate start and cubes numbers by iterating over dimensions as described in moodle.
It is very slow and requires O(k) operations. We can use the fact that when moving in BFS we always go to
adjaccent cube, thus changing only 1 coordinate. When changing i-th coordinate by one we just need to add 
or subtract n_1 * n_2 * ... * n_{i-1} from previous cube's number. This obvoiusly implies usage of an array with 
preprocessed values (*pref* in code).

Bit magic
---------
It is important to understand that % and / operators work very slow on 64 bit numbers, hence the following improvement:
  x % (2^n) = x - (x / (2^n)) * 2^n = x & (2^n - 1) = x & ((1 << n) - 1)
and if n is constant we can just precalculate ((1 << n) - 1).
maze_key is a variable used to store information about structure of given maze. It is supposed to be given in line 4 of input.
For example:
- x % 64 == x & 63
- x * 64 == x << 6
- x / 64 == x >> 6
- x % 16 == x & 15
- x * 16 == x << 4

Coding style
------------
Code for labirynt was created as according to [printf](https://suckless.org/coding_style/),
Important things here:
* Do not use for loop initial declarations
* All variable declarations at top of block.
* Function name and argument list on next line. This allows to grep for function names simply using grep ^functionname(.
* (switch) Do not indent cases another level.
Also I was thinking of changing 64 to some defined value but it worsens the clarity of the code.

Building
--------
Make options can be configured in *config.mk*.

Testing
-------
In order to test labirynt you can use *test.sh*.
It creates temporary files *prog.out*, *prog.err* and *val.err*, so it is important to ensure that no other files are named the same.
Make sure to run it using a shell, which supports color (preferably bash or zsh).
