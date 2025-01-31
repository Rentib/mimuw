# MRJP - Instant

## Requirements

- Cgnu99 compiler.
- C++20 compiler.
- flex
- bison

## Compilation

In order to compile the program one must first generate parser of instant using
`bnfc` for C language. Parser for C++ leaks memory and should not be used.
Afterwards it is enough to run:
```
    make
```

## How to run

First build the program and then for jvm run
```
    ./insc_jvm file.ins
```
for llvm run
```
    ./insc_llvm file.ins
```
