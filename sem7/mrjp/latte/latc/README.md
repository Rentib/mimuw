# latc - Latte Compiler

Latte language compiler for x86_64 architecture compatible with System V ABI.

## Building

```
    make
```

## Running

```
    ./latc_x86_64 <file>
```

## Testing

```
    ./test.sh
```

### Dependencies
- cabal 3.12.1.0
- ghc 9.4.8
- nasm 2.16.03
- clang 19.1.6

## lang

Language grammar as well as syntax highlighting for vim are in `lang/`.

# Points
| Part               | Points  |
| :----------------: | ------: |
| front-end          |  4 *pt* |
| asm                | 10 *pt* |
| lcse               |  3 *pt* |
| sr                 |  2 *pt* |
| structures         |  2 *pt* |
| objects            |  4 *pt* |
| virtual methods    |  4 *pt* |
| **total**          | 29 *pt* |

* Frontend - typechecks and ensures that every function returns;
* ASM - compiler generates NASM code for x86_64 architecture;
* SSA - custom algorithm
* DCE - deadcode elimination
* LCSE - local subexpression elimination together with copy propagation;
* SR - partial strength reduction without loops optimization;
* objects - classes with attributes, methods, virtual methods (allow inheritance).

## if, if/else, while

All those statements create a new block, so one may declare variables inside
them without redeclaration errors.

## Classes

Classes that inherit from others MUST be defined after the ones that they extend.
Virtual methods MUST NOT change the signature.
