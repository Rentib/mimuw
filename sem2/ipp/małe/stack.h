/* See LICENSE file for copyright and license details. */
#ifndef STACK_H
#define STACK_H

#include <inttypes.h>
#include <stdlib.h>

typedef struct Stack Stack;
struct Stack {
  uint64_t *arr; /* pointer for storing elements */
  size_t sz;     /* size of arr (not the number of elements in it) */
  size_t top;
};

Stack stack_create(void);
int stack_empty(Stack *);
size_t stack_size(Stack *);
uint64_t stack_top(Stack *);
void stack_push(Stack *, uint64_t);
void stack_pop(Stack *);
void stack_destroy(Stack *);

#endif
