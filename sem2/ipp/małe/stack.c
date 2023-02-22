/* See LICENSE file for copyright and license details. */
#include <inttypes.h>
#include <stdlib.h>

#include "stack.h"
#include "util.h"

Stack
stack_create(void)
{
  return (Stack){ .arr = NULL, .sz = 0, .top = 0 };
}

int
stack_empty(Stack *s)
{
  return !(s->top);
}

size_t
stack_size(Stack *s)
{
  return s->top;
}

uint64_t
stack_top(Stack *s)
{
  return s->arr[s->top - 1];
}

void
stack_pop(Stack *s)
{
  --(s->top);
}

void
stack_push(Stack *s, uint64_t x)
{
  push_back(&(s->arr), x, (s->top)++, &s->sz);
}

void
stack_destroy(Stack *s)
{
  free(s->arr);
}
