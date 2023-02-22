/* See LICENSE file for copyright and license details. */
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"

void
die(const char *fmt, ...)
{
  va_list ap;

  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  exit(1);
}

void 
free_mult(size_t cnt, ...)
{
  va_list ap;

  va_start(ap, cnt);
  while (cnt--)
    free(va_arg(ap, void *));
  va_end(ap);
}

void
push_back(uint64_t **v, uint64_t x, size_t i, size_t *sz)
{
  if (i >= *sz) {
    *sz = MAX(1, *sz << 1);
    *v = safe_realloc(*v, sizeof(uint64_t) * (*sz));
  }
  (*v)[i] = x;
}

void *
safe_malloc(size_t sz)
{
  void *p;
  if (!(p = malloc(sz))) 
    die("ERROR %d\n", 0);
  return p;
}

void *
safe_calloc(size_t n, size_t sz)
{
  void *p;
  if (!(p = calloc(n, sz)))
    die("ERROR %d\n", 0);
  return p;
}

void *
safe_realloc(void *ptr, size_t sz)
{
  void *p;
  if (!(p = realloc(ptr, sz))) 
    die("ERROR %d\n", 0);
  return p;
}
