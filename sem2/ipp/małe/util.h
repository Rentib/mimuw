/* See LICENSE file for copyright and license details. */
#ifndef UTIL_H
#define UTIL_H

#include <inttypes.h>
#include <stdlib.h>

#define LEN(X)        (sizeof(X) / sizeof(X)[0])
#define MAX(A, B)     ((A) > (B) ? (A) : (B))
#define MIN(A, B)     ((A) < (B) ? (A) : (B))

void die(const char *fmt, ...);
void free_mult(size_t cnt, ...);
void push_back(uint64_t **v, uint64_t x, size_t i, size_t *sz);
void *safe_calloc(size_t n, size_t sz);
void *safe_malloc(size_t sz);
void *safe_realloc(void *ptr, size_t sz);

#endif
