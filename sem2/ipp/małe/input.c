/* See LICENSE file for copyright and license details. */
#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "input.h"
#include "util.h"

static int parse_key(size_t maze_size);
static int parse_hex_key(size_t key_len);
static int parse_rnd_key(size_t maze_size);
static size_t parse_nxy(size_t **n, size_t **x, size_t **y);
static size_t up_to_scanf(size_t **x);

void
parse_input(size_t *k, uint64_t **n, uint64_t **x, uint64_t **y)
{
  int errln;
  size_t i, maze_size = 1;

  *k = parse_nxy(n, x, y);
  /* maze cannot be larger than SIZE_MAX */
  for (i = 0; i < *k; i++) {
    if (maze_size > SIZE_MAX / (*n)[i]) {
      free_mult(3, *n, *x, *y);
      die("ERROR %d\n", 1);
    }
    maze_size *= (*n)[i];
  }

  if ((errln = parse_key(maze_size)) != 0) {
    free_mult(4, *n, *x, *y, maze_key);
    die("ERROR %d\n", errln);
  }
}

/* return number of line with error, 0 means no error */
int
parse_key(size_t maze_size)
{
  int bit = 64, c; /* getchar returns int not char */
  size_t key_len = (maze_size >> 6) + ((maze_size & 63ull) != 0);

  maze_key = safe_calloc(key_len, sizeof(uint64_t));
  while (isspace(c = getchar()) && c != '\n');
  switch (c) {
  case '0':
    if (parse_hex_key(key_len) != 0)
      return 4;
    break;
  case 'R':
    if (parse_rnd_key(maze_size) != 0)
      return 4;
    break;
  default:
    return 4;
  }

  if (maze_key[--key_len]) { /* possible wall outside of the maze */
    while (!(maze_key[key_len] & (1ull << --bit)));
    if ((key_len << 6) + bit >= maze_size)
      return 4; /* wall outside of the maze */
  }

  return ((c = getchar()) == EOF ? 0 : 5);
}

/* if error occured return 1, else return 0 */
int
parse_hex_key(size_t key_len)
{
  int c; /* getchar returns int not char */
  size_t idx = 0, sz = 1, i, shift;
  uint8_t *s;

  if (getchar() != 'x' || !isxdigit(c = getchar()))
    return 1;

  while (c == '0')
    c = getchar();
  s = safe_malloc(sizeof(uint8_t));
  while (isxdigit(c)) {
    if (idx >= sz) { /* writing another push back function would be a waste */
      sz <<= 1;
      s = safe_realloc(s, sizeof(uint8_t) * sz);
    }
    s[idx++] = (isdigit(c) ? c - '0' : toupper(c) - 'A' + 10);
    c = getchar();
  }
  while(c != '\n' && c != EOF) {
    if (!isspace(c)) {
      free(s);
      return 1;
    }
    c = getchar();
  }

  /* hexadecimal digits have 4 bits, uint64_t has 64 bits (16 times more) */
  for (i = 0; i < key_len; i++)
    for (shift = 0; idx && shift < 64; shift += 4)
      maze_key[i] |= ((uint64_t)s[--idx]) << shift;
  free(s);

  return (idx > 0); /* if idx > 0, then too many bits in input */
}

/* if error occured return 1, else return 0 */
int
parse_rnd_key(size_t maze_size)
{
  uint64_t *rnd = NULL, i, w, bit;
  size_t sz, div;
  /*  must be exactly 5 elements       m must not be 0 */
  if ((sz = up_to_scanf(&rnd)) != 5 || rnd[2] == 0) {
    free(rnd);
    return 1;
  }
  for (i = 0; i < sz; i++) {
    if (rnd[i] > UINT32_MAX) {
      free(rnd);
      return 1;
    }
  }

  /* rnd : 0 1 2 3 4 */
  /*       a b m r s */ 
  for (i = 1; i <= rnd[3]; i++) {
    rnd[4] = (rnd[0] * rnd[4] + rnd[1]) % rnd[2];
    w = rnd[4] % maze_size;
    
    while (w < maze_size) {
      div = w >> 6;
      bit = 1ull << (w & 63ull);
      if (maze_key[div] & bit)
        break;
      maze_key[div] |= bit;
      if (w > SIZE_MAX - (1ull << 32))
        break;
      w += 1ull << 32;
    }
  }
  free(rnd);

  return 0;
}

size_t
parse_nxy(size_t **n, size_t **x, size_t **y)
{
  size_t sz[3], i, j;
  uint64_t *ptr[3] = { NULL, NULL, NULL };
  int err = 0;

  for (i = 0; i < 3; i++) {
    sz[i] = up_to_scanf(&ptr[i]);
    sz[i] *= (sz[i] == sz[0]); /* if different k then set to 0 */
    for (j = 0; j < sz[i]; j++)
      err |= (!ptr[i][j] || ptr[i][j] > ptr[0][j]);
    if (!sz[i] || err) {
      free_mult(3, ptr[0], ptr[1], ptr[2]);
      die("ERROR %d\n", i + 1);
    }
  }
  *n = ptr[0], *x = ptr[1], *y = ptr[2];
  return sz[0]; /* return k */
}

/* return number of elements in *ptr or 0 on error */
size_t
up_to_scanf(uint64_t **ptr)
{
  int c;
  size_t idx = 0, sz = 1, prev_was_digit = 0;
  uint64_t num = 0;
  *ptr = safe_malloc(sizeof(size_t));
  while ((c = getchar()) != '\n' && c != EOF) {
    if (isdigit(c)) {
      /* overflow detection */
      if (num > SIZE_MAX / 10 || num * 10 > SIZE_MAX - (c - '0'))
        return 0;
      num = num * 10 + (c - '0');
      prev_was_digit = 1;
    } else {
      if (!isspace(c))
        return 0;
      if (prev_was_digit) {
        push_back(ptr, num, idx++, &sz);
        num = 0;
      }
      prev_was_digit = 0;
    }
  }
  if (prev_was_digit)
    push_back(ptr, num, idx++, &sz);
  return idx;
}

