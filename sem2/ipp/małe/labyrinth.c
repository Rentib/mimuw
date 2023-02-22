/* See LICENSE file for copyright and license details. */
#include <inttypes.h>
#include <stdio.h>

#include "input.h"
#include "stack.h"
#include "util.h"

/* function declarations */
static uint64_t get_cube(uint64_t *crd, uint64_t *n, size_t k);
static inline uint64_t get_bit(size_t idx);
static inline void set_bit(size_t idx);
static void calc_pref(uint64_t **pref, uint64_t *n, size_t k);
static void bfs(uint64_t start, uint64_t end, uint64_t *pref, size_t k);

/* globals */
uint64_t *maze_key; /* number that defines walls and visited nodes */

/* function implementations */
uint64_t 
get_cube(uint64_t *crd, uint64_t *pref, size_t k)
{
  size_t i;
  uint64_t res = 0;
  for (i = 0; i < k; i++)
    res += (crd[i] - 1) * pref[i];
  return res;
}

uint64_t
get_bit(size_t idx)
{
  uint64_t bit = 1ull << (idx & 63ull);
  return maze_key[idx >> 6] & bit;
}

void 
set_bit(size_t idx)
{
  uint64_t bit = 1ull << (idx & 63ull);
  maze_key[idx >> 6] |= bit;
}

void 
calc_pref(uint64_t **pref, uint64_t *n, size_t k)
{
  size_t i;
  *pref = safe_malloc((k + 1) * sizeof(uint64_t));
  (*pref)[0] = 1ull;
  for (i = 0; i < k; i++) 
    (*pref)[i + 1] = (*pref)[i] * n[i];
}

void
bfs(uint64_t start, uint64_t end, uint64_t *pref, size_t k)
{
  int cur = 0, nxt = 1;
  size_t dist = 0, i, j;
  uint64_t v, u;
  Stack s[2] = { stack_create(), stack_create() };
  
  if (start == end) {
    set_bit(end);
  } else {
    stack_push(&s[cur], start);
    set_bit(start);

    while (!stack_empty(&s[cur])) {
      dist++;

      while (!stack_empty(&s[cur])) {
        v = stack_top(&s[cur]);
        stack_pop(&s[cur]);

        for (i = 0; i < k; i++) {
          for (j = 0; j < 2; j++) { /* subtract when j = 0, add when j = 1 */
            if (j ? v >= pref[k] - pref[i] : v < pref[i]) /* overflow */
              continue;
            u = v + (j ? pref[i] : -pref[i]);
            if (!get_bit(u) && v / pref[i + 1] == u / pref[i + 1]) {
              set_bit(u);
              if (u == end)
                goto solution_found;
              stack_push(&s[nxt], u);
            }
          }
        }
      }

      cur ^= 1; /* 0 -> 1, 1 -> 0 */
      nxt ^= 1;
    }
  }

  solution_found:
  stack_destroy(&s[0]);
  stack_destroy(&s[1]);

  (get_bit(end) ? printf("%lu\n", dist) : printf("NO WAY\n"));
}
  
int
main(void)
{
  size_t k;
  uint64_t *n, *x, *y, *pref;
  size_t start, end;

  parse_input(&k, &n, &x, &y);
  calc_pref(&pref, n, k);

  start = get_cube(x, pref, k);
  end   = get_cube(y, pref, k);

  free_mult(3, n, x, y);
  
  /* correct spawn point, incorrect wall generation */
  if (get_bit(start) || get_bit(end)) {
    free_mult(2, maze_key, pref);
    die("ERROR %d\n", 4);
  }

  bfs(start, end, pref, k);
  free_mult(2, maze_key, pref);

  return 0;
}

