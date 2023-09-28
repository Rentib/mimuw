#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

// Ten plik zawiera przykład użycia funkcji:
bool inverse_permutation(size_t n, int *p);

#define SIZE(x) (sizeof x / sizeof x[0])

// Sprawdza, czy permutacje p1 i p2 o długości n są identyczne.
static bool compare_permutations(size_t n, int const *p1, int const *p2) {
  for (size_t i = 0; i < n; ++i)
    if (p1[i] != p2[i])
      return false;
  return true;
}

#include <stdio.h>

// Sprawdza, czy permutacja p2 jest permutacją odwrotną do permutacji p1 o długości n.
static bool check_inverse_permutation(size_t n, int const *p1, int const *p2) {
  for (size_t i = 0; i < n; ++i)
    if ((size_t)p2[p1[i]] != i)
      return false;
  return true;
}

// To są testowe ciągi liczb.
static int seq_a[] = {0};
static int seq_b[] = {0, -1};
static int seq_c[] = {0, 1, 3};
static int seq_d[] = {0, 1, 2, 2};
static int seq_e[] = {1, 2, 3, 4, 0};
static int seq_f[] = {3, 1, 4, 5, 2, 0};
static int seq_g[] = {6, 5, 4, 3, 2, 1, 0};
static int seq_h[] = {1, 2, 3, 4, 5, 6, 7, 0};

// Tablica, w której umieszczamy testowany ciąg liczb i której adres dostaje
// funkcja inverse_permutation. Możemy chcieć odwracać długie permutacje.
// static int work_space[(size_t)INT_MAX + 1];
static int work_space[8];

#define CHECK_SIZE(N, P)                                  \
  do {                                                    \
    memcpy(work_space, P, sizeof P);                      \
    assert(!inverse_permutation(N, work_space));          \
    assert(compare_permutations(SIZE(P), P, work_space)); \
  } while (0)

#define CHECK_FALSE(P) CHECK_SIZE(SIZE(P), P)

#define CHECK_TRUE(P)                                          \
  do {                                                         \
    memcpy(work_space, P, sizeof P);                           \
    assert(inverse_permutation(SIZE(P), work_space));          \
    assert(check_inverse_permutation(SIZE(P), P, work_space)); \
  } while (0)

int main() {
  int *p = NULL;
  assert(!inverse_permutation(0, p));
  assert(!inverse_permutation((size_t)INT_MAX + 2, p));
  assert(!inverse_permutation(INT_MAX + 2, p));
  assert(!inverse_permutation(LLONG_MAX, p));
  assert(!inverse_permutation(ULLONG_MAX, p));

  CHECK_SIZE(0, seq_a);
  CHECK_SIZE((size_t)INT_MAX + 2, seq_a);
  CHECK_SIZE((size_t)-1, seq_a);

  CHECK_FALSE(seq_b);
  CHECK_FALSE(seq_c);
  CHECK_FALSE(seq_d);

  CHECK_TRUE(seq_a);
  CHECK_TRUE(seq_e);
  CHECK_TRUE(seq_f);
  CHECK_TRUE(seq_g);
  CHECK_TRUE(seq_h);
}
