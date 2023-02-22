/** \file
 * Implementacja klasy przechowującej tablicę napisów.
 * 
 * \author Stanisław Bitner <sb438247@students.mimuw.edu.pl>
 * \copyright Informacje o licencji i prawach autorskich są w pliku LICENSE.
 * \date 2022
 */ 

#include <stdbool.h>
#include <stdlib.h>

#include "number.h"
#include "vector.h"

struct Vector {
  /**\{*/
  char **str; /**< Dynamiczna tablica służąca do przechowywania napisów; */
  size_t cnt; /**< Zaalokowany rozmiar tablicy \p str; */
  size_t idx; /**< Liczba elementów w tablicy \p str. */
  /**\}*/
};

/** \brief Porównuje dwa numery.
 * \see numcmp()
 */
static inline int
pstrcmp( const void* a, const void* b )
{
  return numcmp( *(const char**)a, *(const char**)b );
}


Vector *
vecNew(void)
{
  Vector *v = NULL;
  if (!(v = malloc(sizeof(Vector)))
  || !(v->str = malloc(sizeof(char *)))) {
    free(v);
    return NULL;
  }
  v->str[0] = NULL;
  v->cnt = 1;
  v->idx = 0;
  return v;
}

size_t
vecSize(Vector *v)
{
  return v ? v->idx : 0;
}

bool
vecAdd(Vector *v, char *str)
{
  char **tmp;
  if (!v || !str)
    return false;
  if (v->cnt == v->idx) {
    if (!(tmp = realloc(v->str, (v->cnt << 1) * sizeof(char *))))
      return false;
    v->cnt <<= 1;
    v->str = tmp;
  }
  v->str[v->idx++] = str;
  return true;
}

char *
vecGet(Vector *v, size_t idx)
{
  return v && v->idx > idx ? v->str[idx] : NULL;
}

void
vecRemove(Vector *v, size_t idx)
{
  if (v && idx < v->idx) {
    free(v->str[idx]);
    v->str[idx] = NULL;
  }
}

void
vecSort(Vector *v)
{
  if (v && v->str)
    qsort(v->str, vecSize(v), sizeof(vecGet(v, 0)), pstrcmp);
}

void
vecFreeContainer(Vector *v)
{
  if (v) {
    free(v->str);
    free(v);
  }
}

void
vecShrinkToFit(Vector *v)
{
  if (v)
    v->str = realloc(v->str, (v->cnt = v->idx) * sizeof(char *));
}

void
vecDelete(Vector *v)
{
  if (v) {
    while (v->idx--)
      free(v->str[v->idx]);
    free(v->str);
    free(v);
  }
}
