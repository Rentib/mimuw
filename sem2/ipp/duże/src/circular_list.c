/** \file
 * Implementacja klasy przechowującej cykliczną listę.
 * 
 * \author Stanisław Bitner <sb438247@students.mimuw.edu.pl>
 * \copyright Informacje o licencji i prawach autorskich są w pliku LICENSE.
 * \date 2022
 */ 

#include <stdbool.h>
#include <stdlib.h>

#include "circular_list.h"
#include "number.h"
#include "vector.h"

struct CList {
  /**\{*/
  CList *prev; /**< Wskaźnik na poprzedni element listy cyklicznej; */
  CList *next; /**< Wskaźnik na następny  element listy cyklicznej; */
  char  *str;  /**< Wskaźnik na przechowywany napis. */
  /**\}*/
};

CList *
clstNew(void)
{
  CList *l;
  if (!(l = malloc(sizeof(CList))))
    return NULL;
  l->prev = l;
  l->next = l;
  l->str  = NULL;
  return l;
}

void
clstDelete(CList *l)
{
  if (l) {
    while (l->next != l)
      clstKys(l->next);
    free(l->str);
    free(l);
  }
}

void
clstKys(CList *l)
{
  if (l && l->next != l) {
    l->prev->next = l->next;
    l->next->prev = l->prev;
    free(l->str);
    free(l);
  }
}

CList *
clstAdd(CList *l, const char *str, size_t len)
{
  CList *nl;
  if (!l || !(nl = malloc(sizeof(CList))))
    return NULL;
  if (!(nl->str = numcat(str, "", len))) {
    free(nl);
    return NULL;
  }
  nl->prev = l;
  nl->next = l->next;
  l->next->prev = nl;
  l->next = nl;
  return nl;
}

const char *
clstGet(CList *l)
{
  return l ? l->str : NULL;
}

bool
clstCollect(CList *start, Vector *v, const char *suf, size_t len)
{
  CList *l = start->next;
  char *num;
  while (l != start) {
    if (!vecAdd(v, num = numcat(l->str, suf, numlen(l->str) + len))) {
      free(num);
      return false;
    }
    l = l->next;
  }
  return true;
}
