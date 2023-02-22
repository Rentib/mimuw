/** \file
 * Implementacja klasy przechowującej przekierowania numerów telefonów.
 * 
 * \author Stanisław Bitner <sb438247@students.mimuw.edu.pl>
 * \copyright Informacje o licencji i prawach autorskich są w pliku LICENSE.
 * \date 2022
 */ 

#include <stdbool.h>
#include <stdlib.h>

#include "number.h"
#include "phone_forward.h"
#include "trie.h"
#include "vector.h"

struct PhoneForward {
  /**\{*/
  TrieNode *fwdRoot; /**< Wskaźnik na korzeń drzewa trie przechowującego
                          przekierowania numerów telefonów; */
  TrieNode *revRoot; /**< Wskaźnik na korzeń drzewa trie przechowującego
                          odwrotności przekierowań numerów telefonów. */
  /**\}*/
};

struct PhoneNumbers {
  /**\{*/
  Vector *v; /**< Wskaźnik na Vector przechowywujący tablicę numerów
                  telefonów. */
  /**\}*/
};

/** \brief Tworzy nową strukturę.
 * Tworzy nową strukturę przechowującą ciąg numerów telefonów.
 * \return Wskaźnik na utworzoną strukturę lub NULL, gdy nie udało się alokować
 *         pamięci.
 */
static PhoneNumbers *phnumNew(void);

PhoneForward *
phfwdNew(void)
{
  PhoneForward *pf;
  if (!(pf = malloc(sizeof(PhoneForward))))
    return NULL;
  pf->fwdRoot = NULL;
  pf->revRoot = NULL;
  return pf;
}

void
phfwdDelete(PhoneForward *pf)
{
  if (pf != NULL) {
    trieDelete(pf->fwdRoot, FORWARD);
    trieDelete(pf->revRoot, REVERSE);
    free(pf);
  }
}

bool
phfwdAdd(PhoneForward *pf, const char *num1, const char *num2)
{
  TrieNode *v, *u;
  size_t len1, len2;
  if (!pf || !(len1 = numlen(num1)) || !(len2 = numlen(num2))
  || !numcmp(num1, num2)
  || !(v = trieInsert(&pf->fwdRoot, num1, FORWARD))
  || !(u = trieInsert(&pf->revRoot, num2, REVERSE)))
    return false;
  return trndSet(v, num2, len2, u, FORWARD) &&
         trndSet(u, num1, len1, v, REVERSE);
}

void
phfwdRemove(PhoneForward *pf, const char *num)
{
  if (pf && numlen(num))
    trieRemove(&pf->fwdRoot, num);
}

PhoneNumbers *
phfwdGet(const PhoneForward *pf, const char *num)
{
  const char *pref;
  char *str;
  size_t len1, len2, i = 0;
  PhoneNumbers *pnum;
  if (!pf || !(pnum = phnumNew()))
    return NULL;
  if (!(len1 = numlen(num)))
    return pnum;
  pref = trndGet(trieFind(pf->fwdRoot, num, &i));
  len2 = numlen(pref);
  /* numcat nie działa dla NULL, więc musimy to sprawdzić */
  str = numcat((pref ? pref : ""), num + i, len1 + len2 - i);
  if (!vecAdd(pnum->v, str)) {
    free(str);
    phnumDelete(pnum);
    return NULL;
  }
  return pnum;
}

PhoneNumbers *
phfwdGetReverse(const PhoneForward *pf, const char *num)
{
  bool error = false;
  PhoneNumbers *pnum, *rev, *get;
  size_t i;
  if (!pf || !(pnum = phnumNew()))
    return NULL;
  if (!(rev = phfwdReverse(pf, num))) {
    phnumDelete(pnum);
    return NULL;
  }
  for (i = 0; i < vecSize(rev->v); i++) {
    if (!(get = phfwdGet(pf, vecGet(rev->v, i)))) {
      error = true;
      break;
    }
    if (numcmp(num, phnumGet(get, 0)))
      vecRemove(rev->v, i);
    else
      error = !vecAdd(pnum->v, vecGet(rev->v, i));
    phnumDelete(get);
  }
  if (error) {
    phnumDelete(rev);
    vecFreeContainer(pnum->v);
    free(pnum);
    return NULL;
  }
  vecFreeContainer(rev->v);
  free(rev);
  return pnum;
}

PhoneNumbers *
phfwdReverse(const PhoneForward *pf, const char *num)
{
  bool error = false;
  const char *prev = NULL;
  PhoneNumbers *pnum = NULL;
  size_t i;
  Vector *v = NULL;
  if (!pf || !(pnum = phnumNew()))
    return NULL;
  if (!numlen(num))
    return pnum;
  error = !(v = trieCollect(pf->revRoot, num));
  if (!error) {
    vecSort(v);
    error = !vecAdd(pnum->v, vecGet(v, 0));
    prev = vecGet(v, 0);
  }
  for (i = 1; !error && i < vecSize(v); i++) {
    if (!numcmp(prev, vecGet(v, i))) {
      vecRemove(v, i);
      continue;
    }
    error = !vecAdd(pnum->v, vecGet(v, i));
    prev = vecGet(v, i);
  }
  if (error) {
    vecDelete(v);
    vecFreeContainer(pnum->v);
    free(pnum);
    return NULL;
  }
  vecFreeContainer(v);
  vecShrinkToFit(pnum->v);
  return pnum;
}

PhoneNumbers *
phnumNew(void)
{
  PhoneNumbers *pnum = NULL;
  if (!(pnum = malloc(sizeof(PhoneNumbers)))
  ||  !(pnum->v = vecNew())) {
    phnumDelete(pnum);
    return NULL;
  }
  return pnum;
}

void
phnumDelete(PhoneNumbers *pnum)
{
  if (pnum) {
    vecDelete(pnum->v);
    free(pnum);
  }
}

const char *
phnumGet(const PhoneNumbers *pnum, size_t idx)
{
  return pnum ? vecGet(pnum->v, idx) : NULL;
}
