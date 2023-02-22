/** \file
 * Implementacja klasy przechowującej drzewo trie.
 * 
 * \author Stanisław Bitner <sb438247@students.mimuw.edu.pl>
 * \copyright Informacje o licencji i prawach autorskich są w pliku LICENSE.
 * \date 2022
 */ 

#include <stdbool.h>
#include <stdlib.h>

#include "circular_list.h"
#include "number.h"
#include "trie.h"
#include "vector.h"

/** \typedef FwdData
 * Definiuje strukturę FwdData.
 * \struct FwdData
 * To jest struktura przechowująca dane wierzchołków typu FORWARD.
 */
typedef struct FwdData {
  /**\{*/
  char *value; /**< Wartość przekierowania prefiksu reprezentowanego przez
                    wierzchołek; */
  CList *element; /**< Wskaźnik na element listy cklicznej wierzchołka revNode,
                       który powinien być usunięty, gdy usuniemy dany
                       wierzchołek. */
  TrieNode *revNode; /**< Wskaźnik na wierzchołek reprezentujący napis \p value
                          w drzewie odwrotności. */
  /**\}*/
} FwdData;

/** \typedef RevData
 * Definiuje strukturę RevData.
 * \struct RevData
 * To jest struktura przechowująca dane wierzchołków typu REVERSE.
 */
typedef struct RevData {
  /**\{*/
  CList *list;   /**< Wskaźnik na wartownika cyklicznej listy przechowującej
                      odwrotności przekierowań; */
  size_t stored; /**< Wartość określająca liczbę odwrotności przekierowań w
                      cyklicznej liście. */
  /**\}*/
} RevData;

struct TrieNode {
  /**\{*/
  TrieNode *father; /**< Wskaźnik na ojca wierzchołka; */
  TrieNode *children[NUM_DIGITS]; /**< Statyczna tablica zawierająca wskaźniki
                                       na synów wierzchołka; */
  int digit; /**< Wartość oznaczająca którym dzieckiem ojca jest wierzchołek.
                  Jest to także wartość reprezentowana przez krawędź między 
                  wierzchołkiem i jego ojcem; */
  union {
    FwdData fd; /**< Struktura przechowująca dane wierzchołka typu FORWARD; */
    RevData rd; /**< Struktura przechowująca dane wierzchołka typu REVERSE; */
  };
  /**\}*/
};

/** \brief Sprawdza czy wierzchołek jest potrzebny.
 * Sprawdza czy wierzchołek \p v nie zawiera przekierowania i czy jest liściem.
 * \param[in] v    - wskaźnik na strukturę przechowującą sprawdzany wierzchołek.
 * \param[in] flag - flaga określająca rodzaj wierzchołka.
 * \return Wartość \p true jeśli \p v jest potrzebny, wartość \p false w
 *         przeciwnym razie.
 */
static bool isNeeded(TrieNode *v, NodeFlag flag);

/** \brief Sprawdza liczbę dzieci wierzchołka.
 * Sprawdza, czy wierzchołek \p v ma co najmniej \p n synów (NULL to nie syn).
 * \param[in] v - wskaźnik na strukturę przechowującą sprawdzany wierzchołek.
 * \param[in] n - wartość reprezentująca liczbę synów.
 * \return Wartość \p true jeśli \p v ma co najmniej \p n synów, wartość
 *         \p false w przeciwnym razie.
 */
static bool hasNChildren(TrieNode *v, size_t n);

/** \brief Usuwa niepotrzebne wierzchołki.
 * Jeśli wierzchołek \p v jest niepotrzebny w sensie funkcji isNeeded(), to
 * usuwa go i powtarza czynność dla jego ojca. Nie usuwa korzenia drzewa trie.
 * Nic nie robi jeśli \p v ma wartość NULL.
 * \param[in] v    - wskaźnik na sprawdzany wierzchołek;
 * \param[in] flag - flaga określająca rodzaj wierzchołka.
 */
static void triePrun(TrieNode *v, NodeFlag flag);

/** \brief Tworzy nowy wierzchołek.
 * Tworzy nowy wierzchołek niezawierający przekierowania, ustawia jego ojca na
 * \p father i cyfrę na \p digit.
 * \param[in] father - wskaźnik na strukturę reprezentującą ojca nowego
 *                     wierzchołka;
 * \param[in] digit  - wartość oznaczająca, którym synem swojego ojca jest nowo
 *                     tworzony wierzchołek;
 * \param[in] flag   - flaga określająca rodzaj wierzchołka.
 * \return Wskaźnik na utworzoną strukturę lub NULL, gdy nie udało się
 *         alokować pamięci.
 */
static TrieNode *trndNew(TrieNode *father, int digit, NodeFlag flag);

/** \brief Usuwa dane wierzchołka typu FORWARD.
 * Zwalnia pamięć potrzebną do przechowywania danych wierzchołka typu forward
 * wskazywanego przez \p v.
 * \param[in,out] v - wskaźnik na strukturę reprezentującą wierzchołek.
 */
static void trndFreeFwdData(TrieNode *v);

/** \brief Usuwa wierzchołek.
 * Usuwa strukturę wskazywaną przez \p v. Nic nie robi, jeśli wskaźnik ten ma
 * wartość NULL.
 * \param[in] v    - wskaźnik na usuwany wierzchołek;
 * \param[in] flag - flaga określająca rodzaj wierzchołka.
 */
static void trndDelete(TrieNode *v, NodeFlag flag);

Vector *
trieCollect(TrieNode *root, const char *str)
{
  size_t len;
  Vector *vec;
  char *num;
  if (!(len = numlen(str)) || !(vec = vecNew()))
    return NULL;
  if (!vecAdd(vec, num = numcat(str, "", len))) {
    free(num);
    vecDelete(vec);
    return NULL;
  }

  while (root && *str) {
    len--;
    root = root->children[chartoidx(*str++)];
    if (!root || !root->rd.stored)
      continue;
    if (!clstCollect(root->rd.list, vec, str, len)) {
      vecDelete(vec);
      return NULL;
    }
  }

  vecShrinkToFit(vec);
  return vec;
}

TrieNode *
trieFind(TrieNode *root, const char *str, size_t *idx)
{
  TrieNode *v = root, *res = NULL;
  size_t i = 0;
  while (v && str[i]) {
    if (v->fd.value) {
      res = v;
      *idx = i;
    }
    v = v->children[chartoidx(str[i++])];
  }
  if (v && v->fd.value) {
    res = v;
    *idx = i;
  }
  return res;
}

TrieNode *
trieInsert(TrieNode **root, const char *str, NodeFlag flag)
{
  TrieNode *v;
  if (!*root)
    *root = trndNew(NULL, -1, flag); /* korzeń nie ma ojca */
  v = *root;
  while (v && *str) {
    if (!v->children[chartoidx(*str)])
      v->children[chartoidx(*str)] = trndNew(v, chartoidx(*str), flag);
    v = v->children[chartoidx(*str++)];
  }
  return v;
}

void
triePrun(TrieNode *v, NodeFlag flag)
{
  int digit;
  if (!v)
    return;
  while (v->father && !isNeeded(v, flag)) {
    digit = v->digit;
    v = v->father;
    trndDelete(v->children[digit], flag);
    v->children[digit] = NULL;
  }
}

void
trieRemove(TrieNode **root, const char *str)
{
  TrieNode *v = *root;
  while (v && *str)
    v = v->children[chartoidx(*str++)];
  if (v) { /* przedstawiciel napisu str został znaleziony */
    if (v == *root) {
      trieDelete(v, FORWARD);
      *root = NULL;
    } else {
      v = v->father;
      trieDelete(v->children[chartoidx(*--str)], FORWARD);
      v->children[chartoidx(*str)] = NULL;
      triePrun(v, FORWARD);
    }
  }
}

void
trieDelete(TrieNode *root, NodeFlag flag)
{
  TrieNode *v, *u;
  size_t digit = 0;
  if (!root)
    return;
  root->father = NULL;
  v = root;
  while (v) {
    if (digit++ < NUM_DIGITS) {
      if (v->children[digit - 1]) {
        v = v->children[digit - 1];
        digit = 0;
      }
    } else {
      u = v->father;
      digit = v->digit + 1; /* v->digit >= -1 ==> digit >= 0 */
      trndDelete(v, flag);
      v = u;
    }
  }
}

TrieNode *
trndNew(TrieNode *father, int digit, NodeFlag flag)
{
  size_t i;
  TrieNode *v;
  if (!(v = malloc(sizeof(TrieNode))))
    return NULL;
  v->father = father;
  for (i = 0; i < NUM_DIGITS; i++)
    v->children[i] = NULL;
  v->digit = digit;
  switch (flag) {
  case FORWARD:
    v->fd.value   = NULL;
    v->fd.element = NULL;
    v->fd.revNode = NULL;
    break;
  case REVERSE:
    if (!(v->rd.list = clstNew())) {
      free(v);
      return NULL;
    }
    v->rd.stored = 0;
    break;
  }
  return v;
}

void
trndFreeFwdData(TrieNode *v)
{
  free(v->fd.value);
  if (v->fd.revNode) {
    v->fd.revNode->rd.stored--;
    clstKys(v->fd.element);
    triePrun(v->fd.revNode, REVERSE);
    v->fd.revNode = NULL;
  }
}

void
trndDelete(TrieNode *v, NodeFlag flag)
{
  if (v) {
    switch (flag) {
    case FORWARD:
      trndFreeFwdData(v);
      break;
    case REVERSE:
      clstDelete(v->rd.list);
      break;
    }
    free(v);
  }
}

bool
trndSet(TrieNode *v, const char *str, size_t len, TrieNode *u, NodeFlag flag)
{
  CList *l;
  if (!v)
    return false;
  switch (flag) {
  case FORWARD:
    u->rd.stored++;
    trndFreeFwdData(v);
    if (!(v->fd.value = numcat(str, "", len)))
      return false;
    break;
  case REVERSE:
    if (!(l = clstAdd(v->rd.list, str, len)))
      return false;
    u->fd.element = l; /* element, który ma być usunięty wraz z u */
    u->fd.revNode = v;
    break;
  }
  return true;
}

const char *
trndGet(TrieNode *v)
{
  return v ? v->fd.value : NULL;
}

bool
isNeeded(TrieNode *v, NodeFlag flag)
{
  switch (flag) {
  case FORWARD: return v->fd.value  || hasNChildren(v, 1);
  case REVERSE: return v->rd.stored || hasNChildren(v, 1);
  default: return false;
  }
}

bool
hasNChildren(TrieNode *v, size_t n)
{
  size_t i;
  for (i = 0; n && i < NUM_DIGITS; i++)
    n -= (v->children[i] != NULL);
  return !n;
}
