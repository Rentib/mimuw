/** \file
 * Interfejs klasy przechowującej drzewo trie.
 *
 * \author Stanisław Bitner <sb438247@students.mimuw.edu.pl>
 * \copyright Informacje o licencji i prawach autorskich są w pliku LICENSE.
 * \date 2022
 */

#ifndef __TRIE_H__
#define __TRIE_H__

#include <stdbool.h>
#include <stdlib.h>

#include "vector.h"

/** \enum NodeFlag
 * To jest enum określający typ wierzchołka.
 */
typedef enum {
  FORWARD, /**< Flaga wierzchołka drzewa zawierającego przekierowania; */
  REVERSE, /**< Flaga wierzchołka drzewa zawierającego odwrotności
                przekierowań. */
} NodeFlag;

/** \struct TrieNode
 * To jest strukrura przechowywująca wierzchołki drzewa trie.
 */
struct TrieNode;

/** \typedef TrieNode
 *  Definiuje strukturę TrieNode.
 */
typedef struct TrieNode TrieNode;

/** \brief Zbiera napisy ze ścieżki.
 * Tworzy vector złożony z kandydatów na wynik funkcji phfwdReverse, przechodząc
 * po ścieżce reprezentującej napis \p str zaczynającej się w wierzchołku 
 * \p root.
 * \param[in] root - wskaźnik na strukturę przechowywującą korzeń drzewa trie;
 * \param[in] str  - wskaźnik na napis.
 * \return Wskaźnik na strukturę przechowującą Vector. Zwraca NUll, gdy nie
 *         udało się alokować pamięci.
 */
Vector *trieCollect(TrieNode *root, const char *str);

/** \brief Znajduje maksymalny prefiks w drzewie trie.
 *  Znajduje wierzchołek reprezentujący maksymalny przekierowany prefiks napisu
 *  \p str w drzewie trie ukorzenionym w \p root. Ustawia \p idx na długość tego
 *  prefiksu.
 *  \param[in] root    - wskaźnik na strukturę przechowującą korzeń drzewa trie;
 *  \param[in] str     - wskaźnik na szukany napis;
 *  \param[in,out] idx - wskaźnik na długość najdłuższego prefiksu.
 *  \return Wskaźnik na strukturę przechowującą wierzchołek reprezentujący
 *          maksymalny zmapowany prefiks danego napisu. Zwraca NULL, gdy nie
 *          istnieje żaden taki wierzchołek.
 */
TrieNode *trieFind(TrieNode *root, const char *str, size_t *idx);

/** \brief Dodaje napis do drzewa trie.
 *  Dodaje napis \p str do drzewa trie ukorzenionego w \p root i zwraca
 *  reprezentujący ten napis wierzchołek.
 *  \param[in,out] root - podwójny wskaźnik na strukturę przechowywującą korzeń
 *                        drzewa trie;
 *  \param[in] str      - wskaźnik na dodawany napis.
 *  \param[in] flag     - flaga określająca rodzaj wierzchołka.
 *  \return Wskaźnik na strukturę przechowywującą wierzchołek reprezentujący
 *          podany napis lub NULL, gdy nie udało się alokować pamięci.
 */
TrieNode *trieInsert(TrieNode **root, const char *str,
                     NodeFlag flag);

/** \brief Usuwa napisy z drzewa trie.
 *  Usuwa z drzewa trie ukorzenionego w \p root poddrzewo wierzchołka
 *  reprezentującego dany napis.
 *  \param[in,out] root - podwójny wskaźnik na strukturę przechowywującą korzeń
 *                        drzewa trie;
 *  \param[in] str      - wskaźnik na usuwany napis.
 */
void trieRemove(TrieNode **root, const char *str);

/** \brief Usuwa drzewo trie.
 *  Usuwa drzewo trie ukorzenione w \p root.
 *  \param[in] root - wskaźnik na korzeń drzewa trie.
 *  \param[in] flag - flaga określająca rodzaj wierzchołka.
 */
void trieDelete(TrieNode *root,
                NodeFlag flag);

/** \brief Ustawia wartość wierzchołka.
 *  Ustawia wartość wierzchołka \p v lub dodaje do niej napis \p str o długości
 *  \p len. Ustawia wskaźniki wierzchołka drzewa odwrotnego \p u.
 *  \param[in,out] v - wskaźnik na zmieniany wierzchołek;
 *  \param[in] str   - wskaźnik na napis mający zostać przypisany wartości \p v;
 *  \param[in] len   - wartość reprezentująca długość napisu \p str.
 *  \param[in] u     - wskaźnik na powiązany wierzchołek z przeciwnego dzewa;
 *  \param[in] flag  - flaga określająca rodzaj wierzchołka.
 *  \return Wartość \p true jeśli udało się ustawić wartość wierzchołka,
 *          wartość \p false jeśli wystąpił błąd, np. nie udało się alokować
 *          pamięci.
 */
bool trndSet(TrieNode *v, const char *str, size_t len, TrieNode *u,
             NodeFlag flag);

/** \brief Zwraca wartość wierzchołka.
 *  Zwraca wskaźnik na wartość wierzchołka \p v.
 *  \param[in] v - wskaźnik na strukturę przechowującą wierzchołek.
 *  \return Wskaźnik na napis będący wartością danego wierzchołka lub NULL, gdy
 *          \p v ma wartość NULL.
 */
const char *trndGet(TrieNode *v);

#endif /* __TRIE_H__ */
