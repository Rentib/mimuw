/** \file
 * Interfejs klasy przechowującej cykliczną listę.
 * 
 * \author Stanisław Bitner <sb438247@students.mimuw.edu.pl>
 * \copyright Informacje o licencji i prawach autorskich są w pliku LICENSE.
 * \date 2022
 */ 

#ifndef __CIRCULAR_LIST_H__
#define __CIRCULAR_LIST_H__

#include <stdbool.h>
#include <stdlib.h>

#include "vector.h"

/** \struct CList
 * To jest struktura przechowująca elementy listy cyklicznej.
 */
struct CList;

/** \typedef CList
 *  Definiuje strukturę CList.
 */
typedef struct CList CList;

/** \brief Tworzy nową listę cykliczną.
 *  Tworzy nową jednoelementową listę cykliczną zawierającą jedynie wartownika.
 *  \return Wskaźnik na utworzoną strukturę lub NULL, gdy nie udało się 
 *         alokować pamięci.
 */
CList *clstNew(void);

/** \brief Usuwa listę cykliczną.
 *  Usuwa wszystkie elementy listy cyklicznej, której jednym z elementów jest
 *  \p l. Nic nie robi, jeśli wskaźnik ten ma wartość NULL.
 *  \param[in] l - wskaźnik na dowolny z elementów usuwanej listy.
 */
void clstDelete(CList *l);

/** \brief Usuwa element listy cyklicznej.
 *  Usuwa strukturę wskazywaną przez \p l i zacykla listę. Nic nie robi, jeśli
 *  wskaźnik ten ma wartość NULL.
 *  \param[in] l - wskaźnik na strukturę przechowującą element listy cyklicznej.
 */
void clstKys(CList *l);

/** \brief Dodaje element do listy cyklicznej.
 *  Dodaje nowy element do listy cyklicznej, której jednym z elementów jest
 *  \p l i ustawia jego wartość na napis \p str o długości \p len. Nic nie robi,
 *  jeśli wskaźnik \p l ma wartość NULL.
 *  \param[in,out] l - wskaźnik na dowolny z elementów zmienianej listy;
 *  \param[in] str   - wskaźnik na napis mający zostać przypisany wartości 
 *                     nowego elementu;
 *  \param[in] len   - wartość reprezentująca długość napisu \p str.
 *  \return Wskaźnik na utworzoną strukturę lub NULL, gdy nie udało się
 *          alokować pamięci, bądź wskaźnik \p ma wartość NULL.
 */
CList *clstAdd(CList *l, const char *str, size_t len);

/** \brief Zwraca wartość elementu listy cyklicznej.
 *  Zwraca wskaźnik na napis zawarty w strukturze \p l.
 *  \param[in] l - wskaźnik na strukturę przechowującą element listy;
 *  \return Wskaźnik na napis przechowywany w strukturze wskazywanej przez \p l
 *          lub NULL, gdy \p l ma wartość NULL.
 */
const char *clstGet(CList *l);

/** \brief Dodaje wszystkie wartości listy do vectora.
 * Przechodzi po wszystkich elementach listy i dodaje do Vectora \p v kopie ich
 * wartości połączonych z napisem \p suf. Nie dodaje wartości wartownika.
 * \param[in] start - wskaźnik na strukturę przechowywującą wartownika listy;
 * \param[in,out] v - wskaźnik na strukturę przechowywującą Vector;
 * \param[in] suf   - wskaźnik na napis reprezentujący sufiks pewnego napisu;
 * \param[in] len   - długość napisu \p suf.
 * \return Wartość \p true, jeśli pomyślnie udało się dodać wszystkie wartości,
 *         wartość \p false, gdy nie udało się alokować pamięci.
 */
bool clstCollect(CList *start, Vector *v, const char *suf, size_t len);

#endif /* __CIRCULAR_LIST_H__ */
