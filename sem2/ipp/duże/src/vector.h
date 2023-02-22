/** \file
 * Interfejs klasy przechowującej tablicę napisów.
 * Jest to okrojona wersja vector<string> z C++.
 * 
 * \author Stanisław Bitner <sb438247@students.mimuw.edu.pl>
 * \copyright Informacje o licencji i prawach autorskich są w pliku LICENSE.
 * \date 2022
 */ 

#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <stdbool.h>
#include <stdlib.h>

/** \struct Vector
 * To jest struktura przechowująca napisy.
 */
struct Vector;

/** \typedef Vector
 * Definiuje strukturę Vector.
 */
typedef struct Vector Vector;

/** \brief Tworzy nowy Vector.
 * Tworzy nowy jednoelementowy Vector.
 * \return Wskaźnik na utworzoną strukturę lub NULL, gdy nie udało się 
 *        alokować pamięci.
 */
Vector *vecNew(void);

/** \brief Podaje rozmiar Vectora.
 * Podaje liczbę elementów już znajdujących się w Vectorze wskazywanym przez
 * \p v.
 * \param[in] v - wskaźnik na strukturę przechowywującą Vector.
 * \return Rozmiar Vectora lub 0, jeśli \p v ma wartość NULL.
 */
size_t vecSize(Vector *v);

/** \brief Dodaje wartość do Vectora.
 * Dodaje do Vectora wskazywanego przez \p v napis reprezentowany przez \p str.
 * \param[in] v   - wskaźnik na strukturę przechowującą Vector;
 * \param[in] str - wskaźnik na dodawany napis.
 * \return Wartość \p true, gdy udało się dodać nowy napis, wartość \p false
 *         w przeciwnym przypadku.
 */
bool vecAdd(Vector *v, char *str);

/** \brief Udostępnia wartość.
 * Udostępnia wskaźnik na wartość Vectora \p v. Wartości są indeksowane kolejno
 * od zera.
 * \param[in] v   - wskaźnik na Vector;
 * \param[in] idx - indeks wartości.
 * \return Wskaźnik na wartość. Wartość NULL, jeśli wskaźnik \p v ma wartość
 *         NULL lub indeks ma za dużą wartość.
 */
char *vecGet(Vector *v, size_t idx);

/** \brief Zwalnia wartość.
 * Zwalnia wartość Vectora wskazywanego przez \p v o indeksie \p idx i ustawia
 * ją na NULL. Nic nie robi jeśli wskaźnik \p v ma wartość NULL lub indeks ma za
 * dużą wartość.
 * \param[in,out] v - wskaźnik na Vector;
 * \param[in] idx   - indeks wartości.
 */
void vecRemove(Vector *v, size_t idx);

/** \brief Sortuje wartości Vectora.
 * Używa funkcji qsort, aby posortować napisy znajdujące się w Vectorze \p v.
 * \param[in,out] v - wskaźnik na Vector.
 */
void vecSort(Vector *v);

/** \brief Zwalnia tablicę wartości.
 * Zwalnia Vector i tablicę, w której trzymane są jego wartości bez zwalniania
 * tych wartości. Nic nie robi jeśli wskaźnik \p v ma wartość NULL.
 * \param[in] v - wskaźnik na Vector.
 */
void vecFreeContainer(Vector *v);

/** \brief Zmniejsza zużycie pamięci poprzez opróżnienie nieużywanego miejsca.
 * \param[in] v - wskaźnik na strukturę przechowującą Vector.
 */
void vecShrinkToFit(Vector *v);

/** \brief Usuwa Vector.
 * Usuwa wszystkie napisy z Vectora. Nic nie robi, jeśli wskaźnik \p ma wartość
 * NULL.
 * \param[in] v - wskaźnik na strukturę przechowującą Vector.
 */
void vecDelete(Vector *v);

#endif /* __VECTOR_H__ */
