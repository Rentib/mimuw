/** \file
 * Interfejs definiujący kilka funkcji do manipulacji numerami.
 *
 * \author Stanisław Bitner <sb438247@students.mimuw.edu.pl>
 * \copyright Informacje o licencji i prawach autorskich są w pliku LICENSE.
 * \date 2022
 */

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stddef.h>

/** Stała wartość oznaczająca liczbę różnych cyfr. */
#define NUM_DIGITS 12

/** \brief Przekształca znak w indeks.
 * Przekształca dany znak w odpowiedni indeks.
 * \param[in] c - znak do przekształcenia.
 * \return Wartość \p c - '0' dla znaków od '0' do '9', 10 dla '*', 11 dla '#'.
           W innym przypadku zwraca -1.
 */
int chartoidx(int c);

/** \brief Sprawdza czy znak jest cyfrą numeru.
 * Sprawdza czy znak to cyfra lub * lub #.
 * \param[in] c - znak, który ma być sprawdzony.
 * \return Wartość niezerową jeśli znak jest poprawną cyfrą numeru, zero w
 *         przeciwnym przypadku.
 */
int isnumdigit(int c);

/** \brief Konkatenuje dwa numery.
 * Zwraca nowy numer powstały w wyniku konkatenacji numeru \p num1 z numerem
 * \p num2.
 * \param[in] num1 - wskaźnik na pierwszy numer;
 * \param[in] num2 - wskaźnik na drugi    numer;
 * \param[in] len  - łączna długość obu numerów.
 * \return Wskaźnik na nowy numer lub NULL, gdy pojawił się błąd.
 */
char *numcat(const char *num1, const char *num2, size_t len);

/** \brief Porównuje numery.
 * Zaczyna porównując pierwszy znak obu napisów. Jeśli znaki te są sobie równe,
 * to kontynuuje porównywanie dopóki nie zostanie osiągnięty znak o wartości 0.
 * \param[in] num1 - wskaźnik na napis reprezentujący pierwszy numer;
 * \param[in] num2 - wskaźnik na napis reprezentujący drugi numer.
 * \return Wartość całkowita wskazująca związek między numerami.
 *         
 */
int numcmp(const char *num1, const char *num2);

/** \brief Oblicza długość numeru telefonu.
 * Oblicza długość numeru telefonu reprezentowanego przez \p num.
 * \param[in] num - wskaźnik na napis reprezentujący numer telefonu.
 * \return Długość danego napisu lub 0, gdy numer jest niepoprawny.
 */
size_t numlen(const char *num);

#endif /* __NUMBER_H__ */
