/** \file
 * Interfejs klasy przechowującej przekierowania numerów telefonicznych
 *
 * \authors Marcin Peczarski <marpe@mimuw.edu.pl>, 
 *          Stanisław Bitner <sb438247@students.mimuw.edu.pl>
 * \copyright Uniwersytet Warszawski
 * \date 2022
 */

#ifndef __PHONE_FORWARD_H__
#define __PHONE_FORWARD_H__

#include <stdbool.h>
#include <stddef.h>

/** \struct PhoneForward
 * To jest struktura przechowująca przekierowania numerów telefonów.
 */
struct PhoneForward;

/** \typedef PhoneForward
 * Definiuje strukturę PhoneForward.
 */
typedef struct PhoneForward PhoneForward;

/** \struct PhoneNumbers
 * To jest struktura przechowująca ciąg numerów telefonów.
 */
struct PhoneNumbers;

/** \typedef PhoneNumbers
 * Definiuje strukturę PhoneNumbers.
 */
typedef struct PhoneNumbers PhoneNumbers;

/** \brief Tworzy nową strukturę.
 * Tworzy nową strukturę niezawierającą żadnych przekierowań.
 * \return Wskaźnik na utworzoną strukturę lub NULL, gdy nie udało się
 *         alokować pamięci.
 */
PhoneForward *phfwdNew(void);

/** \brief Usuwa strukturę.
 * Usuwa strukturę wskazywaną przez \p pf. Nic nie robi, jeśli wskaźnik ten ma
 * wartość NULL.
 * \param[in] pf – wskaźnik na usuwaną strukturę.
 */
void phfwdDelete(PhoneForward *pf);

/** \brief Dodaje przekierowanie.
 * Dodaje przekierowanie wszystkich numerów mających prefiks \p num1, na numery,
 * w których ten prefiks zamieniono odpowiednio na prefiks \p num2. Każdy numer
 * jest swoim własnym prefiksem. Jeśli wcześniej zostało dodane przekierowanie
 * z takim samym parametrem \p num1, to jest ono zastępowane.
 * Relacja przekierowania numerów nie jest przechodnia.
 * \param[in,out] pf – wskaźnik na strukturę przechowującą przekierowania
 *                     numerów;
 * \param[in] num1   – wskaźnik na napis reprezentujący prefiks numerów
 *                     przekierowywanych;
 * \param[in] num2   – wskaźnik na napis reprezentujący prefiks numerów,
 *                     na które jest wykonywane przekierowanie.
 * \return Wartość \p true, jeśli przekierowanie zostało dodane.
 *         Wartość \p false, jeśli wystąpił błąd, np. podany napis nie
 *         reprezentuje numeru, oba podane numery są identyczne lub nie udało
 *         się alokować pamięci.
 */
bool phfwdAdd(PhoneForward *pf, const char *num1, const char *num2);

/** \brief Usuwa przekierowania.
 * Usuwa wszystkie przekierowania, w których parametr \p num jest prefiksem
 * parametru \p num1 użytego przy dodawaniu. Jeśli nie ma takich przekierowań
 * lub napis nie reprezentuje numeru, nic nie robi.
 * \param[in,out] pf – wskaźnik na strukturę przechowującą przekierowania
 *                     numerów;
 * \param[in] num    – wskaźnik na napis reprezentujący prefiks numerów.
 */
void phfwdRemove(PhoneForward *pf, const char *num);

/** \brief Wyznacza przekierowanie numeru.
 * Wyznacza przekierowanie podanego numeru. Szuka najdłuższego pasującego
 * prefiksu. Wynikiem jest ciąg zawierający co najwyżej jeden numer. Jeśli dany
 * numer nie został przekierowany, to wynikiem jest ciąg zawierający ten numer.
 * Jeśli podany napis nie reprezentuje numeru, wynikiem jest pusty ciąg.
 * Alokuje strukturę \p PhoneNumbers, która musi być zwolniona za pomocą
 * funkcji \ref phnumDelete.
 * \param[in] pf  – wskaźnik na strukturę przechowującą przekierowania numerów;
 * \param[in] num – wskaźnik na napis reprezentujący numer.
 * \return Wskaźnik na strukturę przechowującą ciąg numerów lub NULL, gdy nie
 *         udało się alokować pamięci.
 */
PhoneNumbers *phfwdGet(const PhoneForward *pf, const char *num);

/** \brief Wyznacza przekierowania na dany numer.
 * Wyznacza następujący ciąg numerów:
 * jeśli istnieje numer \p x, taki że wynik wywołania phfwdGet() z numerem \p x
 * zawiera numer \p num, to numer \p x należy do wyniku wywołania
 * \ref phfwdReverse() z numerem \p num. Wynikowe numery są posortowane
 * leksykograficznie i nie mogą się powtarzać. Jeśli podany napis nie
 * reprezentuje numeru, wynikiem jest pusty ciąg.
 * \param[in] pf  – wskaźnik na strukturę przechowującą przekierowania numerów;
 * \param[in] num – wskaźnik na napis reprezentujący numer.
 * \return Wskaźnik na strukturę przechowującą ciąg numerów lub NULL, gdy nie
 *         udało się alokować pamięci.
 */
PhoneNumbers *phfwdGetReverse(const PhoneForward *pf, const char *num);

/** \brief Wyznacza pseudo-przekierowania na dany numer.
 * Wyznacza następujący ciąg numerów:
 * dla każdego numeru \p p będącego prefiksem numeru \p num do wyniku zostaje
 * dodany każdy numer \p x zadany poprzez konkatenacje numerów \p a i \p b,
 * takich że istnieje przekierowanie z \p a do \p p oraz \p b jest maksymalnym
 * sufiksem numeru \p num rozłącznym z numerem \p p. Dodatkowo ciąg wynikowy
 * zawsze zawiera numer \p num. Wynikowe numery są posortowane
 * leksykograficznie i nie mogą się powtarzać, przy czym należy pamiętać, że
 * np. '*' nie reprezentuje swojego znaku ASCII lecz liczbę 10.
 * \param[in] pf  – wskaźnik na strukturę przechowującą przekierowania numerów;
 * \param[in] num – wskaźnik na napis reprezentujący numer.
 * \return Wskaźnik na strukturę przechowującą ciąg numerów lub NULL, gdy nie
 *         udało się alokować pamięci lub wskaźnik \p pf ma wartość NULL.
 */
PhoneNumbers *phfwdReverse(const PhoneForward *pf, const char *num);

/** \brief Usuwa strukturę.
 * Usuwa strukturę wskazywaną przez \p pnum. Nic nie robi, jeśli wskaźnik ten ma
 * wartość NULL.
 * \param[in] pnum – wskaźnik na usuwaną strukturę.
 */
void phnumDelete(PhoneNumbers *pnum);

/** \brief Udostępnia numer.
 * Udostępnia wskaźnik na napis reprezentujący numer. Napisy są indeksowane
 * kolejno od zera.
 * \param[in] pnum – wskaźnik na strukturę przechowującą ciąg numerów telefonów;
 * \param[in] idx  – indeks numeru telefonu.
 * \return Wskaźnik na napis reprezentujący numer telefonu. Wartość NULL, jeśli
 *         wskaźnik \p pnum ma wartość NULL lub indeks ma za dużą wartość.
 */
const char *phnumGet(const PhoneNumbers *pnum, size_t idx);

#endif /* __PHONE_FORWARD_H__ */
