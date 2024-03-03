#ifndef A3_UTIL_H_
#define A3_UTIL_H_

/** @brief Zapisuje liczbę całkowitą do @p dst.
 * Zapisuje wartość @p x w postaci dziesiętnej do bufora wskazywanego
 * przez @p dst.
 * @param[in,out] dst - wskaźnik na bufor;
 * @param[in] x       - zapisywana wartość.
 * @return Wskaźnik na koniec bufora po zapisie.
 */
char *print_dec(char *dst, int x);

/** @brief Zapisuje napis do @p dst.
 * Zapisuje zakończony zerowym bajtem napis @p str do bufora wskazywanego
 * przez @p dst.
 * @param[in,out] dst - wskaźnik na bufor;
 * @param[in] str     - zapisywany napis.
 * @return Wskaźnik na koniec bufora po zapisie.
 */
char *print_str(char *dst, const char *str);

#endif /* A3_UTIL_H_ */
