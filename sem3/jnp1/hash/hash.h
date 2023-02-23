#ifndef HASH_H_
#define HASH_H_

#ifdef __cplusplus
#include <cinttypes>
#include <cstdlib>
#else
#include <inttypes.h>
#include <stdbool.h>
#include <stdlib.h>
#endif

#ifdef __cplusplus
namespace jnp1 {
extern "C" {
#endif

#ifdef __cplusplus
using hash_function_t = uint64_t (*)(const uint64_t *, size_t);
#else
typedef uint64_t (*hash_function_t)(const uint64_t *, size_t);
#endif

/**
 * Tworzy tablicę haszującą i zwraca jej identyfikator. Parametr
 * hash_function jest wskaźnikiem na funkcję haszującą, która daje w wyniku
 * liczbę uint64_t i ma kolejno parametry uint64_t const * oraz size_t.
 */
unsigned long hash_create(hash_function_t hash_function);

/**
 * Usuwa tablicę haszującą o identyfikatorze id, o ile ona istnieje.
 * W przeciwnym przypadku nic nie robi.
 */
void hash_delete(unsigned long id);

/**
 * Daje liczbę ciągów przechowywanych w tablicy haszującej
 * o identyfikatorze id lub 0, jeśli taka tablica nie istnieje.
 */
size_t hash_size(unsigned long id);

/**
 * Wstawia do tablicy haszującej o identyfikatorze id ciąg liczb
 * całkowitych seq o długości size. Wynikiem jest informacja, czy operacja
 * się powiodła. Operacja się nie powiedzie, jeśli nie ma takiej tablicy
 * haszującej, jeśli tablica haszująca zawiera już taki ciąg, jeśli
 * parametr seq ma wartość NULL lub parametr size ma wartość 0.
 */
bool hash_insert(unsigned long id, uint64_t const * seq, size_t size);

/**
 * Usuwa z tablicy haszującej o identyfikatorze id ciąg liczb całkowitych
 * seq o długości size. Wynikiem jest informacja, czy operacja się
 * powiodła. Operacja się nie powiedzie, jeśli nie ma takiej tablicy
 * haszującej, jeśli tablica haszująca nie zawiera takiego ciągu,
 * jeśli parametr seq ma wartość NULL lub parametr size ma wartość 0.
 */
bool hash_remove(unsigned long id, uint64_t const * seq, size_t size);

/**
 * Jeśli tablica haszująca o identyfikatorze id istnieje i nie jest pusta,
 * to usuwa z niej wszystkie elementy. W przeciwnym przypadku nic nie robi.
 */
void hash_clear(unsigned long id);

/**
 * Daje wynik true, jeśli istnieje tablica haszująca o identyfikatorze id
 * i zawiera ona ciąg liczb całkowitych seq o długości size. Daje wynik
 * false w przeciwnym przypadku oraz gdy parametr seq ma wartość NULL lub
 * parametr size ma wartość 0.
 */
bool hash_test(unsigned long id, uint64_t const * seq, size_t size);

#ifdef __cplusplus
} // extern "C"
} // namespace jnp1
#endif

#endif /* HASH_H_ */
