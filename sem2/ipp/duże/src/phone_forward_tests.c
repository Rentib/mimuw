#ifndef _GNU_SOURCE
  #define _GNU_SOURCE
#endif

// Ten plik włączamy jako pierwszy i dwa razy, aby sprawdzić, czy zawiera
// wszystkie potrzebne deklaracje i definicje oraz ochronę przed wielokrotnym
// włączeniem.
#include "phone_forward.h"
#include "phone_forward.h"

#include <malloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

// Gdzieś musi być zdefiniowany magiczny napis służący do spawdzania, czy
// program w całości wykonał się poprawnie.
extern char quite_long_magic_string[];
char quite_long_magic_string[] = "MAGIC";

// Możliwe wyniki testu
#define PASS 0
#define FAIL 1
#define WRONG_TEST 2
#define PASS_INSTRUMENTED 3

// Liczba elementów tablicy x
#define SIZE(x) (sizeof(x) / sizeof(x)[0])

// Początek testu
#define INIT(p)                 \
  PhoneForward *p = phfwdNew(); \
  if (p == NULL)                \
    return FAIL

// Utworzenie nowej bazy przekierowań
#define REINIT(p)    \
  phfwdDelete(pf);   \
  N(p = phfwdNew());

// Koniec testu
#define CLEAN(p)  \
  phfwdDelete(p); \
  return PASS;

// Wypełnienie numeru znakiem
#define FILL(p, from, to, c)           \
  do {                                 \
    for (int _k = from; _k < to; ++_k) \
      p[_k] = c;                       \
    p[to] = '\0';                      \
  } while (0)

// Skopiowanie numeru
#define COPY(dst, src) \
  strcpy(dst, src)

// Oczekiwane jednakowe numery
#define C(x, y)            \
  do {                     \
    if (strcmp(x, y) != 0) \
      return FAIL;         \
  } while (0)

// Oczekiwany zerowy wynik funkcji
#define Z(f)       \
  do {             \
    if ((f) != 0)  \
      return FAIL; \
  } while (0)

// Oczekiwany wynik funkcji false
#define F(f)          \
  do {                \
    if ((f) != false) \
      return FAIL;    \
  } while (0)

// Oczekiwany niezerowy wynik funkcji
#define N(f)       \
  do {             \
    if ((f) == 0)  \
      return FAIL; \
  } while (0)

// Oczekiwany wynik funkcji true
#define T(f)         \
  do {               \
    if ((f) != true) \
      return FAIL;   \
  } while (0)

// Oczekiwany numer n jako i-ty w strukturze PhoneNumbers
#define R(p, i, n) \
  C(phnumGet(p, i), n)

// Oczekiwany brak i-tego numeru w strukturze PhoneNumbers
#define Q(p, i) \
  Z(phnumGet(p, i))

// Oczekiwana pusta struktura PhoneNumbers
#define E(f)         \
  do {               \
   PhoneNumbers *_p; \
   N(_p = f);        \
   Q(_p, 0);         \
   Q(_p, 1);         \
   Q(_p, 9999);      \
   Q(_p, SIZE_MAX);  \
   phnumDelete(_p);  \
  } while (0)

// Oczekiwane przekierowanie z A na B
#define CHECK(p, A, B)      \
  do {                      \
    PhoneNumbers *_p;       \
    N(_p = phfwdGet(p, A)); \
    R(_p, 0, B);            \
    Q(_p, 1);               \
    phnumDelete(_p);        \
  } while (0)

// Oczekiwane przekierowanie z A na B i dodatkowe sprawdzenie indeksu SIZE_MAX
#define CHECK_SIZE(p, A, B) \
  do {                      \
    PhoneNumbers *_p;       \
    N(_p = phfwdGet(p, A)); \
    R(_p, 0, B);            \
    Q(_p, 1);               \
    Q(_p, SIZE_MAX);        \
    phnumDelete(_p);        \
  } while (0)

// Oczekiwane odwrotne przekierowania z A
#define RCHCK(p, A, ...)                     \
  do {                                       \
    PhoneNumbers *_p;                        \
    N(_p = phfwdReverse(p, A));              \
    char const *_r[] = {__VA_ARGS__};        \
    for (size_t _k = 0; _k < SIZE(_r); ++_k) \
      R(_p, _k, _r[_k]);                     \
    Q(_p, SIZE(_r));                         \
    phnumDelete(_p);                         \
  } while (0)

// Oczekiwane odwrotne przekierowania z A i dodatkowe sprawdzenie indeksu SIZE_MAX
#define RCHCK_SIZE(p, A, ...)                \
  do {                                       \
    PhoneNumbers *_p;                        \
    N(_p = phfwdReverse(p, A));              \
    char const *_r[] = {__VA_ARGS__};        \
    for (size_t _k = 0; _k < SIZE(_r); ++_k) \
      R(_p, _k, _r[_k]);                     \
    Q(_p, SIZE(_r));                         \
    Q(_p, SIZE_MAX);                         \
    phnumDelete(_p);                         \
  } while (0)

// Oczekiwane odwrotne przekierowania z A
#define GRCHK(p, A, ...)                     \
  do {                                       \
    PhoneNumbers *_p;                        \
    N(_p = phfwdGetReverse(p, A));           \
    char const *_r[] = {__VA_ARGS__};        \
    for (size_t _k = 0; _k < SIZE(_r); ++_k) \
      R(_p, _k, _r[_k]);                     \
    Q(_p, SIZE(_r));                         \
    phnumDelete(_p);                         \
  } while (0)


// Oczekiwane odwrotne przekierowania z A i dodatkowe sprawdzenie indeksu SIZE_MAX
#define GRCHK_SIZE(p, A, ...)                \
  do {                                       \
    PhoneNumbers *_p;                        \
    N(_p = phfwdGetReverse(p, A));           \
    char const *_r[] = {__VA_ARGS__};        \
    for (size_t _k = 0; _k < SIZE(_r); ++_k) \
      R(_p, _k, _r[_k]);                     \
    Q(_p, SIZE(_r));                         \
    Q(_p, SIZE_MAX);                         \
    phnumDelete(_p);                         \
  } while (0)

/** WŁAŚCIWE TESTY **/

// Tylko utworzenie i usunięcie struktury
static int empty_struct(void) {
  INIT(pf);
  CLEAN(pf);
}

// Brak przekierowań
static int no_forward(void) {
  char n[2];

  INIT(pf);

  n[1] = '\0';
  for (n[0] = '9'; n[0] >= '0'; --n[0])
    phfwdRemove(pf, n);
  for (n[0] = '0'; n[0] <= '9'; ++n[0])
    CHECK(pf, n, n);
  for (n[0] = '9'; n[0] >= '0'; --n[0])
    RCHCK(pf, n, n);

  CLEAN(pf);
}

// Błędne argumenty funkcji
static int wrong_arguments(void) {
  INIT(pf);

  F(phfwdAdd(pf, "123", ""));
  F(phfwdAdd(pf, "123", "a"));
  F(phfwdAdd(pf, "123", " "));
  F(phfwdAdd(pf, "123", "9b"));
  F(phfwdAdd(pf, "123", "@"));
  F(phfwdAdd(pf, "123", "/"));
  F(phfwdAdd(pf, "123", ":"));
  F(phfwdAdd(pf, "123", ";"));
  F(phfwdAdd(pf, "123", ">"));
  F(phfwdAdd(pf, "123", "0?"));
  F(phfwdAdd(pf, "", "123"));
  F(phfwdAdd(pf, "c", "123"));
  F(phfwdAdd(pf, " ", "123"));
  F(phfwdAdd(pf, "8d", "123"));
  F(phfwdAdd(pf, "/", "123"));
  F(phfwdAdd(pf, ":", "123"));
  F(phfwdAdd(pf, ";", "123"));
  F(phfwdAdd(pf, "1>", "123"));
  F(phfwdAdd(pf, "?", "123"));
  F(phfwdAdd(pf, "@", "123"));
  F(phfwdAdd(pf, "0", "0"));

  F(phfwdAdd(NULL, "1", "2"));
  F(phfwdAdd(NULL, NULL, "2"));
  F(phfwdAdd(NULL, NULL, "abc"));
  F(phfwdAdd(NULL, "1", NULL));
  F(phfwdAdd(NULL, "abc", NULL));
  F(phfwdAdd(NULL, NULL, NULL));

  E(phfwdGet(pf, ""));
  E(phfwdGet(pf, "e"));
  E(phfwdGet(pf, " "));
  E(phfwdGet(pf, "7f"));
  E(phfwdGet(pf, "%"));
  E(phfwdGet(pf, "/"));
  E(phfwdGet(pf, "%"));
  E(phfwdGet(pf, ":"));
  E(phfwdGet(pf, ";"));
  E(phfwdGet(pf, "?"));
  E(phfwdGet(pf, ">"));
  E(phfwdGet(pf, "@"));

  Z(phfwdGet(NULL, "123"));
  Z(phfwdGet(NULL, NULL));
  Z(phfwdGet(NULL, "abc"));

  E(phfwdReverse(pf, ""));
  E(phfwdReverse(pf, "e"));
  E(phfwdReverse(pf, " "));
  E(phfwdReverse(pf, "7f"));
  E(phfwdReverse(pf, "/"));
  E(phfwdReverse(pf, "%"));
  E(phfwdReverse(pf, ":"));
  E(phfwdReverse(pf, ";"));
  E(phfwdReverse(pf, ">"));
  E(phfwdReverse(pf, "@"));

  Z(phfwdReverse(NULL, "123"));
  Z(phfwdReverse(NULL, NULL));
  Z(phfwdReverse(NULL, "abc"));

  E(phfwdGetReverse(pf, ""));
  E(phfwdGetReverse(pf, "e"));
  E(phfwdGetReverse(pf, " "));
  E(phfwdGetReverse(pf, "7f"));
  E(phfwdGetReverse(pf, "/"));
  E(phfwdGetReverse(pf, "%"));
  E(phfwdGetReverse(pf, ":"));
  E(phfwdGetReverse(pf, ";"));
  E(phfwdGetReverse(pf, ">"));
  E(phfwdGetReverse(pf, "@"));

  Z(phfwdGetReverse(NULL, "123"));
  Z(phfwdGetReverse(NULL, NULL));
  Z(phfwdGetReverse(NULL, "abc"));

  phfwdRemove(pf, "");
  phfwdRemove(pf, "g");
  phfwdRemove(pf, " ");
  phfwdRemove(pf, "6h");
  phfwdRemove(pf, "/");
  phfwdRemove(pf, "%");
  phfwdRemove(pf, ":");
  phfwdRemove(pf, ";");
  phfwdRemove(pf, ">");
  phfwdRemove(pf, "@");

  phfwdRemove(NULL, "1");

  phfwdDelete(NULL);

  phnumDelete(NULL);

  Z(phnumGet(NULL, 0));
  Z(phnumGet(NULL, 1));
  Z(phnumGet(NULL, SIZE_MAX));

  CLEAN(pf);
}

// Błędne argumenty funkcji bez kończącego zera
static int malicious_arguments(void) {
  INIT(pf);

  long page_size = sysconf(_SC_PAGE_SIZE);
  char *mem = mmap(NULL, 2 * page_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == (char *)-1)
    return FAIL;
  if (munmap(mem + page_size, page_size) < 0)
    return FAIL;
  memset(mem, 'A', page_size);

  F(phfwdAdd(pf, mem, mem));
  F(phfwdAdd(pf, mem, "1"));
  F(phfwdAdd(pf, "2", mem));
  E(phfwdGet(pf, mem));
  E(phfwdReverse(pf, mem));
  E(phfwdGetReverse(pf, mem));
  phfwdRemove(pf, mem);

  if (munmap(mem, page_size) < 0)
    return FAIL;

  CLEAN(pf);
}

// Niepsucie bazy przez błędne wywołania funkcji
static int breaking_struct(void) {
  char a[2], b[2];

  INIT(pf);

  a[1] = b[1] = '\0';
  for (a[0] = '0', b[0] = '9'; a[0] <= '9'; ++a[0], --b[0])
    T(phfwdAdd(pf, a, b));

  F(phfwdAdd(pf, "1", "A"));
  F(phfwdAdd(pf, "2D", "0"));
  F(phfwdAdd(pf, "B", "2"));
  phfwdRemove(pf, "C");

  for (a[0] = '0', b[0] = '9'; a[0] <= '9'; ++a[0], --b[0]) {
    CHECK(pf, a, b);
    if (a[0] < b[0])
      RCHCK(pf, b, a, b);
    else
      RCHCK(pf, b, b, a);
  }

  CLEAN(pf);
}

// Długie numery
static int long_numbers(void) {
  #define LEN1 1000
  #define LEN2 4500
  #define LEN3 1500
  #define LEN4 5000

  char n1[LEN1 + 1], n2[LEN2 + 1], n3[LEN3 + 1], n4[LEN4 + 1];

  FILL(n1, 0, LEN1, '1');
  FILL(n2, 0, LEN1, '1');
  FILL(n2, LEN1, LEN2, '2');
  FILL(n3, 0, LEN3, '3');
  FILL(n4, 0, LEN3, '3');
  FILL(n4, LEN3, LEN4, '2');

  INIT(pf);

  T(phfwdAdd(pf, n1, n3));
  CHECK(pf, n2, n4);
  RCHCK(pf, n4, n2, n4);

  CLEAN(pf);

  #undef LEN1
  #undef LEN2
  #undef LEN3
  #undef LEN4
}

// Dużo numerów w strukturze PhoneNumbers
static int many_numbers(void) {
  #define LEN9 20
  #define COUNT9 9

  char n9[COUNT9 * COUNT9][LEN9 + 1];
  PhoneNumbers *pn;

  for (int i = 0; i < COUNT9; ++i) {
    for (int j = 0; j < COUNT9; ++j) {
      FILL(n9[COUNT9 * i + j], 0, LEN9 / 2, '0' + i);
      FILL(n9[COUNT9 * i + j], LEN9 / 2, LEN9, '0' + j);
    }
  }

  INIT(pf);

  for (int i = 0; i < COUNT9 * COUNT9; ++i)
    T(phfwdAdd(pf, n9[i], "9"));
  N(pn = phfwdReverse(pf, "9"));
  for (int i = 0; i < COUNT9 * COUNT9; ++i)
    R(pn, i, n9[i]);
  R(pn, COUNT9 * COUNT9, "9");
  Q(pn, COUNT9 * COUNT9 + 1);
  phnumDelete(pn);

  CLEAN(pf);

  #undef LEN9
  #undef COUNT9
}

// Kopiowanie wartości do wnętrza struktury
static int copy_arguments(void) {
  #define LEN5 5

  static const char p1[] = "123";
  static const char p2[] = "456";
  char p3[LEN5 + 1], p4[LEN5 + 1];

  INIT(pf);

  COPY(p3, p1);
  COPY(p4, p2);

  T(phfwdAdd(pf, p3, p4));

  FILL(p3, 0, LEN5, '\0');
  FILL(p4, 0, LEN5, '\0');

  CHECK(pf, "12345", "45645");

  CLEAN(pf);

  #undef LEN5
}

// Dwie instancje struktury
static int two_structs(void) {
  INIT(pf1);
  INIT(pf2);

  T(phfwdAdd(pf1, "1", "5"));
  T(phfwdAdd(pf2, "2", "6"));

  CHECK(pf1, "1", "5");
  CHECK(pf1, "2", "2");
  CHECK(pf2, "1", "1");
  CHECK(pf2, "2", "6");

  phfwdDelete(pf1);
  CLEAN(pf2);
}

// Delete ze wskaźnikiem NULL
static int delete_null(void) {
  phnumDelete(NULL);
  CLEAN(NULL);
}

// Zachowanie wyniku po zmianie struktury
static int persistent_results(void) {
  PhoneNumbers *png;
  PhoneNumbers *pnr;

  INIT(pf);

  T(phfwdAdd(pf, "3", "4"));

  N(png = phfwdGet(pf, "33"));
  N(pnr = phfwdReverse(pf, "43"));

  T(phfwdAdd(pf, "3", "5"));
  phfwdRemove(pf, "3");
  phfwdDelete(pf);

  R(png, 0, "43");
  Q(png, 1);
  phnumDelete(png);

  R(pnr, 0, "33");
  R(pnr, 1, "43");
  Q(pnr, 2);
  phnumDelete(pnr);

  return PASS;
}

// Nadpisanie przekierowania
static int forward_overwrite(void) {
  INIT(pf);

  CHECK(pf, "000", "000");

  T(phfwdAdd(pf, "0", "5"));
  CHECK(pf, "000", "500");

  T(phfwdAdd(pf, "0", "7"));
  CHECK(pf, "000", "700");

  CLEAN(pf);
}

// Usunięcie przekierowania
static int remove_forward(void) {
  INIT(pf);

  T(phfwdAdd(pf, "0", "5"));
  CHECK(pf, "000", "500");

  phfwdRemove(pf, "0");
  CHECK(pf, "000", "000");

  CLEAN(pf);
}

// Proste przekierowanie odwrotne
static int simple_reverse(void) {
  INIT(pf);

  T(phfwdAdd(pf, "2", "4"));
  T(phfwdAdd(pf, "23", "4"));
  RCHCK(pf, "434", "2334", "234", "434");

  CLEAN(pf);
}

// Różne operacje na strukturze
static int various_ops(void) {
  INIT(pf);

  T(phfwdAdd(pf, "123", "9"));
  CHECK(pf, "1234", "94");
  T(phfwdAdd(pf, "123456", "777777"));
  CHECK(pf, "12345", "945");
  CHECK(pf, "123456", "777777");
  CHECK(pf, "997", "997");
  RCHCK(pf, "987654321", "12387654321", "987654321");
  phfwdRemove(pf, "12");
  CHECK(pf, "123456", "123456");
  RCHCK(pf, "987654321", "987654321");
  T(phfwdAdd(pf, "567", "0"));
  T(phfwdAdd(pf, "5678", "08"));
  RCHCK(pf, "08", "08", "5678");
  T(phfwdAdd(pf, "12", "123"));
  CHECK(pf, "123", "1233");
  T(phfwdAdd(pf, "2", "4"));
  T(phfwdAdd(pf, "23", "4"));
  RCHCK(pf, "434", "2334", "234", "434");

  REINIT(pf);

  T(phfwdAdd(pf, "123", "76"));
  CHECK_SIZE(pf, "1234581", "764581");
  CHECK_SIZE(pf, "7581", "7581");
  RCHCK_SIZE(pf, "7581", "7581");

  CLEAN(pf);
}

// Dużo operacji na strukturze
static int many_ops(void) {
  #define xfrom   00000
  #define xto     99999
  #define div0    1
  #define format0 "%05u"
  #define div1    10
  #define format1 "%04u"
  #define div2    100
  #define format2 "%03u"
  #define div3    1000
  #define format3 "%02u"

  char b1[16], b2[16];

  INIT(pf);

  for (unsigned i = xfrom / div0; i <= xto / div0; ++i) {
    sprintf(b1, format0, i);
    sprintf(b2, format2, i / div2);
    T(phfwdAdd(pf, b1, b2));
  }

  for (unsigned i = xfrom / div0; i <= xto / div0; ++i) {
    sprintf(b1, format0 "123", i);
    sprintf(b2, format2 "123", i / div2);
    CHECK(pf, b1, b2);
  }

  for (unsigned i = xfrom / div1; i <= xto / div1; ++i) {
    sprintf(b1, format1, i);
    CHECK(pf, b1, b1);
  }

  for (unsigned i = xfrom / div3; i <= xto / div3; ++i) {
    sprintf(b1, format3, i);
    RCHCK(pf, b1, b1);
  }

  for (unsigned i = xfrom / div2; i <= xto / div2; ++i) {
    sprintf(b1, format2, i);
    phfwdRemove(pf, b1);
  }

  for (unsigned i = xfrom / div0; i <= xto / div0; ++i) {
    sprintf(b1, format0, i);
    CHECK(pf, b1, b1);
    RCHCK(pf, b1, b1);
  }

  CLEAN(pf);

  #undef xfrom
  #undef xto
  #undef div0
  #undef format0
  #undef div1
  #undef format1
  #undef div2
  #undef format2
  #undef div3
  #undef format3
}

// Bardzo długie numery
static int very_long(void) {
  #define LONG_LEN 250000

  char *base, b[3];

  INIT(pf);
  N(base = malloc(sizeof(char) * (LONG_LEN + 3)));

  for (int i = 0; i < LONG_LEN; ++i)
    base[i] = '0' + i % 10;
  base[LONG_LEN + 2] = '\0';
  b[2] = '\0';
  for (int i = 0; i <= 99; ++i) {
    b[0] = '0' + i / 10;
    b[1] = '0' + i % 10;
    base[LONG_LEN]     = '0' + i % 10;
    base[LONG_LEN + 1] = '0' + i / 10;
    T(phfwdAdd(pf, base, b));
  }
  for (int i = 0 ; i <= 99; ++i) {
    b[0] = '0' + i / 10;
    b[1] = '0' + i % 10;
    base[LONG_LEN]     = '0' + i % 10;
    base[LONG_LEN + 1] = '0' + i / 10;
    CHECK(pf, base, b);
    if (i < 2)
      RCHCK(pf, b, b, base);
    else
      RCHCK(pf, b, base, b);
  }

  free(base);
  CLEAN(pf);

  #undef LONG_LEN
}

// Duża liczba przekierowań
static int many_remove(void) {
  #define BIG_COUNT 10000

  char *base;

  INIT(pf);
  N(base = malloc(sizeof(char) * (BIG_COUNT + 1)));

  for (int i = 0; i < BIG_COUNT; ++i)
    base[i] = '0' + i % 10;
  for (int i = BIG_COUNT; i > 1; --i) {
    base[i] = '\0';
    T(phfwdAdd(pf, base, "0"));
  }
  T(phfwdAdd(pf, "123456789", "0"));
  phfwdRemove(pf, "0");
  RCHCK(pf, "0123456789", "0123456789", "123456789123456789");

  free(base);
  CLEAN(pf);

  #undef BIG_COUNT
}

// Intensywne dodawanie i usuwanie przekierowań
static int add_remove(void) {
  #define yfrom   0000
  #define yto     9999
  #define yformat "%04u"
  #define REPEAT_COUNT 27

  char b1[8], b2[16];

  INIT(pf);

  for (unsigned i = yfrom; i <= yto; ++i) {
    sprintf(b1, yformat, i);
    sprintf(b2, yformat yformat, i, i);
    T(phfwdAdd(pf, b1, b2));
  }
  for (unsigned i = yfrom; i <= yto; ++i) {
    sprintf(b1, yformat, i);
    sprintf(b2, yformat yformat, i, i);
    for (unsigned j = 0; j < REPEAT_COUNT; ++j) {
      phfwdRemove(pf, b1);
      T(phfwdAdd(pf, b1, b2));
    }
  }
  for (unsigned i = yfrom; i <= yto; ++i) {
    sprintf(b1, yformat, i);
    sprintf(b2, yformat yformat, i, i);
    CHECK(pf, b1, b2);
    RCHCK(pf, b2, b1, b2);
  }

  CLEAN(pf);

  #undef yfrom
  #undef yto
  #undef yformat
  #undef REPEAT_COUNT
}

// Różne operacje na strukturze z wykorzystaniem 12 cyfr
static int twelve_digits(void) {
  INIT(pf);

  T(phfwdAdd(pf, "*#3", "9"));
  CHECK(pf, "*#34", "94");
  T(phfwdAdd(pf, "*#3456", "777777"));
  CHECK(pf, "*#345", "945");
  CHECK(pf, "*#3456", "777777");
  CHECK(pf, "997", "997");
  RCHCK(pf, "9876543#*", "9876543#*", "*#3876543#*");
  phfwdRemove(pf, "*#");
  CHECK(pf, "*#3456", "*#3456");
  RCHCK(pf, "9876543#*", "9876543#*");
  T(phfwdAdd(pf, "567", "0"));
  T(phfwdAdd(pf, "5678", "08"));
  RCHCK(pf, "08", "08", "5678");
  T(phfwdAdd(pf, "*#", "*#3"));
  CHECK(pf, "*#3", "*#33");
  T(phfwdAdd(pf, "#", "4"));
  T(phfwdAdd(pf, "#3", "4"));
  RCHCK(pf, "434", "434", "#334", "#34");

  REINIT(pf);

  T(phfwdAdd(pf, "1#3", "76"));
  CHECK(pf, "1#3458*", "76458*");
  CHECK(pf, "758*2", "758*2");
  RCHCK(pf, "758*2", "758*2");

  CLEAN(pf);
}

// Test dwóch numerów z wzajemnym przekierowaniem jednego na drugi
static int cycle(void) {
  INIT(pf);

  T(phfwdAdd(pf, "123", "124"));
  T(phfwdAdd(pf, "124", "123"));
  CHECK(pf, "123", "124");
  CHECK(pf, "124", "123");
  RCHCK(pf, "123", "123", "124");
  RCHCK(pf, "124", "123", "124");
  GRCHK(pf, "123", "124");
  GRCHK(pf, "124", "123");

  CLEAN(pf);
}

// Test sortowania wyniku phfwdReverse i phfwdGetReverse
static int sort(void) {
  INIT(pf);

  T(phfwdAdd(pf, "027", "07"));
  T(phfwdAdd(pf, "0*7", "07"));
  T(phfwdAdd(pf, "097", "07"));
  T(phfwdAdd(pf, "037", "07"));
  T(phfwdAdd(pf, "057", "07"));
  T(phfwdAdd(pf, "077", "07"));
  T(phfwdAdd(pf, "067", "07"));
  T(phfwdAdd(pf, "047", "07"));
  T(phfwdAdd(pf, "007", "07"));
  T(phfwdAdd(pf, "017", "07"));
  T(phfwdAdd(pf, "0#7", "07"));
  T(phfwdAdd(pf, "087", "07"));
  RCHCK(pf, "07", "007", "017", "027", "037", "047", "057", "067", "07", "077", "087", "097", "0*7", "0#7");
  GRCHK(pf, "07", "007", "017", "027", "037", "047", "057", "067", "07", "077", "087", "097", "0*7", "0#7");

  CLEAN(pf);
}

// Testy funkcji phfwdGetReverse
static int get_reverse(void) {
  INIT(pf);

  T(phfwdAdd(pf, "1", "2"));
  T(phfwdAdd(pf, "3", "4"));
  T(phfwdAdd(pf, "5", "6"));
  T(phfwdAdd(pf, "7", "8"));
  T(phfwdAdd(pf, "9", "10"));
  T(phfwdAdd(pf, "11", "12"));
  T(phfwdAdd(pf, "13", "14"));
  T(phfwdAdd(pf, "15", "16"));
  GRCHK(pf, "12", "11");

  REINIT(pf);

  T(phfwdAdd(pf, "1", "2"));
  T(phfwdAdd(pf, "1", "3"));
  T(phfwdAdd(pf, "1", "4"));
  T(phfwdAdd(pf, "1", "5"));
  T(phfwdAdd(pf, "1", "6"));
  T(phfwdAdd(pf, "1", "7"));
  T(phfwdAdd(pf, "1", "8"));
  T(phfwdAdd(pf, "1", "9"));
  T(phfwdAdd(pf, "1", "10"));
  T(phfwdAdd(pf, "1", "11"));
  GRCHK(pf, "11", "1");

  phfwdRemove(pf, "1");

  T(phfwdAdd(pf, "1", "2"));
  T(phfwdAdd(pf, "2", "3"));
  T(phfwdAdd(pf, "3", "4"));
  T(phfwdAdd(pf, "4", "5"));
  T(phfwdAdd(pf, "5", "6"));
  T(phfwdAdd(pf, "6", "7"));
  T(phfwdAdd(pf, "7", "8"));
  T(phfwdAdd(pf, "8", "9"));
  T(phfwdAdd(pf, "9", "*"));
  T(phfwdAdd(pf, "*", "#"));
  T(phfwdAdd(pf, "#", "0"));
  T(phfwdAdd(pf, "0", "1"));
  GRCHK(pf, "5", "4");

  REINIT(pf);

  T(phfwdAdd(pf, "123", "9"));
  T(phfwdAdd(pf, "123456", "777777"));
  T(phfwdAdd(pf, "431", "432"));
  T(phfwdAdd(pf, "432", "433"));

  GRCHK(pf, "432", "431");
  GRCHK(pf, "433", "432", "433");
  GRCHK(pf, "987654321", "12387654321", "987654321");

  phfwdRemove(pf, "12");

  GRCHK(pf, "987654321", "987654321");

  T(phfwdAdd(pf, "567", "0"));
  T(phfwdAdd(pf, "5678", "08"));

  GRCHK(pf, "08", "08", "5678");

  T(phfwdAdd(pf, "12", "123"));
  T(phfwdAdd(pf, "2", "4"));
  T(phfwdAdd(pf, "23", "4"));
  GRCHK(pf, "434", "2334", "434");

  CLEAN(pf);
}

/** TESTY ALOKACJI PAMIĘCI
    Te testy muszą być linkowane z opcjami
    -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc
    -Wl,--wrap=reallocarray -Wl,--wrap=free -Wl,--wrap=strdup -Wl,--wrap=strndup
**/

// Przechwytujemy funkcje alokujące i zwalniające pamięć.
void *__real_malloc(size_t size) __attribute__((weak));
void *__real_calloc(size_t nmemb, size_t size)__attribute__((weak));
void *__real_realloc(void *ptr, size_t size)__attribute__((weak));
void *__real_reallocarray(void *ptr, size_t nmemb, size_t size)__attribute__((weak));
char *__real_strdup(const char *s)__attribute__((weak));
char *__real_strndup(const char *s, size_t size)__attribute__((weak));
void __real_free(void *ptr)__attribute__((weak));

// Trzymamy globalnie informacje o alokacjach i zwolnieniach pamięci i nie
// pozwalamy ich optymalizować.
static volatile unsigned call_counter = 0;  // licznik wywołań alokacji
static volatile unsigned fail_counter = 0;  // numer błędnej alokacji
static volatile unsigned alloc_counter = 0; // liczba wykonanych alokacji
static volatile unsigned free_counter = 0;  // liczba wykonanych zwolnień
static volatile char *function_name = NULL; // nazwa nieudanej funkcji
static volatile bool wrap_flag = false;     // przechwytujące funkcje były wołane

// W zadanym momencie alokacja pamięci zawodzi.
static bool should_fail(void) {
  return ++call_counter == fail_counter;
}

// Realokacja musi się udać, jeśli nie zwiększamy rozmiaru alokowanej pamięci.
static bool can_fail(void const *old_ptr, size_t new_size) {
  if (old_ptr == NULL)
    return true;
  else
    return new_size > malloc_usable_size((void *)old_ptr);
}

// Symulujemy brak pamięci.
#define UNRELIABLE_ALLOC(ptr, size, fun, name) \
  do { \
    wrap_flag = true; \
    if (ptr != NULL && size == 0) { \
      /* Takie wywołanie realloc jest równoważne wywołaniu free(ptr). */ \
      ++free_counter; \
      return fun; \
    } \
    void *p = can_fail(ptr, size) && should_fail() ? NULL : (fun); \
    if (p) { \
      alloc_counter += ptr != p; \
      free_counter += ptr != p && ptr != NULL; \
    } \
    else { \
      function_name = name; \
    } \
    return p; \
  } while (0)

void *__wrap_malloc(size_t size) {
  UNRELIABLE_ALLOC(NULL, size, __real_malloc(size), "malloc");
}

void *__wrap_calloc(size_t nmemb, size_t size) {
  UNRELIABLE_ALLOC(NULL, nmemb * size, __real_calloc(nmemb, size), "calloc");
}

void *__wrap_realloc(void *ptr, size_t size) {
  UNRELIABLE_ALLOC(ptr, size, __real_realloc(ptr, size), "realloc");
}

void *__wrap_reallocarray(void *ptr, size_t nmemb, size_t size) {
  UNRELIABLE_ALLOC(ptr, nmemb * size, __real_reallocarray(ptr, nmemb, size), "reallocarray");
}

char *__wrap_strdup(const char *s) {
  UNRELIABLE_ALLOC(NULL, 0, __real_strdup(s), "strdup");
}

char *__wrap_strndup(const char *s, size_t size) {
  UNRELIABLE_ALLOC(NULL, 0, __real_strndup(s, size), "strndup");
}

// Zwalnianie pamięci zawsze się udaje. Odnotowujemy jedynie fakt zwolnienia.
void __wrap_free(void *ptr) {
  __real_free(ptr);
  if (ptr)
    ++free_counter;
}

#define V(code, where) (((unsigned long)code) << (3 * where))

// Test reakcji implementacji na niepowodzenie alokacji pamięci
// w funkcjach phfwdNew, phfwdAdd, phfwdGet
static unsigned long alloc_fail_test_1(void) {
  unsigned long visited = 0;
  PhoneForward *pf = NULL;
  PhoneNumbers *pn = NULL;
  char const   *ps = NULL;

  if ((pf = phfwdNew()) != NULL) {
    visited |= V(1, 0);
  }
  else if ((pf = phfwdNew()) != NULL) {
    visited |= V(2, 0);
  }
  else {
    visited |= V(4, 0);
    return visited;
  }

  if (phfwdAdd(pf, "579", "4")) {
    visited |= V(1, 1);
  }
  else if (phfwdAdd(pf, "579", "4")) {
    visited |= V(2, 1);
  }
  else {
    visited |= V(4, 1);
    phfwdDelete(pf);
    return visited;
  }

  if ((pn = phfwdGet(pf, "5791")) != NULL) {
    visited |= V(1, 2);
  }
  else if ((pn = phfwdGet(pf, "5791")) != NULL) {
    visited |= V(2, 2);
  }
  else {
    visited |= V(4, 2);
    phfwdDelete(pf);
    return visited;
  }

  if ((ps = phnumGet(pn, 0)) != NULL)
    visited |= V(1, 3);
  else if ((ps = phnumGet(pn, 0)) != NULL)
    visited |= V(2, 3);
  else
    visited |= V(4, 3);

  if (ps != NULL && strcmp(ps, "41") != 0)
    visited |= V(4, 3);

  if (phnumGet(pn, 1) == NULL)
    visited |= V(1, 4);
  else
    visited |= V(4, 4);

  phnumDelete(pn);
  phfwdDelete(pf);
  return visited;
}

// Test reakcji implementacji na niepowodzenie alokacji pamięci
// w funkcjach phfwdNew, phfwdAdd, phfwdReverse
static unsigned long alloc_fail_test_2(void) {
  unsigned long visited = 0;
  PhoneForward *pf = NULL;
  PhoneNumbers *pn = NULL;
  char const   *ps = NULL;

  if ((pf = phfwdNew()) != NULL) {
    visited |= V(1, 0);
  }
  else if ((pf = phfwdNew()) != NULL) {
    visited |= V(2, 0);
  }
  else {
    visited |= V(4, 0);
    return visited;
  }

  if ((pn = phfwdReverse(pf, "0123456789*#")) != NULL) {
    visited |= V(1, 1);
  }
  else if ((pn = phfwdReverse(pf, "0123456789*#")) != NULL) {
    visited |= V(2, 1);
  }
  else {
    visited |= V(4, 1);
    phfwdDelete(pf);
    return visited;
  }

  if ((ps = phnumGet(pn, 0)) != NULL)
    visited |= V(1, 2);
  else if ((ps = phnumGet(pn, 0)) != NULL)
    visited |= V(2, 2);
  else
    visited |= V(4, 2);

  if  (ps != NULL && strcmp(ps, "0123456789*#") != 0)
    visited |= V(4, 2);

  if (phnumGet(pn, 1) == NULL)
    visited |= V(1, 3);
  else
    visited |= V(4, 3);

  phnumDelete(pn);
  pn = NULL;

  if (phfwdAdd(pf, "4", "7")) {
    visited |= V(1, 4);
  }
  else if (phfwdAdd(pf, "4", "7")) {
    visited |= V(2, 4);
  }
  else {
    visited |= V(4, 4);
    phfwdDelete(pf);
    return visited;
  }

  if (phfwdAdd(pf, "1", "7")) {
    visited |= V(1, 5);
  }
  else if (phfwdAdd(pf, "1", "7")) {
    visited |= V(2, 5);
  }
  else {
    visited |= V(4, 5);
    phfwdDelete(pf);
    return visited;
  }

  if ((pn = phfwdReverse(pf, "7")) != NULL) {
    visited |= V(1, 6);
  }
  else if ((pn = phfwdReverse(pf, "7")) != NULL) {
    visited |= V(2, 6);
  }
  else {
    visited |= V(4, 6);
    phfwdDelete(pf);
    return visited;
  }

  if ((ps = phnumGet(pn, 0)) != NULL)
    visited |= V(1, 7);
  else if ((ps = phnumGet(pn, 0)) != NULL)
    visited |= V(2, 7);
  else
    visited |= V(4, 7);

  if (ps != NULL && strcmp(ps, "1") != 0)
    visited |= V(4, 7);

  if ((ps = phnumGet(pn, 1)) != NULL)
    visited |= V(1, 8);
  else if ((ps = phnumGet(pn, 1)) != NULL)
    visited |= V(2, 8);
  else
    visited |= V(4, 8);

  if (ps != NULL && strcmp(ps, "4") != 0)
    visited |= V(4, 8);

  if ((ps = phnumGet(pn, 2)) != NULL)
    visited |= V(1, 9);
  else if ((ps = phnumGet(pn, 2)) != NULL)
    visited |= V(2, 9);
  else
    visited |= V(4, 9);

  if (ps != NULL && strcmp(ps, "7") != 0)
    visited |= V(4, 9);

  if (phnumGet(pn, 3) == NULL)
    visited |= V(1, 10);
  else
    visited |= V(4, 10);

  phnumDelete(pn);
  phfwdDelete(pf);
  return visited;
}

// Test reakcji implementacji na niepowodzenie alokacji pamięci
// w funkcjach phfwdNew, phfwdAdd, phfwdGetReverse
static unsigned long alloc_fail_test_3(void) {
  unsigned long visited = 0;
  PhoneForward *pf = NULL;
  PhoneNumbers *pn = NULL;
  char const   *ps = NULL;

  if ((pf = phfwdNew()) != NULL) {
    visited |= V(1, 0);
  }
  else if ((pf = phfwdNew()) != NULL) {
    visited |= V(2, 0);
  }
  else {
    visited |= V(4, 0);
    return visited;
  }

  if ((pn = phfwdGetReverse(pf, "0123456789*#")) != NULL) {
    visited |= V(1, 1);
  }
  else if ((pn = phfwdGetReverse(pf, "0123456789*#")) != NULL) {
    visited |= V(2, 1);
  }
  else {
    visited |= V(4, 1);
    phfwdDelete(pf);
    return visited;
  }

  if ((ps = phnumGet(pn, 0)) != NULL)
    visited |= V(1, 2);
  else if ((ps = phnumGet(pn, 0)) != NULL)
    visited |= V(2, 2);
  else
    visited |= V(4, 2);

  if  (ps != NULL && strcmp(ps, "0123456789*#") != 0)
    visited |= V(4, 2);

  if (phnumGet(pn, 1) == NULL)
    visited |= V(1, 3);
  else
    visited |= V(4, 3);

  phnumDelete(pn);
  pn = NULL;

  if (phfwdAdd(pf, "*#5", "*#8")) {
    visited |= V(1, 4);
  }
  else if (phfwdAdd(pf, "*#5", "*#8")) {
    visited |= V(2, 4);
  }
  else {
    visited |= V(4, 4);
    phfwdDelete(pf);
    return visited;
  }

  if (phfwdAdd(pf, "*#2", "*#8")) {
    visited |= V(1, 5);
  }
  else if (phfwdAdd(pf, "*#2", "*#8")) {
    visited |= V(2, 5);
  }
  else {
    visited |= V(4, 5);
    phfwdDelete(pf);
    return visited;
  }

  if ((pn = phfwdGetReverse(pf, "*#8")) != NULL) {
    visited |= V(1, 6);
  }
  else if ((pn = phfwdGetReverse(pf, "*#8")) != NULL) {
    visited |= V(2, 6);
  }
  else {
    visited |= V(4, 6);
    phfwdDelete(pf);
    return visited;
  }

  if ((ps = phnumGet(pn, 0)) != NULL)
    visited |= V(1, 7);
  else if ((ps = phnumGet(pn, 0)) != NULL)
    visited |= V(2, 7);
  else
    visited |= V(4, 7);

  if (ps != NULL && strcmp(ps, "*#2") != 0)
    visited |= V(4, 7);

  if ((ps = phnumGet(pn, 1)) != NULL)
    visited |= V(1, 8);
  else if ((ps = phnumGet(pn, 1)) != NULL)
    visited |= V(2, 8);
  else
    visited |= V(4, 8);

  if (ps != NULL && strcmp(ps, "*#5") != 0)
    visited |= V(4, 8);

  if ((ps = phnumGet(pn, 2)) != NULL)
    visited |= V(1, 9);
  else if ((ps = phnumGet(pn, 2)) != NULL)
    visited |= V(2, 9);
  else
    visited |= V(4, 9);

  if (ps != NULL && strcmp(ps, "*#8") != 0)
    visited |= V(4, 9);

  if (phnumGet(pn, 3) == NULL)
    visited |= V(1, 10);
  else
    visited |= V(4, 10);

  phnumDelete(pn);
  phfwdDelete(pf);
  return visited;
}

// Sprawdzenie reakcji implementacji na niepowodzenie alokacji pamięci
static int memory_test(unsigned long (* test_function)(void)) {
  unsigned fail, pass;
  for (fail = 0, pass = 0, fail_counter = 1; fail < 3 && pass < 3; ++fail_counter) {
    call_counter = 0;
    alloc_counter = 0;
    free_counter = 0;
    function_name = NULL;
    unsigned long visited_point = test_function();

    if (alloc_counter != free_counter || (visited_point & 0444444444444444444444UL) != 0) {
      fprintf(stderr,
              "fail_counter %u, alloc_counter %u, free_counter %u, "
              "function_name %s, visited_point %lo\n",
              fail_counter, alloc_counter, free_counter,
              function_name, visited_point);
      ++fail;
    }
    if (function_name == NULL)
      ++pass;
    else
      pass = 0;
  }
  if (wrap_flag && fail == 0)
    return PASS_INSTRUMENTED;
  else if (!wrap_flag && fail == 0)
    return PASS;
  else
    return FAIL;
}

static int alloc_fail_1(void) {
  return memory_test(alloc_fail_test_1);
}

static int alloc_fail_2(void) {
  return memory_test(alloc_fail_test_2);
}

static int alloc_fail_3(void) {
  return memory_test(alloc_fail_test_3);
}

/** URUCHAMIANIE TESTÓW **/

typedef struct {
  char const *name;
  int (*function)(void);
} test_list_t;

#define TEST(t) {#t, t}

static const test_list_t test_list[] = {
  TEST(empty_struct),
  TEST(no_forward),
  TEST(wrong_arguments),
  TEST(malicious_arguments),
  TEST(breaking_struct),
  TEST(long_numbers),
  TEST(many_numbers),
  TEST(copy_arguments),
  TEST(two_structs),
  TEST(delete_null),
  TEST(persistent_results),
  TEST(forward_overwrite),
  TEST(remove_forward),
  TEST(simple_reverse),
  TEST(various_ops),
  TEST(many_ops),
  TEST(very_long),
  TEST(many_remove),
  TEST(add_remove),
  TEST(twelve_digits),
  TEST(cycle),
  TEST(sort),
  TEST(get_reverse),
  TEST(alloc_fail_1),
  TEST(alloc_fail_2),
  TEST(alloc_fail_3),
};

static int do_test(int (*function)(void)) {
  int result = function();
  puts(quite_long_magic_string);
  return result;
}

int main(int argc, char *argv[]) {
  if (argc == 2)
    for (size_t i = 0; i < SIZE(test_list); ++i)
      if (strcmp(argv[1], test_list[i].name) == 0)
        return do_test(test_list[i].function);

  fprintf(stderr, "Użycie:\n%s nazwa_testu\n", argv[0]);
  return WRONG_TEST;
}
