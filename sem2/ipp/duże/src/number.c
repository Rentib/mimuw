/** \file
 * Implementacja kilku funkcji do manipulacji numerami.
 *
 * \author Stanisław Bitner <sb438247@students.mimuw.edu.pl>
 * \copyright Informacje o licencji i prawach autorskich są w pliku LICENSE.
 * \date 2022
 */

#include <stddef.h>
#include <stdlib.h>

#include "number.h"

int
chartoidx(int c)
{
  switch (c) {
  case '\0': return -1;
  case '*': return 10;
  case '#': return 11;
  default : return c - '0';
  }
}

int
isnumdigit(int c)
{
  return ('0' <= c && c <= '9') || c == '*' || c == '#';
}

char *
numcat(const char *num1, const char *num2, size_t len)
{
  char *res;
  size_t idx = 0;
  if (!(res = malloc(sizeof(char) * (len + 1))))
    return NULL;
  while (*num1)
    res[idx++] = *num1++;
  while (*num2)
    res[idx++] = *num2++;
  res[len] = 0;
  return res;
}

int
numcmp(const char *num1, const char *num2)
{
  if (!num1)
    return num2 ? +1 : 0;
  if (!num2)
    return num1 ? -1 : 0;
  while (*num1 && *num2 && *num1 == *num2) {
    num1++;
    num2++;
  }
  return chartoidx(*num1) - chartoidx(*num2);
}

size_t
numlen(const char *num)
{
  size_t len = 0;
  if (!num)
    return 0;
  while (isnumdigit(num[len]))
    len++;
  return num[len] ? 0 : len;
}
