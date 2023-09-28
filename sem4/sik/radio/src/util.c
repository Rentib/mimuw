/* See LICENSE file for copyright and license details. */

#include "util.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *argv0;

void
die(const char *fmt, ...)
{
  va_list ap;

  va_start(ap, fmt);
  warn(fmt, ap);
  va_end(ap);

  exit(EXIT_FAILURE);
}

extern void *
ecalloc(size_t nmemb, size_t size)
{
  void *p;
  if (!(p = calloc(nmemb, size))) die("calloc:");
  return p;
}

unsigned long
estrtoul(const char *nptr, int base)
{
  if (strchr(nptr, '-')) die("%s is not a valid positive number", nptr);
  char *endptr;
  errno = 0;
  unsigned long val = strtoul(nptr, &endptr, base);
  if (errno != 0) die("strtoul:");
  if (*endptr != '\0' || !val) die("%s is not a valid positive number", nptr);
  return val;
}

uint16_t
strtoport(const char *string)
{
  unsigned long port = estrtoul(string, 10);
  if (port > UINT16_MAX) die("%s is not a valid port number", string);
  return (uint16_t)port;
}

void
warn(const char *fmt, ...)
{
  va_list ap;

  if (strlen(argv0)) fprintf(stderr, "%s: ", argv0);

  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  if (fmt[0] && fmt[strlen(fmt) - 1] == ':') {
    fputc(' ', stderr);
    perror(NULL);
  } else {
    fputc('\n', stderr);
  }
}
