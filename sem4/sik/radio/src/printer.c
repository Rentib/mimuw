/* See LICENSE file for copyright and license details. */

#include "printer.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>

#include "bitset.h"
#include "cyclicbuffer.h"
#include "util.h"

enum {
  RUNNING_FLAG = 1,
  READY_FLAG = 2,
  NOT_EMPTY_FLAG = 4,
};

struct Printer {
  /**@{*/
  unsigned flags;       /**< flags; */
  mtx_t mtx;            /**< mutex; */
  cnd_t cnd;            /**< condition variable; */
  thrd_t thrd;          /**< thread; */
  CyclicBuffer *cb;     /**< cyclic buffer; */
  CyclicBuffer *mb;     /**< missing buffer - TODO: change to bitset; */
  size_t bsize, psize;  /**< buffer size and packet size; */
  uint64_t first, last; /**< first and last byte. */
  /**@}*/
};

static int printer_thread(void *arg);

Printer *
printer_create(void)
{
  Printer *p = ecalloc(1, sizeof(Printer));
  p->flags = RUNNING_FLAG;
  p->cb = cb_create(1, 1);
  p->mb = cb_create(1, 1);

  if (mtx_init(&p->mtx, mtx_plain) != thrd_success) die("mtx_init:");
  if (cnd_init(&p->cnd) != thrd_success) die("cnd_init:");
  if (thrd_create(&p->thrd, printer_thread, p) != thrd_success)
    die("thrd_create:");
  return p;
}

void
printer_destroy(Printer *p)
{
  if (p == NULL) return;
  if (mtx_lock(&p->mtx) != thrd_success) die("mtx_lock:");
  p->flags &= ~RUNNING_FLAG;
  if (cnd_signal(&p->cnd) != thrd_success) die("cnd_signal:");
  if (mtx_unlock(&p->mtx) != thrd_success) die("mtx_unlock:");
  if (thrd_join(p->thrd, NULL) != thrd_success) die("thrd_join:");
  mtx_destroy(&p->mtx);
  cb_destroy(p->cb);
  cb_destroy(p->mb);
  free(p);
}

void
printer_print(Printer *p, char *restrict msg, uint64_t byte)
{
  if (p == NULL) return;
  if (mtx_lock(&p->mtx) != thrd_success) die("mtx_lock:");
  if (byte < p->first) goto unlock;  // skip very old messages

  char x = 1;
  cb_push(p->cb, msg, byte - p->first);
  cb_push(p->mb, &x, (byte - p->first) / p->psize);
  p->last = MAX(p->last, byte);
  if (p->last > p->cb->cap) p->first = MAX(p->first, p->last - p->cb->cap);

  // p->first will not change until this condition is met
  if (byte + p->psize > p->first + p->bsize * 3 / 4) p->flags |= READY_FLAG;

  p->flags |= NOT_EMPTY_FLAG;
  if (p->flags & READY_FLAG)
    if (cnd_signal(&p->cnd) != thrd_success) die("cnd_signal:");

unlock:
  if (mtx_unlock(&p->mtx) != thrd_success) die("mtx_unlock:");
}

void
printer_set_buffer(Printer *p, size_t bsize, size_t psize, uint64_t byte)
{
  if (p == NULL) return;
  if (mtx_lock(&p->mtx) != thrd_success) die("mtx_lock:");
  p->flags &= ~READY_FLAG;
  p->flags &= ~NOT_EMPTY_FLAG;
  cb_resize(p->cb, bsize / psize * psize, psize);
  cb_resize(p->mb, bsize / psize, 1);
  p->bsize = bsize, p->psize = psize;
  p->first = byte, p->last = byte;
  if (mtx_unlock(&p->mtx) != thrd_success) die("mtx_unlock:");
}

int
printer_thread(void *arg)
{
  Printer *p = arg;
  unsigned flags = p->flags;

  while (flags & RUNNING_FLAG) {
    if (mtx_lock(&p->mtx) != thrd_success) die("mtx_lock:");

    flags = p->flags;
    while ((flags & RUNNING_FLAG) && (~flags & (READY_FLAG | NOT_EMPTY_FLAG))) {
      if (cnd_wait(&p->cnd, &p->mtx) != thrd_success) die("cnd_wait:");
      flags = p->flags;
    }

    if (~flags & (RUNNING_FLAG | READY_FLAG)) goto unlock;

    char *present = cb_pop(p->mb);

    p->first += p->psize;

    if (!*present) {
      p->flags &= ~READY_FLAG;

      // NOTE:
      // Aby uzyskać rozwiązanie zgodne z treścią wystarczy ustawić
      // STATEMENT_MODE na 1. To rozwiązanie okazuje się być jednak
      // nieskuteczne -- zacinanie występuje nawet, gdy w losowy sposób traci
      // się 1/1000 pakietów.

#ifndef STATEMENT_MODE
#define STATEMENT_MODE 0
#endif
#if STATEMENT_MODE
      if (mtx_unlock(&p->mtx) != thrd_success) die("mtx_unlock:");
      printer_set_buffer(p, p->bsize, p->psize, p->last + p->psize);
#endif

      // NOTE:
      // Zgodnie z odpowiedzią na forum, lekko modyfikuję rozpoczynanie
      // odtwarzania -- nie czyszczę bufora, a jedynie ustawiam go na niegotowy
      // i przesuwam jego pierwszy pakiet. Następnie czekam z wypisywaniem tak
      // jak jest w treści, czyli dopóki bufor nie osiągnie rozmiaru 3/4. Bufor
      // może mieć rozmiar 3/4 w tym warunku, ale na podstawie dokonywanych
      // testów, okazuje się, że warto poczekać na kolejną paczkę, nawet jeśli
      // bufor ma już wymagany rozmiar.
      // Rozwiązanie takie okazuje się być lepsze od opisanego w treści,
      // ponieważ nie są tracone dane, które program zdążył odebrać.

      goto unlock;
    }

    char *msg = cb_pop(p->cb);

    if (p->cb->size == 0) p->flags &= ~NOT_EMPTY_FLAG;

    fwrite(msg, p->psize, 1, stdout);

  unlock:
    if (mtx_unlock(&p->mtx) != thrd_success) die("mtx_unlock:");
  }

  return thrd_success;
}
