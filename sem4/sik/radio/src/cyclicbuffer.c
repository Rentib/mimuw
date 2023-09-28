/* See LICENSE file for copyright and license details. */

#include "cyclicbuffer.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

CyclicBuffer *
cb_create(size_t cap, size_t item)
{
  CyclicBuffer *cb = ecalloc(1, sizeof(CyclicBuffer));
  cb->cap = cap;
  cb->item = item;
  cb->size = 0;
  cb->buf = ecalloc(cap, sizeof(char));
  cb->begin = 0, cb->end = 0;
  return cb;
}

void
cb_destroy(CyclicBuffer *cb)
{
  if (cb == NULL) return;
  free(cb->buf);
  free(cb);
}

char *
cb_pop(CyclicBuffer *cb)
{
  if (cb == NULL) return NULL;
  char *res = cb->buf + cb->begin;

  cb->begin = (cb->begin + cb->item) % cb->cap;
  --cb->size;

  return res;
}

void
cb_push(CyclicBuffer *cb, const char *msg, uint64_t offset)
{
  if (cb == NULL) return;

  offset += cb->item;  // offset should be from last byte not first byte

  // bytes between begin and end
  uint64_t space = (cb->end - cb->begin + cb->cap) % cb->cap;

  // delayed packet
  if (offset < space) {
    memcpy(cb->buf + (cb->begin + offset) % cb->cap, msg, cb->item);
    return;
  }

  // make offset relative to cb->end
  offset -= space;

  cb->size = MIN(cb->size + offset / cb->item, cb->cap / cb->item);

  if (offset >= cb->cap) {
    memset(cb->buf, 0, cb->cap);
    cb->begin = 0, cb->end = 0;
    offset = cb->item;
  } else {
    if (offset <= cb->cap - cb->end) {
      memset(cb->buf + cb->end, 0, offset);
    } else {
      memset(cb->buf + cb->end, 0, cb->cap - cb->end);
      memset(cb->buf, 0, offset - (cb->cap - cb->end));
    }
  }

  cb->end = (cb->end + offset - cb->item) % cb->cap;
  memcpy(cb->buf + cb->end, msg, cb->item);
  cb->end = (cb->end + cb->item) % cb->cap;

  if (cb->size * cb->item == cb->cap) cb->begin = cb->end;
}

void
cb_resize(CyclicBuffer *cb, size_t cap, size_t item)
{
  if (cb == NULL) return;
  cb->cap = cap;
  cb->item = item;
  cb->size = 0;
  cb->buf = realloc(cb->buf, cap);
  if (cb->buf == NULL) die("realloc:");
  memset(cb->buf + cb->end, 0, cb->cap - cb->end);
  cb->begin = 0, cb->end = 0;
}
