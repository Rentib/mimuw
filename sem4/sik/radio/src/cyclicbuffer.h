/* See LICENSE file for copyright and license details. */

#ifndef RADIO_CYCLICBUFFER_H_
#define RADIO_CYCLICBUFFER_H_

#include <stddef.h>
#include <stdint.h>

typedef struct {
  /**@{*/
  size_t cap;        /**< capacity; */
  size_t item;       /**< item size; */
  size_t size;       /**< number of items; */
  char *buf;         /**< buffer; */
  size_t begin, end; /**< begin and end of buffer. */
  /**@}*/
} CyclicBuffer;

CyclicBuffer *cb_create(size_t cap, size_t item);
void cb_destroy(CyclicBuffer *cb);
char *cb_pop(CyclicBuffer *cb);
void cb_push(CyclicBuffer *cb, const char *msg, uint64_t offset);
void cb_resize(CyclicBuffer *cb, size_t cap, size_t item);

#endif  // RADIO_CYCLICBUFFER_H_
