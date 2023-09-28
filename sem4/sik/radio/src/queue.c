/* See LICENSE file for copyright and license details. */

#include "queue.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

Queue *
queue_create(void)
{
  Queue *q = ecalloc(1, sizeof(Queue));

  q->size = 0;
  q->capacity = 1;
  q->front = 0;
  q->back = 0;
  q->data = ecalloc(q->capacity, sizeof(void *));

  return q;
}

void
queue_destroy(Queue *q)
{
  if (!q) return;
  free(q->data);
  free(q);
}

void
queue_destroy_full(Queue *q, void (*destroy)(void *item))
{
  if (!q || !destroy) return;
  while (q->size) destroy(queue_pop(q));
  queue_destroy(q);
}

void *
queue_pop(Queue *q)
{
  if (!q || !q->size) return NULL;

  void *item = q->data[q->front++];
  q->front &= q->capacity - 1;
  q->size--;

  return item;
}

int
queue_push(Queue *q, void *item)
{
  if (!q) return 0;

  if (q->size == q->capacity) {
    void **tmp = realloc(q->data, (q->size << 1) * sizeof(void *));
    if (!tmp) die("realloc:");
    q->data = tmp;
    q->capacity <<= 1;

    size_t len = q->size - q->front;
    memmove(&q->data[q->capacity - len], &q->data[q->front],
            len * sizeof(void *));
    q->front = q->capacity - len;
  }

  q->data[q->back++] = item;
  q->back &= q->capacity - 1;
  q->size++;

  return 1;
}
