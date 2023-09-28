/* See LICENSE file for copyright and license details. */

#ifndef RADIO_QUEUE_H_
#define RADIO_QUEUE_H_

#include <stddef.h>

struct Queue {
  size_t size;
  size_t capacity;
  size_t front;
  size_t back;
  void **data;
};

typedef struct Queue Queue;

struct Queue *queue_create(void);
void queue_destroy(Queue *queue);
void queue_destroy_full(Queue *queue, void (*destroy)(void *item));
void *queue_pop(Queue *queue);
int queue_push(Queue *queue, void *item);
size_t queue_size(Queue *queue);

#endif  // RADIO_QUEUE_H_
