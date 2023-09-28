/* See LICENSE for copyright and license details. */

#ifndef RADIO_RECEIVER_H_
#define RADIO_RECEIVER_H_

#include <stddef.h>
#include <stdint.h>

typedef struct Receiver Receiver;

Receiver *receiver_create(uint16_t port, char *addr);
void receiver_destroy(Receiver *receiver);

/* global variables */
extern char *discover_addr;
extern uint16_t ctrl_port;
extern uint16_t ui_port;
extern size_t bsize;
extern size_t rtime;
extern char *name;

#endif  // RADIO_RECEIVER_H_
