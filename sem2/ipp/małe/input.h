/* See LICENSE file for copyright and license details. */
#ifndef INPUT_H
#define INPUT_H

#include <inttypes.h>
#include <stdlib.h>

extern uint64_t *maze_key;

void parse_input(size_t *k, size_t **n, size_t **x, size_t **y);

#endif

