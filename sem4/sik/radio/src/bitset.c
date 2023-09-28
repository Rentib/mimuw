/* See LICENSE file for copyright and license details. */

#include "bitset.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

typedef uint64_t chunk_t;
#define BITS_PER_CHUNK (sizeof(chunk_t) * 8)

struct Bitset {
  /**@{*/
  size_t size;     /**< number of bits; */
  size_t nchunks;  /**< number of chunks; */
  chunk_t *chunks; /**< array for storing bits; */
  size_t count;    /**< number of set bits. */
  /**@}*/
};

Bitset *
bitset_create(size_t size)
{
  Bitset *bs = ecalloc(1, sizeof(Bitset));
  bs->size = size;
  bs->nchunks = (bs->size + BITS_PER_CHUNK - 1) / BITS_PER_CHUNK;
  bs->chunks = ecalloc(bs->nchunks, sizeof(chunk_t));
  bs->count = 0;
  return bs;
}

void
bitset_destroy(Bitset *bs)
{
  if (!bs) return;
  free(bs->chunks);
  free(bs);
}

int
bitset_test(Bitset *bs, size_t pos)
{
  const size_t wshift = pos / BITS_PER_CHUNK;
  const size_t offset = pos % BITS_PER_CHUNK;
  return (bs->chunks[wshift] >> offset) & 1;
}

size_t
bitset_count(Bitset *bs)
{
  return bs->count;
}

size_t
bitset_size(Bitset *bs)
{
  return bs->size;
}

void
bitset_set(Bitset *bs, size_t pos)
{
  const size_t wshift = pos / BITS_PER_CHUNK;
  const size_t offset = pos % BITS_PER_CHUNK;
  if (!(bs->chunks[wshift] >> offset & 1)) bs->count++;
  bs->chunks[wshift] |= ((chunk_t)1 << offset);
}

void
bitset_reset(Bitset *bs, size_t pos)
{
  const size_t wshift = pos / BITS_PER_CHUNK;
  const size_t offset = pos % BITS_PER_CHUNK;
  if (bs->chunks[wshift] >> offset & 1) bs->count--;
  bs->chunks[wshift] &= ~((chunk_t)1 << offset);
}
