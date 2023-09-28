/* See LICENSE file for copyright and license details. */

#ifndef RADIO_BITSET_H_
#define RADIO_BITSET_H_

#include <stddef.h>

/**@brief bitset structure
 * The structure represents a sequence of bitst.
 */
typedef struct Bitset Bitset;

/**@brief creates a bitset
 * Creates a bitset of size @p size.
 * @param size - the size of the bitset
 * @return A pointer to the newly created bitset.
 */
Bitset *bitset_create(size_t size);

/**@brief destroys a bitset
 * Destroys a bitset previously created with bitset_create().
 * @param bs - pointer to the bitset to destroy
 */
void bitset_destroy(Bitset *bs);

/**@brief accesses specific bit
 * Returns the value of the bit at position @p pos (counting from 0).
 * @param bs - pointer to the bitset
 * @param pos - position of the bit to return (counting from 0)
 * @return The value of the bit at position @p pos.
 */
int bitset_test(Bitset *bs, size_t pos);

/**@brief returns the number of bits set to 1
 * Returns the number of bits set to 1 in the bitset.
 * @param bs - pointer to the bitset
 * @return The number of bits set to 1 in the bitset.
 */
size_t bitset_count(Bitset *bs);

/**@brief returns the number of bits that the bitset holds
 * Returns the size of the bitset.
 * @param bs - pointer to the bitset
 * @return The size of the bitset.
 */
size_t bitset_size(Bitset *bs);

/**@ sets the bit to 1
 * Sets the bit at position @p pos to 1.
 * @param bs - pointer to the bitset
 * @param pos - the position (counting from 0) of the bit to set
 */
void bitset_set(Bitset *bs, size_t pos);

/**@ sets the bit to 0
 * Sets the bit at position @p pos to 0.
 * @param bs - pointer to the bitset
 * @param pos - the position (counting from 0) of the bit to reset
 */
void bitset_reset(Bitset *bs, size_t pos);

#endif  // RADIO_BITSET_H_
