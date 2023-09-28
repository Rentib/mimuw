/* See LICENSE file for copyright and license details. */

#ifndef RADIO_UTIL_H_
#define RADIO_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#define DATA_LENGTH_LIMIT 65507

#ifndef htonll
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define htonll(x) (((uint64_t)htonl(x)) << 32 | htonl(x >> 32))
#else
#define htonll(x) (x)
#endif
#endif

#ifndef ntohll
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define ntohll(x) (((uint64_t)ntohl(x)) << 32 | ntohl(x >> 32))
#else
#define ntohll(x) (x)
#endif
#endif

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

/**@brief Audio packet structure.
 * Structure use to store audio packet information.
 */
typedef struct AudioPacket AudioPacket;
struct AudioPacket {
  /**@{*/
  uint64_t session_id;     /**< Session ID. */
  uint64_t first_byte_num; /**< Number of first byte of the audio data. */
  char *audio_data;        /**< Pointer to the audio data. */
  /**@}*/
} __attribute__((packed));

/**@brief Die.
 * Prints an error message to stderr and exits with code EXIT_FAILURE.
 * @param fmt Format string.
 * @param ... Format arguments.
 */
void die(const char *fmt, ...);

/**@brief Allocates memory.
 * Allocates memory for an array of @p nmemb elements of @p size bytes each and
 * initializes all bytes in the allocated storage to zero.
 * @param nmemb Number of elements.
 * @param size Size of each element.
 * @return On success, returns a pointer to beginning of the allocated memory.
 * To avoid a memory leak, the returned pointer must be deallocated with free().
 * On failuer, calls die().
 */
void *ecalloc(size_t nmemb, size_t size);

/**@brief Converts a string to a positive, unsigned long.
 * Interprets the string @p nptr as a positive, unsigned long in the given @p
 * base.
 * @param nptr String to interpret.
 * @param base Base of the interpreted integer.
 * @return On success, returns the converted value.
 * On failure, calls die().
 */
unsigned long estrtoul(const char *nptr, int base);

/**@brief Interprets a string as a port number.
 * Interprets the string @p string as a port number.
 * @param string String to interpret.
 * @return On success, returns the port number.
 * On failure, calls die().
 */
uint16_t strtoport(const char *string);

/**@brief Warn.
 * Prints an error message to stderr.
 * @param fmt Format string.
 * @param ... Format arguments.
 */
void warn(const char *fmt, ...);

/**@brief Program name.
 * Program name passed to main().
 */
extern char *argv0;

#endif  // RADIO_UTIL_H_
