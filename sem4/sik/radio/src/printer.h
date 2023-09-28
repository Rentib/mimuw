/* See LICENSE file for copyright and license details. */

#ifndef RADIO_PRINTER_H_
#define RADIO_PRINTER_H_

#include <stddef.h>
#include <stdint.h>

/**@brief Printer structure.
 * Structure use to store printer information.
 */
typedef struct Printer Printer;

/**@brief Creates a printer.
 * Creates a printer structure.
 * @return On success, returns the pointer to the newly created printer
 * structure. To aviod a memory leak, the returned printer must be destroyed
 * with printer_destroy().
 * On failure, prints an error message to stderr and exits.
 */
Printer *printer_create(void);

/**@brief Destroys a printer.
 * Destroys a printer structure previously created with printer_create().
 * If @p p is NULL, the function does nothing.
 * The behavior is undefined if the value of @p p does not equal a value
 * returned earlier by printer_create().
 * The behavior is undefined if the printer refered to by @p p has already been
 * destroyed, that is, printer_destroy() has already been called with pointer
 * @p p as the argument and no calls to printer_create() resulted in a pointer
 * equal to @p p afterwards.
 * @param p Pointer to the printer to destroy.
 */
void printer_destroy(Printer *p);

/**@brief Prints a message.
 * Enqueues a message to be printed by the printer.
 * If @p p is NULL, the function does nothing.
 * The behavior is undefined it length of @p msg is not equal to @p psize set
 * earlier by printer_set_buffer().
 * @param p Pointer to the printer.
 * @param msg Pointer to the message to print.
 * @param byte Number of first byte of the message.
 */
void printer_print(Printer *p, char *msg, uint64_t byte);

/**@brief Sets the printer buffer.
 * Sets the printer buffer size and the printer buffer page size.
 * If @p p is NULL, the function does nothing.
 * @param p Pointer to the printer.
 * @param bsize Size of the printer buffer.
 * @param psize Size of the printer buffer item.
 * @param byte Number of first byte of the message.
 */
void printer_set_buffer(Printer *p, size_t bsize, size_t psize, uint64_t byte);

#endif  // RADIO_PRINTER_H_
