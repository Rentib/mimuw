#ifndef I2C_H_
#define I2C_H_

#include <stdint.h>

/** @brief Typ callbacku dla operacji I2C.
 * Funkcja wywoływana po odczytaniu lub zapisaniu danych poprzez I2C.
 * @param[in] arg - odczyatny bajt lub NULL w przypadku zapisu.
 */
typedef void (*i2c_callback_t)(void *);

/** @brief Inicjalizacja magistrali I2C. */
void i2c_setup(void);

/** @brief Asynchroniczny odczyt bajtu.
 * Zleca odczyt bajtu z urządzenia o adresie @p addr z rejestru @p reg.
 * @param[in] addr - adres urządzenia.
 * @param[in] reg - adres rejestru.
 * @param[in] cb - funkcja wywoływana po odczytaniu bajtu.
 */
void i2c_read(uint8_t addr, uint8_t reg, i2c_callback_t cb);

/** @brief Asynchroniczny zapis bajtu.
 * Zleca zapis bajtu @p data do urządzenia o adresie @p addr do rejestru @p reg.
 * @param[in] addr - adres urządzenia.
 * @param[in] reg - adres rejestru.
 * @param[in] data - zapisywany bajt.
 * @param[in] cb - funkcja wywoływana po zapisaniu bajtu.
 */
void i2c_write(uint8_t addr, uint8_t reg, uint8_t data, i2c_callback_t cb);

#endif /* I2C_H_ */
