#ifndef A3_I2C_H_
#define A3_I2C_H_

#include <stdint.h>

#define LIS35DE_ADDR              0x1D
#define LIS35DE_CTRL0             0x1F
#define LIS35DE_CTRL1             (0x01 + LIS35DE_CTRL0)
#define LIS35DE_CTRL2             (0x02 + LIS35DE_CTRL0)
#define LIS35DE_CTRL3             (0x03 + LIS35DE_CTRL0)
#define LIS35DE_CTRL4             (0x04 + LIS35DE_CTRL0)
#define LIS35DE_CTRL5             (0x05 + LIS35DE_CTRL0)
#define LIS35DE_CTRL6             (0x06 + LIS35DE_CTRL0)
#define LIS35DE_CTRL7             (0x07 + LIS35DE_CTRL0)
#define LIS35DE_OUT_X_L           (0x09 + LIS35DE_CTRL0)
#define LIS35DE_OUT_X_H           (0x0A + LIS35DE_CTRL0)
#define LIS35DE_OUT_Y_L           (0x0B + LIS35DE_CTRL0)
#define LIS35DE_OUT_Y_H           (0x0C + LIS35DE_CTRL0)
#define LIS35DE_OUT_Z_L           (0x0D + LIS35DE_CTRL0)
#define LIS35DE_OUT_Z_H           (0x0E + LIS35DE_CTRL0)

/** @brief Typ callbacka dla odczytu.
 * Funkcja wywoływana po odczycie danych z I2C.
 * Argumentem funkcji będzie odczytany bajt.
 */
typedef void (*i2c_callback_t)(uint8_t);

/** @brief Inicjuje I2C. */
void i2c_init(void);

/** @brief Czyta 1 bajt z I2C.
 * Nieblokująco odczytuje jeden bajt z rejestru @p reg adresu @p addr I2C.
 * Po odczycie wywoływana jest funkcja ustawiona przez i2c_set_read_callback()
 * z odczytanym bajtem jako argumentem. Jeśli nie ma ustawionej funkcji, odczyt
 * jest ignorowany.
 * @param[in] addr - adres peryferyjny;
 * @param[in] reg  - numer rejestru.
 * @return Wartość danego rejestru.
 */
void i2c_read(uint8_t addr, uint8_t reg);

/** @brief Ustawia callback dla odczytu.
 * Ustawia funkcję, która będzie wywoływana po odczycie danych z I2C.
 * Argumentem funkcji będzie odczytany bajt.
 * @param[in] cb - wskaźnik na funkcję.
 */
void i2c_set_read_callback(i2c_callback_t cb);

/** @brief Zapisuje 1 bajt do I2c.
 * Zapisuje jeden bajt pod rejestr @p reg adresu @p addr I2c.
 * Ta funkcja jest blokująca.
 * @param[in] addr - adres peryferyjny;
 * @param[in] reg  - numer rejestru.
 */
void i2c_write(uint8_t addr, uint8_t reg, uint8_t val);

#endif /* A3_I2C_H_ */
