#ifndef LED_RGB_H_
#define LED_RGB_H_

/** @brief Typ diody LED. */
enum led_type {
	LED_RED,
	LED_GREEN,
	LED_BLUE,
	LED_LAST,
};

/** @brief Typ operacji na diodzie LED. */
enum led_operation {
	LED_OFF,
	LED_ON,
	LED_TOGGLE,
};

/** @brief Konfiguruje diody LED.
 * Najpierw wyłącza diodę, a następnie ustawia pin jako wyjście.
 * @param led_type - Typ diody LED.
 */
void led_setup(enum led_type);

/** @brief Steruje diodami LED.
 * @param led_type - Typ diody LED.
 * @param led_operation - Operacja na diodzie LED.
 */
void led_turn(enum led_type, enum led_operation);

#endif /* LED_RGB_H_ */
