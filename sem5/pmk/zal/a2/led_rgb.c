#include <gpio.h>

#include "led_rgb.h"

struct led {
	GPIO_TypeDef *gpio;
	uint32_t pin;
	unsigned state:1;
};

static struct led leds[LED_LAST] = {
	[LED_RED]   = { GPIOA, 6, 0 },
	[LED_GREEN] = { GPIOA, 7, 0 },
	[LED_BLUE]  = { GPIOB, 0, 0 },
};

void
led_setup(enum led_type lt)
{
	struct led *led = &leds[lt];

	led_turn(lt, LED_OFF);
	GPIOoutConfigure(
		led->gpio,
		led->pin,
		GPIO_OType_PP,
		GPIO_Low_Speed,
		GPIO_PuPd_NOPULL
	);
}

void
led_turn(enum led_type lt, enum led_operation lo)
{
	struct led *led = &leds[lt];

	led->state = lo == LED_ON     ? 1
	           : lo == LED_OFF    ? 0
	           : lo == LED_TOGGLE ? led->state ^ 1 : 0;
	led->gpio->BSRR = 1 << (led->pin + (led->state << 4));
}
