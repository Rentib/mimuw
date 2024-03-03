#include <gpio.h>
#include <stm32.h>

#include "led.h"

typedef struct {
	GPIO_TypeDef *gpio;
	uint32_t pin;
	int state;
} LightEmittingDiode;

static LightEmittingDiode leds[LED_LAST] = {
	[LED_RED]    = { GPIOA, 6, 0 },
	[LED_GREEN]  = { GPIOA, 7, 0 },
	[LED_BLUE]   = { GPIOB, 0, 0 },
	[LED_GREEN2] = { GPIOA, 5, 0 },
};

void
led_configure(enum led_type lt)
{
	led_make_operation(lt, LED_OFF);
	GPIOoutConfigure(
		leds[lt].gpio,
		leds[lt].pin,
		GPIO_OType_PP,
		GPIO_Low_Speed,
		GPIO_PuPd_NOPULL
	);
}

void
led_make_operation(enum led_type lt, enum led_operation lo)
{
	LightEmittingDiode *led = &leds[lt];

	led->state = lo == LED_ON     ? 1
	           : lo == LED_OFF    ? 0
	           : lo == LED_TOGGLE ? led->state ^ 1 : 0;
        led->gpio->BSRR =
            1 << (led->pin + ((led->state ^ (lt == LED_GREEN2)) << 4));
}
