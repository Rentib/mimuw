#include <delay.h>
#include <gpio.h>
#include <stm32.h>

enum PMK_LED {
	PMK_LED_RED    = 6, /* GPIOA */
	PMK_LED_GREEN  = 7, /* GPIOA */
	PMK_LED_BLUE   = 0, /* GPIOB */
	PMK_LED_GREEN2 = 5, /* GPIOA */
};

void
PMK_TurnLED(enum PMK_LED led, int state)
{
	switch (led) {
	case PMK_LED_RED: {
		GPIOA->BSRR = 1 << (6 + (state != 0) * 16);
	} break;
	case PMK_LED_GREEN: {
		GPIOA->BSRR = 1 << (7 + (state != 0) * 16);
	} break;
	case PMK_LED_BLUE: {
		GPIOB->BSRR = 1 << (0 + (state != 0) * 16);
	} break;
	case PMK_LED_GREEN2: {
		GPIOA->BSRR = 1 << (5 + (state == 0) * 16);
	} break;
	}
}

int
main(void)
{
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN | 
	                RCC_AHB1ENR_GPIOBEN;
	__NOP();

	PMK_TurnLED(PMK_LED_RED,    0);
	PMK_TurnLED(PMK_LED_GREEN,  0);
	PMK_TurnLED(PMK_LED_BLUE,   0);
	PMK_TurnLED(PMK_LED_GREEN2, 0);

	GPIOoutConfigure(
		GPIOA,
		PMK_LED_RED,
		GPIO_OType_PP,
		GPIO_Low_Speed,
		GPIO_PuPd_NOPULL
	);
	GPIOoutConfigure(
		GPIOA,
		PMK_LED_GREEN,
		GPIO_OType_PP,
		GPIO_Low_Speed,
		GPIO_PuPd_NOPULL
	);
	GPIOoutConfigure(
		GPIOB,
		PMK_LED_BLUE,
		GPIO_OType_PP,
		GPIO_Low_Speed,
		GPIO_PuPd_NOPULL
	);
	GPIOoutConfigure(
		GPIOA,
		PMK_LED_GREEN2,
		GPIO_OType_PP,
		GPIO_Low_Speed,
		GPIO_PuPd_NOPULL
	);

	int delay = 4e6;
	while (1) {
		PMK_TurnLED(PMK_LED_RED, 1);
		Delay(delay);
		PMK_TurnLED(PMK_LED_RED, 0);
		PMK_TurnLED(PMK_LED_GREEN, 1);
		Delay(delay);
		PMK_TurnLED(PMK_LED_GREEN, 0);
		PMK_TurnLED(PMK_LED_BLUE, 1);
		Delay(delay);
		PMK_TurnLED(PMK_LED_BLUE, 0);
		PMK_TurnLED(PMK_LED_GREEN2, 1);
		Delay(delay);
		PMK_TurnLED(PMK_LED_GREEN2, 0);
	}

	return 0;
}
