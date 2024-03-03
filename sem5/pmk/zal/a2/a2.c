/*
 * Zadanie A2 – sterowanie LED-ami za pomocą pojedynczego i podwójnego
 *              kliknięcia akcelerometru
 *
 * Pojedyncze kliknięcie powoduje włączenie czerwonej diody na 3 sekundy, a
 * jeśli jest już włączona, to przedłuża czas jej świecenia o kolejne 3 sekundy
 *
 * Podwójne kliknięcie powoduje włączenie zielonej diody na 3 sekundy, a jeśli
 * jest już włączona, to przedłuża czas jej świecenia o kolejne 3 sekundy
 *
 * Na ocenę maksymalną:
 * - komunikacja po I2C za pomocą przerwań
 * - zgłaszanie zdarzenia pojedynczego lub podwójnego kliknięcia za pomocą
 *   przerwania zewnętrznego
 * - odmierzanie czasu za pomocą przerwania licznika
 */

#include <gpio.h>
#include <stddef.h>
#include <stm32.h>

#include "uniexp.h"

#include "led_rgb.h"
#include "i2c.h"

#define PERIOD 3000

#define CLICK_THS 0x05 /* wartość graniczna przyspieszenia uznawanego za kliknięcie */
#define CLICK_TLE 0x80 /* czas między kliknięciami, aby uznać je za podwójne */
#define CLICK_LAT 0x80 /* czas między kliknięciem a zarejestrowaniem tego przez urządzenie */
#define CLICK_WIN 0x80 /* czas przez jaki oczekujemy na nadejście drugiego kliknięcia */

/** @brief Obsługuje kliknięcie akcelerometru.
 * Na podstawie wartości CLICK_SRC podanej w @p arg włącza odpowiednią diodę.
 * @param arg - wskaźnik na wartość rejestru CLICK_SRC
 */
static void handle_click(void *arg);

/** @brief Konfiguruje akcelerometr LIS35DE. */
static void lis35de_setup(void);

/** @brief Konfiguruje licznik TIM3. */
static void tim3_setup(void);

void
EXTI1_IRQHandler(void)
{
	uint32_t pr = EXTI->PR;
	EXTI->PR = pr;
	if (pr & (1 << 1))
		i2c_read(LIS35DE_ADDR, LIS35DE_CLICK_SRC, handle_click);
}

void
handle_click(void *arg)
{
	uint8_t src = *(uint8_t *)arg;
	uint32_t stop = (TIM3->CNT - 1 + PERIOD) % PERIOD;

	/* pojedyncze kliknięcie na dowolnej osi */
	if (src & 0b00010101) {
		TIM3->CCR1 = stop;
		led_turn(LED_RED, LED_ON);
	}

	/* podwójne kliknięcie podwójne kliknięcie na dowolnej osi */
	if (src & 0b00101010) {
		TIM3->CCR2 = stop;
		led_turn(LED_GREEN, LED_ON);
	}
}

void
lis35de_setup(void)
{
	/* włączamy akcelerometr i wszyskie osie */
	i2c_write(LIS35DE_ADDR, LIS35DE_CTRL1, 0b01000111, NULL);

	/* przerwania kliknięcia na Int1 (PA1) */
	i2c_write(LIS35DE_ADDR, LIS35DE_CTRL3, 0b00000111, NULL);

	/* pojedyncze i podwójne kliknięcie na wszystkich osiach */
	i2c_write(LIS35DE_ADDR, LIS35DE_CLICK_CFG, 0b00111111, NULL);

	/* wartość graniczna przyspieszenia uznawanego za kliknięcie */
	i2c_write(LIS35DE_ADDR, LIS35DE_CLICK_THSY_X, CLICK_THS << 4 | CLICK_THS, NULL);
	i2c_write(LIS35DE_ADDR, LIS35DE_CLICK_THSZ,           0 << 4 | CLICK_THS, NULL);

	/* czasy między kliknięciami, aby uznać je za podwójne */
	i2c_write(LIS35DE_ADDR, LIS35DE_CLICK_TIME_LIMIT, CLICK_TLE, NULL);
	i2c_write(LIS35DE_ADDR, LIS35DE_CLICK_LATENCY, CLICK_LAT, NULL);
	i2c_write(LIS35DE_ADDR, LIS35DE_CLICK_WINDOW, CLICK_WIN, NULL);
}

void
tim3_setup(void)
{
	TIM3->CR1 = 0;
	TIM3->PSC = 50000 / 4 - 1; /* (50MHz / 4) / (50'000 / 4) = 1KHz */
	TIM3->ARR = PERIOD - 1;    /* 1KHz / 3'000 = 1/3Hz */
	TIM3->EGR = TIM_EGR_UG;

	TIM3->CCR1 = 0;
	TIM3->CCR2 = 0;

	TIM3->SR = ~(TIM_SR_CC1IF | TIM_SR_CC2IF);
	TIM3->DIER = TIM_DIER_CC1IE | TIM_DIER_CC2IE;

	NVIC_EnableIRQ(TIM3_IRQn);

	TIM3->CR1 |= TIM_CR1_ARPE | TIM_CR1_CEN;
}

void
TIM3_IRQHandler(void)
{
	uint32_t it_status = TIM3->SR & TIM3->DIER;
	TIM3->SR &= ~it_status;

	if (it_status & TIM_SR_CC1IF)
		led_turn(LED_RED, LED_OFF);
	if (it_status & TIM_SR_CC2IF)
		led_turn(LED_GREEN, LED_OFF);
}

int
main(void)
{
	/* włączamy odpowiednie zegary */
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN
	             |  RCC_AHB1ENR_GPIOBEN;
	RCC->APB1ENR |= RCC_APB1ENR_I2C1EN
	             |  RCC_APB1ENR_TIM3EN;
	RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;
	__NOP();

	led_setup(LED_RED);
	led_setup(LED_GREEN);
	i2c_setup();
	lis35de_setup();
	tim3_setup();

	while (1) {
		__WFI();
	}
}
