/*
 * Zaimplementować komunikację Bluetooth z komputerem
 * osobistym lub telefonem
 *
 * Zaimplementować odbieranie komunikatów sterujących
 * diodami świecącymi
 *
 * Zaimplementować wysyłanie komunikatów o użyciu dżojstika
 * (każde wciśnięcie i puszczenie przycisku)
 *
 * Napisać bardzo prostą aplikację demonstracyjną na telefon
 * komórkowy lub komputer osobisty
 *
 * Na ocenę maksymalną:
 * - obsługa transmisji między mikrokontrolerem a modułem BT za
 *   pomocą DMA i przerwań
 * - wykrywanie użycia dżojstika za pomocą przerwań zewnętrznych
 * - likwidowanie drgania styków przycisków z wykorzystaniem
 *   licznika i jego przerwania
 */

#include <gpio.h>
#include <stddef.h>
#include <stm32.h>

#include "btn.h"
#include "led.h"

#define PCLK1_HZ                  16000000U
#define BAUD                      38400U

#define BTN_HANDLER()                            \
	do {                                     \
		volatile uint32_t pr = EXTI->PR; \
		EXTI->PR = pr;                   \
		button_interrupt(pr);            \
	} while (0)

struct dma_buffer {
	unsigned cur;
        char buf[2][1024];
        size_t len;
};

static void button_interrupt(uint32_t pr);
static void dma_try_to_send(const char *msg);

static struct dma_buffer buffer;
static uint8_t btn_bouncing[BTN_LAST] = {0};
static uint8_t input_buf[3];

void EXTI0_IRQHandler(void) { BTN_HANDLER(); }
void EXTI3_IRQHandler(void) { BTN_HANDLER(); }
void EXTI4_IRQHandler(void) { BTN_HANDLER(); }
void EXTI9_5_IRQHandler(void) { BTN_HANDLER(); }
void EXTI15_10_IRQHandler(void) { BTN_HANDLER(); }

void
button_interrupt(uint32_t pr)
{
	enum btn_type btn;

	for (btn = 0; btn < BTN_LAST; btn++) {
		if (!(pr & (0x1UL << btn_get_pin(btn)))
		||  btn_bouncing[btn])
			continue;

		btn_bouncing[btn] = 2; // za 2 tiki zegara się wyzeruje

		dma_try_to_send(btn_get_name(btn));
		if (btn_get_state(btn) == BTN_STATE_PRESSED)
			dma_try_to_send(" pressed\r\n");
		else
			dma_try_to_send(" released\r\n");
	}
}

void
handle_input(void)
{
	enum led_type lt;
	enum led_operation lo;
	if (input_buf[0] != 'l')
		goto shift;
	switch (input_buf[1]) {
	case 'r': { lt = LED_RED;    } break;
	case 'g': { lt = LED_GREEN;  } break;
	case 'b': { lt = LED_BLUE;   } break;
	case 'G': { lt = LED_GREEN2; } break;
	default:  goto shift;
	}
	switch (input_buf[2]) {
	case '0': { lo = LED_OFF;    } break;
	case '1': { lo = LED_ON;     } break;
	case 't': { lo = LED_TOGGLE; } break;
	default:  goto shift;
	}

	led_make_operation(lt, lo);
shift:
	input_buf[0] = input_buf[1];
	input_buf[1] = input_buf[2];
	DMA2_Stream5->M0AR = (uint32_t)&input_buf[2];
	DMA2_Stream5->NDTR = 1;
	DMA2_Stream5->CR |= DMA_SxCR_EN;
}

void
DMA2_Stream5_IRQHandler(void)
{
	uint32_t isr = DMA2->HISR;
	if (~isr & DMA_HISR_TCIF5)
		return;
	DMA2->HIFCR = DMA_HIFCR_CTCIF5;

	handle_input();
}

void
DMA2_Stream7_IRQHandler(void)
{
	uint32_t isr = DMA2->HISR;
	if (~isr & DMA_HISR_TCIF7)
		return;
	DMA2->HIFCR = DMA_HIFCR_CTCIF7;
	dma_try_to_send(NULL);
}

void
dma_try_to_send(const char *msg)
{
	if (msg != NULL) {
		while (*msg)
			buffer.buf[buffer.cur][buffer.len++] = *msg++;
	}

	if (buffer.len == 0
	|| (DMA2_Stream7->CR & DMA_SxCR_EN) || (DMA2->HISR & DMA_HISR_TCIF7))
		return;

	DMA2_Stream7->M0AR = (uint32_t)buffer.buf[buffer.cur];
	DMA2_Stream7->NDTR = buffer.len;
	DMA2_Stream7->CR |= DMA_SxCR_EN;

	buffer.cur ^= 1;
	buffer.len = 0;
}

void
TIM3_IRQHandler(void)
{
	uint32_t it_status = TIM3->SR & TIM3->DIER;
	enum btn_type btn;

	TIM3->SR &= ~it_status;

	if (it_status & TIM_SR_UIF) {
		for (btn = 0; btn < BTN_LAST; btn++)
			btn_bouncing[btn] >>= 1;
	}
}

void
enable_interrupts() {
	unsigned i;
        int intettupts[] = {
	    EXTI0_IRQn,     EXTI3_IRQn,        EXTI4_IRQn,        EXTI9_5_IRQn,
	    EXTI15_10_IRQn, DMA2_Stream5_IRQn, DMA2_Stream7_IRQn, TIM3_IRQn};

        for (i = 0; i < sizeof(intettupts) / sizeof(*intettupts); i++)
                NVIC_EnableIRQ(intettupts[i]);
}

void
tim_init(TIM_TypeDef *TIMx, uint32_t prescaler, uint32_t period)
{
	TIMx->CR1 = TIM_CR1_URS;
	TIMx->PSC = prescaler - 1;
	TIMx->ARR = period - 1;
	TIMx->EGR = TIM_EGR_UG;

	TIMx->SR = ~TIM_SR_UIF;
	TIMx->DIER = TIM_DIER_UIE;
}

int
main(void)
{
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN
	             |  RCC_AHB1ENR_GPIOBEN
	             |  RCC_AHB1ENR_GPIOCEN
	             |  RCC_AHB1ENR_DMA2EN;
	RCC->APB1ENR |= RCC_APB1ENR_TIM3EN;
	RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN
	             |  RCC_APB2ENR_USART1EN;
	__NOP();

	for (enum led_type led = 0; led < LED_LAST; led++)
		led_configure(led);

	for (enum btn_type btn = 0; btn < BTN_LAST; btn++)
		btn_configure(btn);

	// 50MHz / (5000 * 100 / 2) = 10KHz / (100 / 2) = 200Hz
	tim_init(TIM3, 5000, 100 / 2);

	GPIOafConfigure(
		GPIOA,
		9,
		GPIO_OType_PP,
		GPIO_Fast_Speed,
		GPIO_PuPd_NOPULL,
		GPIO_AF_USART1
	);
	GPIOafConfigure(
		GPIOA,
		10,
		GPIO_OType_PP,
		GPIO_Fast_Speed,
		GPIO_PuPd_NOPULL,
		GPIO_AF_USART1
	);

	USART1->CR1 = USART_CR1_TE | USART_CR1_RE;
	USART1->CR2 = 0;
	USART1->CR3 = USART_CR3_DMAT | USART_CR3_DMAR;
	USART1->BRR = (PCLK1_HZ + (BAUD / 2)) / BAUD;

	DMA2_Stream5->CR = 4U << 25
	                 | DMA_SxCR_PL_1
	                 | DMA_SxCR_MINC
	                 | DMA_SxCR_TCIE;
	DMA2_Stream5->PAR = (uint32_t)&USART1->DR;

	DMA2_Stream7->CR = 4U << 25
	                 | DMA_SxCR_PL_1
	                 | DMA_SxCR_MINC
	                 | DMA_SxCR_DIR_0
	                 | DMA_SxCR_TCIE;
	DMA2_Stream7->PAR = (uint32_t)&USART1->DR;

	DMA2->HIFCR = DMA_HIFCR_CTCIF5 | DMA_HIFCR_CTCIF7;

	USART1->CR1 |= USART_CR1_UE;

	buffer.len = 0;
	buffer.cur = 0;

	enable_interrupts();
	handle_input();

	TIM3->CR1 |= (TIM_CR1_ARPE | TIM_CR1_CEN);

        while (1) {
		__WFI();
	}
}

// TODO: fix MODE btn
