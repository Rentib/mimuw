/*
 * Zaimplementować sterowanie kursorem myszy na ekranie komputera za pomocą
 * wychyleń płytki
 *
 * Położenie płytki jest ustalane za pomocą akcelerometru
 *
 * Odczyt wyniku z akcelerometru jest inicjowany za pomocą licznika
 *
 * Informacja o zmianie położenia kursura jest wysyłana do komputera za pomocą
 * interfejsu szeregowego (UART)
 *
 * Po stronie komputera działa prosty skrypt zmieniający położenie kursora
 * myszy na podstawie danych odczytanych z portu szeregowego
 *
 * Komunikacja po I2 C za pomocą przerwań
 *
 * Licznik zgłaszający przerwanie
 *
 * Obsługa UART-u za pomocą DMA i związanego z nim przerwania
 */

#include <gpio.h>
#include <stm32.h>
#include <stddef.h>

#include "i2c.h"
#include "util.h"

#define PCLK1_HZ                  16000000U
#define BAUD                      9600U

/** @struct dma_buffer
 * Struktura przechowująca bufor używany przez DMA.
 */
struct dma_buffer {
	/** @{ */
	unsigned cur; /**< Obecnie używany bufor; */
        char buf[2][1024]; /**< Dostępne bufory:
                                @p buf[cur]    - bufor na który piszemy;
                                @p buf[cur^1]  - bufor używany przez dma; */
        size_t len; /*<< Długość buf[cur]. */
	/** @} */
};

/** @brief Inicjuje DMA. */
static void dma_init(void);

/** @brief Próbuje zainicjować wysyłanie.
 * Dokłada wiadomość @p msg do kolejki jeżeli nie jest ona równa NULL.
 * Następnie sprawdza, czy DMA jest dostępne. Jeśli jest oraz kolejka jest
 * niepusta, rozpoczyna wysyłanie.
 * @param[in] msg - wskaźnik na zakończoną bajtem zerowym wiadomość do wysłania
 *                  lub równy NULL.
 */
static void dma_try_to_send(const char *msg);

/** @brief Inicjuje licznik. */
static void timer_init(void);

/** @brief Inicjuje USART. */
static void usart_init(void);

/** @brief Wysyła wartość przyspieszenia.
 * Wysyła przez USART wartość przyspieszenia @p x odczytaną z akcelerometru.
 * @param[in] x - wysyłana wartość.
 */
static void output_acceleration(uint8_t x);

struct dma_buffer buffer; /**< Bufor używany przez DMA. */

void
dma_init(void)
{
	DMA1_Stream6->CR = 4U << 25       |
	                   DMA_SxCR_PL_1  |
	                   DMA_SxCR_MINC  |
	                   DMA_SxCR_DIR_0 |
	                   DMA_SxCR_TCIE;

	/* ustawiamy peryferyjny adres DMA na adres USART2 */
	DMA1_Stream6->PAR = (uint32_t)&USART2->DR;

	DMA1->HIFCR = DMA_HIFCR_CTCIF6;
	NVIC_EnableIRQ(DMA1_Stream6_IRQn); /* włącz jego przerwania */
}

void
DMA1_Stream6_IRQHandler(void)
{
	uint32_t isr = DMA1->HISR;
	if (isr & DMA_HISR_TCIF6) {
		DMA1->HIFCR = DMA_HIFCR_CTCIF6;
		dma_try_to_send(NULL);
	}
}

void
dma_try_to_send(const char *msg)
{
	if (msg != NULL) {
		while (*msg)
			buffer.buf[buffer.cur][buffer.len++] = *msg++;
	}

	if (buffer.len == 0
	|| (DMA1_Stream6->CR & DMA_SxCR_EN) || (DMA1->HISR & DMA_HISR_TCIF6))
		return;

	DMA1_Stream6->M0AR = (uint32_t)buffer.buf[buffer.cur];
	DMA1_Stream6->NDTR = buffer.len;
	DMA1_Stream6->CR |= DMA_SxCR_EN;

	buffer.cur ^= 1;
	buffer.len = 0;
}

void
timer_init(void)
{
	TIM3->CR1 = TIM_CR1_URS;
	TIM3->PSC = 64000 - 1;
	TIM3->ARR = 10 - 1;
	TIM3->EGR = TIM_EGR_UG;

	TIM3->SR = ~(TIM_SR_UIF);
	TIM3->DIER = TIM_DIER_UIE;

	NVIC_EnableIRQ(TIM3_IRQn);

	TIM3->CR1 |= TIM_CR1_ARPE | TIM_CR1_CEN;
}

void
usart_init(void)
{
	GPIOafConfigure(
		GPIOA,
		2,
		GPIO_OType_PP,
		GPIO_Fast_Speed,
		GPIO_PuPd_NOPULL,
		GPIO_AF_USART2
	);
	GPIOafConfigure(
		GPIOA,
		3,
		GPIO_OType_PP,
		GPIO_Fast_Speed,
		GPIO_PuPd_UP,
		GPIO_AF_USART2
	);

	USART2->CR1 = USART_CR1_TE;
	USART2->CR2 = 0;
	USART2->CR3 = USART_CR3_DMAT;
	USART2->BRR = (PCLK1_HZ + (BAUD / 2U)) / BAUD;

	USART2->CR1 |= USART_CR1_UE;
}

void
output_acceleration(uint8_t x)
{
	static unsigned cnt = 0;
	char buf[32], *p = buf;
	int16_t val = x;

	/* konwersja z komplementu 2 */
	val = val & 0x80 ? ~((~val + 1) & 0xFF) : val;

	p = print_dec(p, val);
	if (cnt ^= 1) {
		p = print_str(p, "\r\n");
	} else {
		p = print_str(p, " ");
	}
	*p = '\0';

	dma_try_to_send(buf);
}

void
TIM3_IRQHandler(void)
{
	uint32_t it_status = TIM3->SR & TIM3->DIER;
	TIM3->SR &= ~it_status;

	if (it_status & TIM_SR_UIF) {
		i2c_read(LIS35DE_ADDR, LIS35DE_OUT_X_H);
		i2c_read(LIS35DE_ADDR, LIS35DE_OUT_Y_H);
	}
}

int
main(void)
{
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN |
	                RCC_AHB1ENR_GPIOBEN |
	                RCC_AHB1ENR_DMA1EN;
	RCC->APB1ENR |= RCC_APB1ENR_USART2EN |
	                RCC_APB1ENR_I2C1EN |
	                RCC_APB1ENR_TIM3EN;
	RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;

	dma_init();
	i2c_init();
	i2c_set_read_callback(output_acceleration);
	timer_init();
	usart_init();

	i2c_write(LIS35DE_ADDR, LIS35DE_CTRL1, 0b01000111);

	while (1) {
		__WFI();
	}
}
