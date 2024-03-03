#include <gpio.h>
#include <irq.h>
#include <stm32.h>

/* makra */
#define PCLK1_HZ                  16000000U
#define BAUD                      9600U
#define BTN_HANDLER() \
	do { \
		volatile uint32_t pr = EXTI->PR; \
		EXTI->PR = pr; \
		handle_button(pr); \
	} while (0)


/* typy wyliczeniowe*/
enum BTN_TYPE { BTN_USER = 0, BTN_LEFT, BTN_RIGHT, BTN_UP, BTN_DOWN,
                BTN_ACTION, BTN_MODE, BTN_LAST };

/* struktury */
typedef struct {
	int           pin;
	GPIO_TypeDef *gpio;
	char         *name;
} Button;

/* deklaracje funkcji */
static void configure_buttons(void);
static void configure_usart(void);
static void enable_interrupts(void);
static void handle_button(uint32_t pr);
static void setup(void);
static uint32_t str_len(const char * restrict str);
static void str_cat(char * restrict dest, const char *restrict src);

/* deklaracje zmiennych globalnych */
static Button buttons[] = {
	[BTN_USER  ] = { 13, GPIOC, "USER"  },
	[BTN_LEFT  ] = {  3, GPIOB, "LEFT"  },
	[BTN_RIGHT ] = {  4, GPIOB, "RIGHT" },
	[BTN_UP    ] = {  5, GPIOB, "UP"    },
	[BTN_DOWN  ] = {  6, GPIOB, "DOWN"  },
	[BTN_ACTION] = { 10, GPIOB, "FIRE"  },
	[BTN_MODE  ] = {  0, GPIOA, "MODE"  },
};

static struct {
	uint8_t  cur;
	char     buf[2][1024];
} buffer;

/* definicje funkcji */
void EXTI0_IRQHandler(void)     { BTN_HANDLER(); }
void EXTI3_IRQHandler(void)     { BTN_HANDLER(); }
void EXTI4_IRQHandler(void)     { BTN_HANDLER(); }
void EXTI9_5_IRQHandler(void)   { BTN_HANDLER(); }
void EXTI15_10_IRQHandler(void) { BTN_HANDLER(); }

void
configure_buttons(void)
{
	enum BTN_TYPE btn;

	for (btn = 0; btn < BTN_LAST; btn++) {
		GPIOinConfigure(
			buttons[btn].gpio,
			buttons[btn].pin,
			GPIO_PuPd_UP,
			EXTI_Mode_Interrupt,
			EXTI_Trigger_Rising_Falling
		);

		EXTI->PR = (0x1UL << (buttons[btn].pin));
	}
}

void
configure_usart(void)
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
}

void
DMA1_Stream6_IRQHandler(void)
{
	uint32_t isr = DMA1->HISR;
	irq_level_t irq_level;
	if (~isr & DMA_HISR_TCIF6)
		return;
	DMA1->HIFCR = DMA_HIFCR_CTCIF6;

	irq_level = IRQprotect(MIDDLE_IRQ_PRIO);
	if (buffer.buf[buffer.cur][0] == '\0')
		return;

	if (!(DMA1_Stream6->CR & DMA_SxCR_EN)
	&&  !(DMA1->HISR & DMA_HISR_TCIF6)) {
		DMA1_Stream6->M0AR = (uint32_t)buffer.buf[buffer.cur];
		DMA1_Stream6->NDTR = str_len(buffer.buf[buffer.cur]);
		DMA1_Stream6->CR |= DMA_SxCR_EN;
		buffer.cur ^= 1;
		buffer.buf[buffer.cur][0] = '\0';
	}
	IRQunprotect(irq_level);
}

void
enable_interrupts(void)
{
	unsigned i;
	const int interrupts[] = { EXTI0_IRQn, EXTI3_IRQn, EXTI4_IRQn,
	                           EXTI9_5_IRQn, EXTI15_10_IRQn,
	                           DMA1_Stream6_IRQn };
	for (i = 0; i < sizeof(interrupts) / sizeof(*interrupts); ++i)
		NVIC_EnableIRQ(interrupts[i]);
}

void
handle_button(uint32_t pr)
{
	int state;
	char buf[32];
	irq_level_t irq_level;
	enum BTN_TYPE btn;
	for (btn = 0; btn < BTN_LAST; btn++) {
		if (!(pr & (0x1UL << buttons[btn].pin)))
			continue;

		state = (buttons[btn].gpio->IDR >> buttons[btn].pin) & 1;
		if (btn == BTN_MODE)
			state ^= 1;

		buf[0] = '\0';
		str_cat(buf, buttons[btn].name);
		str_cat(buf, " ");
		str_cat(buf, state ? "RELEASED"
				   : (btn == BTN_MODE ? "PRESET" : "PRESSED"));
		str_cat(buf, "\r\n");

		irq_level = IRQprotect(MIDDLE_IRQ_PRIO);
		str_cat(buffer.buf[buffer.cur], buf);
		if (!(DMA1_Stream6->CR & DMA_SxCR_EN)
		&&  !(DMA1->HISR & DMA_HISR_TCIF6)) {
			DMA1_Stream6->M0AR = (uint32_t)buffer.buf[buffer.cur];
			DMA1_Stream6->NDTR = str_len(buffer.buf[buffer.cur]);
			DMA1_Stream6->CR |= DMA_SxCR_EN;
			buffer.cur ^= 1;
			buffer.buf[buffer.cur][0] = '\0';
		}
		IRQunprotect(irq_level);
	}
}

void
setup(void)
{
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN | 
	                RCC_AHB1ENR_GPIOBEN |
	                RCC_AHB1ENR_GPIOCEN |
			RCC_AHB1ENR_DMA1EN;
	RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
	RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;
	
	configure_buttons();
	configure_usart();

	/* konfiguracja rejestrów CR1, CR2, CR3 */
	USART2->CR1 = USART_CR1_RE | USART_CR1_TE; /* TODO: może wystarczy TE? */
	USART2->CR2 = 0;
	USART2->CR3 = USART_CR3_DMAT;

	/* konfiguracja prędkości transmisji */
	USART2->BRR = (PCLK1_HZ + (BAUD / 2U)) / BAUD;

	/* konfiguracja nadawczego strumienia DMA */
	DMA1_Stream6->CR = 4U << 25       |
	                   DMA_SxCR_PL_1  |
	                   DMA_SxCR_MINC  |
	                   DMA_SxCR_DIR_0 |
	                   DMA_SxCR_TCIE;
	DMA1_Stream6->PAR = (uint32_t)&USART2->DR;

	/* aktywowanie przerwań DMA */
	DMA1->HIFCR = DMA_HIFCR_CFEIF6;

	enable_interrupts();

	/* uaktywnienie interfejsu */
	USART2->CR1 |= USART_CR1_UE;

	buffer.cur   = 0;
	**buffer.buf = 0;
	**buffer.buf = 0;
}

uint32_t
str_len(const char * restrict str)
{
	uint32_t len = 0;
	while (str[len])
		len++;
	return len;
}

void
str_cat(char * restrict dest, const char *restrict src)
{
	char *p = dest + str_len(dest);
	while (*src)
		*p++ = *src++;
	*p = '\0';
}

int
main(void)
{
	setup();
	while (1);
	return 0;
}
