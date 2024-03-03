#include <gpio.h>
#include <stm32.h>

/* makra */
#define USART_Mode_Rx_Tx         (USART_CR1_RE \
                                | USART_CR1_TE)
#define USART_Enable              USART_CR1_UE
#define USART_WordLength_8b       0x0000
#define USART_WordLength_9b       USART_CR1_M
#define USART_Parity_No           0x0000
#define USART_Parity_Even         USART_CR1_PCE
#define USART_Parity_Odd         (USART_CR1_PCE \
                                | USART_CR1_PS)
#define USART_StopBits_1          0x0000
#define USART_StopBits_0_5        0x1000
#define USART_StopBits_2          0x2000
#define USART_StopBits_1_5        0x3000
#define USART_FlowControl_None    0x0000
#define USART_FlowControl_RTS     USART_CR3_RTSE
#define USART_FlowControl_CTS     USART_CR3_CTSE
#define HSI_HZ                    16000000U
#define PCLK1_HZ                  HSI_HZ
#define BAUD                      9600U

/* typy wyliczeniowe*/
enum LED_OPERATION { LED_OFF = 0, LED_ON, LED_TOGGLE };
enum LED_TYPE { LED_RED = 0, LED_GREEN, LED_BLUE, LED_GREEN2, LED_LAST };
enum BTN_TYPE { BTN_USER = 0, BTN_LEFT, BTN_RIGHT, BTN_UP, BTN_DOWN,
                BTN_ACTION, BTN_MODE, BTN_LAST };

/* struktury */
typedef struct {
	int           pin;
	GPIO_TypeDef *gpio;
	char         *name;
	int           state;
} Button;

typedef struct {
	int           pin;
	GPIO_TypeDef *gpio;
	int           state;
} LightEmittingDiode;

/* deklaracje funkcji */
static void check_btn(enum BTN_TYPE btn);
static void configure_leds(void);
static void configure_usart(void);
static void handle_input(void);
static void led_make_op(enum LED_TYPE led, enum LED_OPERATION op);
static void output_push(const char *str);
static void run(void);
static void setup(void);

/* deklaracje zmiennych globalnych */
static char input[3];
static struct {
	char buf[1024];
	unsigned first, last;
	unsigned size;
} output;
static Button buttons[] = {
	[BTN_USER  ] = { 13, GPIOC, "USER", 1 },
	[BTN_LEFT  ] = {  3, GPIOB, "LEFT", 1 },
	[BTN_RIGHT ] = {  4, GPIOB, "RIGHT",1 },
	[BTN_UP    ] = {  5, GPIOB, "UP",   1 },
	[BTN_DOWN  ] = {  6, GPIOB, "DOWN", 1 },
	[BTN_ACTION] = { 10, GPIOB, "FIRE", 1 },
	[BTN_MODE  ] = {  0, GPIOA, "MODE", 0 },
};

static LightEmittingDiode leds[] = {
	[LED_RED   ] = { 6, GPIOA, 0 },
	[LED_GREEN ] = { 7, GPIOA, 0 },
	[LED_BLUE  ] = { 0, GPIOB, 0 },
	[LED_GREEN2] = { 5, GPIOA, 0 },
};

/* definicje funkcji */
void
check_btn(enum BTN_TYPE btn_type)
{
	int state;
	Button *btn = &buttons[btn_type];

	if (btn_type < 0 || btn_type >= BTN_LAST)
		return;

	state = (btn->gpio->IDR >> btn->pin) & 1;

	if (btn->state == state)
		return;
	btn->state = state;
	output_push(btn->name);
	output_push(" ");
	output_push(state ? "RELEASED"
	                  : (btn_type == BTN_MODE ? "PRESET" : "PRESSED"));
	output_push("\r\n");
}

void
configure_leds(void)
{
	enum LED_TYPE led;

	for (led = LED_RED; led < LED_LAST; led++)
		led_make_op(led, LED_OFF);

	for (led = LED_RED; led < LED_LAST; led++) {
		GPIOoutConfigure(
			leds[led].gpio,
			leds[led].pin,
			GPIO_OType_PP,
			GPIO_Low_Speed,
			GPIO_PuPd_NOPULL
		);
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
handle_input(void)
{
	enum LED_TYPE lt;
	enum LED_OPERATION op;

	if (input[0] != 'L')
		return;

	switch (input[1]) {
	case 'R': { lt = LED_RED;    } break;
	case 'G': { lt = LED_GREEN;  } break;
	case 'B': { lt = LED_BLUE;   } break;
	case 'g': { lt = LED_GREEN2; } break;
	default: return;
	}

	switch (input[2]) {
	case '0': { op = LED_OFF;    } break;
	case '1': { op = LED_ON;     } break;
	case 'T': { op = LED_TOGGLE; } break;
	default: return;
	}

	led_make_op(lt, op);
}

void
led_make_op(enum LED_TYPE lt, enum LED_OPERATION op)
{
	LightEmittingDiode *led = &leds[lt];
	int state;

	if (lt < 0 || lt >= LED_LAST)
		return;

	led->state = op == LED_TOGGLE ? led->state ^ 1
	           : op == LED_ON;
	state = lt == LED_GREEN2 ? led->state : !led->state;
	led->gpio->BSRR = 1 << (led->pin + !state * 16);
}

void
output_push(const char *str)
{
	while (*str) {
		output.buf[output.last] = *str++;
		output.last = (output.last + 1) & (sizeof(output.buf) - 1);
		output.size++;

		if (output.size > sizeof(output.buf)) {
			output.size--;
			output.first = (output.first + 1)
			             & (sizeof(output.buf) - 1);
		}
	}
}

void
run(void)
{
	enum BTN_TYPE btn;
	while (1) {
		/* czy odebrano znak */
		if (USART2->SR & USART_SR_RXNE) {
			input[0] = input[1];
			input[1] = input[2];
			input[2] = USART2->DR;

			handle_input();
		}

		for (btn = BTN_USER; btn < BTN_LAST; btn++)
			check_btn(btn);
		
		/* czy można wysłać znak */
		if (output.size) {
			if (USART2->SR & USART_SR_TXE) {
				USART2->DR = output.buf[output.first];
				output.first = (output.first + 1)
					     & (sizeof(output.buf) - 1);
				output.size--;
			}
		}
	}
}

void
setup(void)
{
	RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN | 
	                RCC_AHB1ENR_GPIOBEN |
	                RCC_AHB1ENR_GPIOCEN;
	RCC->APB1ENR |= RCC_APB1ENR_USART2EN;

	configure_leds();
	configure_usart();

	/* konfiguracja rejestrów CR1, CR2, CR3 */
	USART2->CR1 = USART_Mode_Rx_Tx | USART_WordLength_8b | USART_Parity_No;
	USART2->CR2 = USART_StopBits_1;
	USART2->CR3 = USART_FlowControl_None;

	/* konfiguracja prędkości transmisji */
	USART2->BRR = (PCLK1_HZ + (BAUD / 2U)) / BAUD;

	/* uaktywnienie interfejsu */
	USART2->CR1 |= USART_Enable;

	output.first = output.last = output.size = 0;
}

int
main(void)
{
	setup();
	run();
	return 0;
}
