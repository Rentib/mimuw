#include <gpio.h>
#include <stdint.h>
#include <stm32.h>

#include "btn.h"

typedef struct {
	GPIO_TypeDef *gpio;
	uint32_t pin;
	const char *name;
} Button;

static Button buttons[] = {
	[BTN_USER  ] = { GPIOC, 13, "USER"  },
	[BTN_LEFT  ] = { GPIOB,  3, "LEFT"  },
	[BTN_RIGHT ] = { GPIOB,  4, "RIGHT" },
	[BTN_UP    ] = { GPIOB,  5, "UP"    },
	[BTN_DOWN  ] = { GPIOB,  6, "DOWN"  },
	[BTN_ACTION] = { GPIOB, 10, "FIRE"  },
	[BTN_MODE  ] = { GPIOA,  0, "MODE"  },
};

void
btn_configure(enum btn_type bt)
{
	Button *btn = &buttons[bt];

	GPIOinConfigure(
		btn->gpio,
		btn->pin,
		GPIO_PuPd_UP,
		EXTI_Mode_Interrupt,
		EXTI_Trigger_Rising_Falling
	);

	EXTI->PR = 0x1UL << btn->pin;
}

const char *
btn_get_name(enum btn_type bt)
{
	return buttons[bt].name;
}

uint32_t
btn_get_pin(enum btn_type bt)
{
	return buttons[bt].pin;
}

enum btn_state
btn_get_state(enum btn_type bt)
{
        return (buttons[bt].gpio->IDR >> buttons[bt].pin & 1) ^
               (bt == BTN_MODE);
}
