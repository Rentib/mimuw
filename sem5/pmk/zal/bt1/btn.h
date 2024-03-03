#ifndef BT1_BTN_H_
#define BT1_BTN_H_

#include <stdint.h>

enum btn_type {
	BTN_USER,
	BTN_LEFT,
	BTN_RIGHT,
	BTN_UP,
	BTN_DOWN,
	BTN_ACTION,
	BTN_MODE,
	BTN_LAST,
};

enum btn_state {
	BTN_STATE_PRESSED,
	BTN_STATE_RELEASED,
};

void btn_configure(enum btn_type);
const char *btn_get_name(enum btn_type);
uint32_t btn_get_pin(enum btn_type);
enum btn_state btn_get_state(enum btn_type);

#endif /* BT1_BTN_H_ */
