#ifndef BT1_LED_H_
#define BT1_LED_H_

enum led_type {
	LED_RED = 0,
	LED_GREEN,
	LED_BLUE,
	LED_GREEN2,
	LED_LAST,
};

enum led_operation {
	LED_OFF,
	LED_ON,
	LED_TOGGLE,
};

void led_configure(enum led_type);
void led_make_operation(enum led_type, enum led_operation);

#endif /* BT1_LED_H_ */
