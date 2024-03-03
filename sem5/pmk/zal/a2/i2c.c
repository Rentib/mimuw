#include <gpio.h>
#include <stddef.h>
#include <stm32.h>

#include "i2c.h"
#include "stm32f411xe.h"

#define PCLK1_MHZ                 16
#define I2C_SPEED_HZ              100000U
#define QUEUE_CAPACITY            64

/** @brief Typ operacji I2C. */
enum i2c_operation {
	I2C_READ,
	I2C_WRITE,
};

struct queue_arg {
	/**@{*/
	enum i2c_operation op; /**< Typ operacji. */
	uint8_t addr;      /**< Adres urządzenia. */
	uint8_t reg;       /**< Adres rejestru. */
	uint8_t data;      /**< Dane do zapisu. */
	i2c_callback_t cb; /**< Funkcja wywoływana po zakończeniu operacji. */
	/**@}*/
};

struct queue {
	/**@{*/
	struct queue_arg arg[QUEUE_CAPACITY]; /**< Zakolejkowane operacje. */
	size_t front; /**< Indeks pierwszego elementu w kolejce. */
	size_t back;  /**< Indeks pierwszego wolnego miejsca w kolejce. */
	size_t size;  /**< Liczba elementów w kolejce. */
	/**@}*/
};

/** @brief Próbuje zainicjować operację I2C.
 * Jeśli argument jest niezerowy, dodaje go do kolejki.
 * Jeśli kolejka nie jest pusta i magistrala jest wolna, inicjuje operację.
 * @param[in] arg - argument operacji lub NULL.
 */
static void i2c_try_init(const struct queue_arg *arg);

static struct {
	/**@{*/
	struct queue queue;   /**< Kolejka operacji. */
	struct queue_arg arg; /**< Aktualnie wykonywana operacja. */
	unsigned busy:1;      /**< Czy magistrala jest zajęta. */
	unsigned set_addr:1;  /**< Czy został wysłany adres urządzenia. */
	/**@}*/
} i2c_state;

void
i2c_setup(void)
{
	GPIOafConfigure(
		GPIOB,
		8,
		GPIO_OType_OD,
		GPIO_Low_Speed,
		GPIO_PuPd_NOPULL,
		GPIO_AF_I2C1
	);
	GPIOafConfigure(
		GPIOB,
		9,
		GPIO_OType_OD,
		GPIO_Low_Speed,
		GPIO_PuPd_NOPULL,
		GPIO_AF_I2C1
	);

	GPIOinConfigure(
		GPIOA,
		1,
		GPIO_PuPd_UP,
		EXTI_Mode_Interrupt,
		EXTI_Trigger_Rising_Falling
	);
	EXTI->PR = 1 << 1;

	I2C1->CR1 = 0;
	I2C1->CR2 = PCLK1_MHZ;
	I2C1->CCR = (PCLK1_MHZ * 1000000) / (I2C_SPEED_HZ << 1);
	I2C1->TRISE = PCLK1_MHZ + 1;

	NVIC_EnableIRQ(EXTI1_IRQn);
	NVIC_EnableIRQ(I2C1_EV_IRQn);

	I2C1->CR1 |= I2C_CR1_PE;

	i2c_state.queue.front = 0;
	i2c_state.queue.back = 0;
	i2c_state.queue.size = 0;

	i2c_state.busy = 0;
}

void
i2c_read(uint8_t addr, uint8_t reg, i2c_callback_t cb)
{
	i2c_try_init(&(struct queue_arg){
		.op = I2C_READ,
		.addr = addr,
		.reg = reg,
		.cb = cb,
	});
}

void
i2c_write(uint8_t addr, uint8_t reg, uint8_t data, i2c_callback_t cb)
{
	i2c_try_init(&(struct queue_arg){
		.op = I2C_WRITE,
		.addr = addr,
		.reg = reg,
		.data = data,
		.cb = cb,
	});
}

void
i2c_try_init(const struct queue_arg *arg)
{
	struct queue *q = &i2c_state.queue;

	if (arg) {
		q->arg[q->back++] = *arg;
		q->back %= QUEUE_CAPACITY;
		q->size++;
	}

	if (i2c_state.busy || !i2c_state.queue.size)
		return;
	i2c_state.busy = 1;

	i2c_state.arg = q->arg[q->front++];
	q->front %= QUEUE_CAPACITY;
	q->size--;

	i2c_state.set_addr = 0;

	I2C1->CR2 |= I2C_CR2_ITEVTEN;
	I2C1->CR1 |= I2C_CR1_START;
}

void
I2C1_EV_IRQHandler(void)
{
	uint32_t it_status = I2C1->SR1;
	(void)I2C1->SR2;

	if (!i2c_state.set_addr) {
		if (it_status & I2C_SR1_SB) {
			I2C1->DR = i2c_state.arg.addr << 1 | 0;
		}
		if (it_status & I2C_SR1_ADDR) {
			i2c_state.set_addr = 1;
			I2C1->CR2 |= I2C_CR2_ITBUFEN;
			I2C1->DR = i2c_state.arg.reg;
		}
		return;
	}

	if (i2c_state.arg.op == I2C_READ)
		goto handle_read;
	if (i2c_state.arg.op == I2C_WRITE)
		goto handle_write;
	return;

handle_read:
	if (it_status & I2C_SR1_BTF) {
		I2C1->CR1 |= I2C_CR1_START;
	}
	if (it_status & I2C_SR1_SB) {
		I2C1->DR = i2c_state.arg.addr << 1 | 1;
		I2C1->CR1 &= ~I2C_CR1_ACK;
	}
	if (it_status & I2C_SR1_ADDR) {
		I2C1->CR1 |= I2C_CR1_STOP;
	}
	if (it_status & I2C_SR1_RXNE) {
		I2C1->CR2 &= ~(I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN);
		i2c_state.arg.data = I2C1->DR;

		if (i2c_state.arg.cb)
			i2c_state.arg.cb(&i2c_state.arg.data);

		i2c_state.busy = 0;
		i2c_try_init(NULL);
	}

	return;
handle_write:
	if (it_status & I2C_SR1_TXE) {
		I2C1->DR = i2c_state.arg.data;
		I2C1->CR2 &= ~I2C_CR2_ITBUFEN;
	}
	if (it_status & I2C_SR1_BTF) {
		I2C1->CR2 &= ~I2C_CR2_ITEVTEN;
		I2C1->CR1 |= I2C_CR1_STOP;

		if (i2c_state.arg.cb)
			i2c_state.arg.cb(NULL);

		i2c_state.busy = 0;
		i2c_try_init(NULL);
	}

	return;
}
