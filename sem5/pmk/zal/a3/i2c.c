#include <gpio.h>
#include <stddef.h>
#include <stdint.h>

#include "i2c.h"

#define PCLK1_MHZ                 16
#define I2C_SPEED_HZ              100000U
#define READ_QUEUE_SIZE           64

/** \enum i2c_mode_t
 * Tryb pracy I2C.
 */
enum i2c_mode_t {
	I2C_MODE_READ = 0,
	I2C_MODE_WRITE,
	I2C_MODE_LAST,
};

/** \struct read_arg
 * Argumenty dla odczytu.
 */
struct read_arg {
	/** @{ */
	uint8_t addr; /**< Adres urządzenia. */
	uint8_t reg;  /**< Numer rejestru. */
	/** @} */
};

/** @brief Próbuje zainicjować odczyt.
 * Jeśli @p arg jest różny od NULL, dodaje go do kolejki odczytów.
 * Jeśli I2C nie jest zajęte i kolejka nie jest pusta, inicjuje odczyt z
 * kolejki i usuwa go z niej.
 */
static void try_init_read(const struct read_arg *arg);

/** \struct i2c_state
 * Stan I2C.
 */
static volatile struct {
	/** @{ */
	uint8_t addr;                 /**< Adres urządzenia. */
	uint8_t reg;                  /**< Numer rejestru. */
	uint8_t data;                 /**< Dane do zapisu lub odczytane. */
	enum i2c_mode_t mode;         /**< Tryb pracy. */
	i2c_callback_t read_callback; /**< Callback dla odczytu. */
	unsigned cnt;                 /**< Licznik bajtów do zapisu lub odczytu. */
	unsigned set_addr:1;          /**< Czy ustawiono adres. */
	unsigned busy:1;              /**< Czy I2C jest zajęte. */
	/** @} */
} i2c_state;

/** \struct read_queue
 * Kolejka odczytów.
 */
static struct {
	/** @{ */
	struct read_arg arg[READ_QUEUE_SIZE]; /**< Argumenty odczytów. */
	size_t front;                         /**< Indeks pierwszego elementu. */
	size_t back;                          /**< Indeks pierwszego wolnego elementu. */
	size_t len;                           /**< Liczba elementów. */
	/** @} */
} read_queue;

void
i2c_init(void)
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

	I2C1->CR1 = 0;
	I2C1->CR2 = PCLK1_MHZ;
	I2C1->CCR = (PCLK1_MHZ * 1000000) / (I2C_SPEED_HZ << 1);
	I2C1->TRISE = PCLK1_MHZ + 1;

	NVIC_EnableIRQ(I2C1_EV_IRQn);

	I2C1->CR1 |= I2C_CR1_PE;

	i2c_state.read_callback = NULL;
	i2c_state.busy = 0;

	read_queue.front = 0;
	read_queue.back = 0;
	read_queue.len = 0;
}

void
i2c_read(uint8_t addr, uint8_t reg)
{
	try_init_read(&(struct read_arg){addr, reg});
}

void
i2c_set_read_callback(i2c_callback_t cb)
{
	i2c_state.read_callback = cb;
}

void
i2c_write(uint8_t addr, uint8_t reg, uint8_t val)
{
	while (i2c_state.busy);
	i2c_state.busy = 1;
	i2c_state.addr = addr;
	i2c_state.reg = reg;
	i2c_state.data = val;
	i2c_state.mode = I2C_MODE_WRITE;
	i2c_state.cnt = 1;
	i2c_state.set_addr = 0;

	I2C1->CR2 |= I2C_CR2_ITEVTEN;
	I2C1->CR1 |= I2C_CR1_START;

	while (i2c_state.busy);
}

void
I2C1_EV_IRQHandler(void)
{
	uint32_t it_status = I2C1->SR1;
	(void)I2C1->SR2;

	if (!i2c_state.set_addr) {
		if (it_status & I2C_SR1_SB) {
			I2C1->DR = i2c_state.addr << 1;
		}
		if (it_status & I2C_SR1_ADDR) {
			i2c_state.set_addr = 1;
			I2C1->CR2 |= I2C_CR2_ITBUFEN;
			I2C1->DR = i2c_state.reg;
		}
		return;
	}

	if (i2c_state.mode == I2C_MODE_WRITE) {
		goto handle_write;
	} else if (i2c_state.mode == I2C_MODE_READ) {
		if (it_status == I2C_SR1_TXE)
			return;
		goto handle_read;
	}
handle_write:
	if (it_status & I2C_SR1_TXE) {
		if (i2c_state.cnt) {
			I2C1->DR = i2c_state.data;
			if (--i2c_state.cnt == 0)
				I2C1->CR2 &= ~I2C_CR2_ITBUFEN;
		}
	}
	if (it_status & I2C_SR1_BTF) {
		if (!i2c_state.cnt) {
			I2C1->CR2 &= ~I2C_CR2_ITEVTEN;
			I2C1->CR1 |= I2C_CR1_STOP;
			i2c_state.busy = 0;
			try_init_read(NULL);
		}
	}
	return;
handle_read:
	if (it_status & I2C_SR1_BTF) {
		I2C1->CR1 |= I2C_CR1_START;
	}
	if (it_status & I2C_SR1_SB) {
		I2C1->DR = i2c_state.addr << 1 | 1;
		I2C1->CR1 &= ~I2C_CR1_ACK;
	}
	if (it_status & I2C_SR1_ADDR) {
		I2C1->CR1 |= I2C_CR1_STOP;
	}
	if (it_status & I2C_SR1_RXNE) {
		I2C1->CR2 &= ~(I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN);
		i2c_state.data = I2C1->DR;
		if (i2c_state.read_callback)
			i2c_state.read_callback(i2c_state.data);
		i2c_state.busy = 0;

		try_init_read(NULL);
	}
}

void
try_init_read(const struct read_arg *arg)
{
	if (arg) {
		read_queue.arg[read_queue.back++] = *arg;
		read_queue.back %= READ_QUEUE_SIZE;
		read_queue.len++;
	}

	if (i2c_state.busy || !read_queue.len)
		return;
	i2c_state.busy = 1;

	i2c_state.addr = read_queue.arg[read_queue.front].addr;
	i2c_state.reg = read_queue.arg[read_queue.front].reg;
	read_queue.front = (read_queue.front + 1) % READ_QUEUE_SIZE;
	read_queue.len--;

	i2c_state.mode = I2C_MODE_READ;
	i2c_state.cnt = 1;
	i2c_state.set_addr = 0;

	I2C1->CR2 |= I2C_CR2_ITEVTEN;
	I2C1->CR1 |= I2C_CR1_START;
}
