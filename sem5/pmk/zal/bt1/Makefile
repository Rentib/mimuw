CC       = arm-none-eabi-gcc
OBJCOPY  = arm-none-eabi-objcopy
FLAGS    = -mthumb -mcpu=cortex-m4
CPPFLAGS = -DSTM32F411xE
CFLAGS   = $(FLAGS) -Wall -Wextra -Wno-unused-function -g \
					 -O2 -ffunction-sections -fdata-sections \
					 -I/opt/arm/stm32/inc \
					 -I/opt/arm/stm32/CMSIS/Include \
					 -I/opt/arm/stm32/CMSIS/Device/ST/STM32F4xx/Include
LDFLAGS  = $(FLAGS) -Wl,--gc-sections -nostartfiles \
					 -L/opt/arm/stm32/lds -Tstm32f411re.lds

vpath %.c /opt/arm/stm32/src

OBJECTS = bt1.o led.o btn.o startup_stm32.o gpio.o
TARGET  = bt1

.SECONDARY: $(TARGET).elf $(OBJECTS)

all: $(TARGET).bin

%.elf : $(OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@

%.bin : %.elf
	$(OBJCOPY) $< $@ -O binary

clean :
	rm -f *.bin *.elf *.hex *.d *.o *.bak *~
