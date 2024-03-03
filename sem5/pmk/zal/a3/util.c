#include "util.h"

char *
print_dec(char *dst, int x)
{
	static char rbuf[10];
	int n = 0;

	if (x < 0) {
		*dst++ = '-';
		x = -x;
	}

	do {
		rbuf[n++] = '0' + x % 10;
		x /= 10;
	} while (x);
	
	while (n)
		*dst++ = rbuf[--n];

	return dst;
}

char *
print_str(char *dst, const char *str)
{
	while (*str)
		*dst++ = *str++;
	return dst;
}
