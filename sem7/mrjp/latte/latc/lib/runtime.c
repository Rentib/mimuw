#include <ctype.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void printInt(int n) { printf("%d\n", n); }

void printString(char *s) { printf("%s\n", s); }

int readInt(void)
{
	int n;
	scanf("%d ", &n);
	return n;
}

void error()
{
	fprintf(stderr, "RUNTIME ERROR\n");
	exit(EXIT_FAILURE);
}

char *readString(void)
{
	char *s = NULL;
	size_t len = 1, idx = 0;
	int c;

	if (!(s = malloc(sizeof(*s) * len)))
		goto fail;

	while (1) {
		if (idx == len) {
			len *= 2;
			if (!(s = realloc(s, sizeof(*s) * len)))
				goto fail;
		}
		c = getchar();
		if (c < 0 || isspace(c))
			break;
		s[idx++] = c;
	}

	/* shrink to fit */
	s = realloc(s, sizeof(*s) * idx);
	return s;

fail:
	fprintf(stderr, "RUNTIME ERROR: buy more ram lol\n");
	exit(EXIT_FAILURE);
}

char *__strcat(char *s1, char *s2)
{
	char *s;
	size_t l1 = strlen(s1), l2 = strlen(s2);
	size_t len = l1 + l2 + 1;

	if (!(s = malloc(sizeof(*s) * len))) {
		fprintf(stderr, "RUNTIME ERROR: buy more ram lol\n");
		exit(EXIT_FAILURE);
	}

	if (l1 > 0)
		memcpy((void *)s, (void *)s1, l1);
	if (l2 > 0)
		memcpy((void *)(s + l1), (void *)s2, l2);
	s[len - 1] = '\0';
	return s;
}

void *__class_new(size_t size, void *vtable)
{
	void *res;
	if (!(res = calloc(1, sizeof(*res)))) {
		fprintf(stderr, "RUNTIME ERROR: buy more ram lol\n");
		exit(EXIT_FAILURE);
	}
	*(void **)res = vtable;
	return res;
}
