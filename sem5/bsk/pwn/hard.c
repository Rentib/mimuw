// clang hard.c -o hard
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <alloca.h>

static size_t read_uint() {
    size_t retval;
    if (scanf("%zu", &retval) != 1)
    {
        puts("scanf failed");
        exit(-3);
    }
    return retval;
}

static void readn(char* buf, size_t len) {
    while (len)
    {
        ssize_t num_read;
        if ((num_read = read(STDIN_FILENO, buf, len)) <= 0)
        {
            fprintf(stderr, "Read error :c\n");
            exit(1);
        }

        buf += num_read;
        len -= num_read;
    }
}

void decrypt() {
    puts("How long is the data?");
    size_t data_len = read_uint();
    char* data = alloca(data_len);
    printf("Gib data: ");
    readn(data, data_len);
    putchar('\n');

    puts("How long is the key?");
    size_t key_len = read_uint();
    char* key = alloca(key_len);
    printf("Gib key: ");
    readn(key, key_len);
    putchar('\n');

    puts("Here's your decrypted data:");
    for (size_t i = 0; i < key_len; ++i)
    {
        data[i] ^= key[i];
        putchar(data[i]);
    }
}

int do_magic(char* number) {
    int converted = atoi(number);
    if (converted == 0) {
        puts("atoi failed");
        exit(-2);
    }
    srand(converted);
    return (rand() & (~0xF)) % 0x400;
}

int main(int argc, char** argv) {
    setbuf(stdin, NULL);
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    if (argc < 2) {
        puts("no argument specified");
        exit(-1);
    }

    alloca(do_magic(argv[1]));
    while (1) {
        decrypt();
    }
    return 0;
}
