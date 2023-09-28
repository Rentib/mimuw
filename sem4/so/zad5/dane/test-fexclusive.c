#include <lib.h>
#include <minix/rs.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    message m;
    endpoint_t vfs_ep;
    int ret;
    int fd;

    // Sprawdź liczbę argumentów.
    if (argc != 2 ) {
        printf("Użycie:\n%s plik\n", argv[0]);
        return 1;
    }

    // Otwórz plik.
    fd = open(argv[1], O_RDONLY);
    if (fd == -1) {
        printf("Nie można otworzyć pliku %s (wynik: %d).\n", argv[1], fd);
        return 1;
    }

    // Zablokuj plik.
    printf("Blokuję plik...\n");
    m.m_lc_vfs_exclusive.fd = fd;
    m.m_lc_vfs_exclusive.flags = EXCL_LOCK; // albo EXCL_LOCK_NO_OTHERS
    minix_rs_lookup("vfs", &vfs_ep);
    ret = _syscall(vfs_ep, VFS_FEXCLUSIVE, &m);
    printf("Wynik VFS_FEXCLUSIVE: %d, errno: %d\n", ret, errno);

    if (ret != 0)
        return (ret != 0);

    // Czekaj.
    printf("Czekam... Naciśnij coś\n");
    getchar();

    // Użytkownik może odblokować plik ręcznie:
    // printf("Odblokowuję plik...\n");
    // m.m_lc_vfs_exclusive.fd = fd;
    // m.m_lc_vfs_exclusive.flags = EXCL_UNLOCK; // albo EXCL_UNLOCK_FORCE
    // minix_rs_lookup("vfs", &vfs_ep);
    // ret = _syscall(vfs_ep, VFS_FEXCLUSIVE, &m);
    // printf("Wynik VFS_FEXCLUSIVE: %d, errno: %d\n", ret, errno);

    // Zamknij plik (odblokuje plik automatycznie, jeśli użytkownik nie odblokował go).
    close(fd);

    // Zakończ.
    return ret != 0;
}
