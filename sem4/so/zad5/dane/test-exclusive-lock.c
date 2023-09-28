#include <lib.h>
#include <minix/rs.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int main(int argc, char *argv[]) {
    message m;
    endpoint_t vfs_ep;
    int flags;
    int ret;
    int fd;

    // Sprawdź liczbę argumentów.
    if (argc != 2 ) {
        printf("Użycie:\n%s plik\n", argv[0]);
        return 1;
    }

    // Zablokuj plik.
    printf("Blokuję plik...\n");
    m.m_lc_vfs_exclusive.name = (vir_bytes)argv[1];
    m.m_lc_vfs_exclusive.len = strlen(argv[1]) + 1;
    m.m_lc_vfs_exclusive.flags = EXCL_LOCK; // albo EXCL_LOCK_NO_OTHERS
    minix_rs_lookup("vfs", &vfs_ep);
    ret = _syscall(vfs_ep, VFS_EXCLUSIVE, &m);
    printf("Wynik VFS_EXCLUSIVE: %d, errno: %d\n", ret, errno);

    // Zakończ.
    return ret != 0;
}
