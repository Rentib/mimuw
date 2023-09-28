#!/bin/sh
mv sb438247.patch /
cd /
patch -t -p1 < sb438247.patch
cd /usr/src; make includes
cd /usr/src/minix/servers/pm; make clean && make && make install
cd /usr/src/minix/fs/procfs; make && make install
cd /usr/src/minix/drivers/storage/ramdisk; make && make install
cd /usr/src/minix/drivers/storage/memory; make && make install
cd /usr/src/lib/libc; make && make install
cd /usr/src/releasetools; make do-hdboot
