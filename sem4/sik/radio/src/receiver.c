/* See LICENSE file for copyright and license details. */

#include "receiver.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <threads.h>
#include <unistd.h>

#include "bitset.h"
#include "printer.h"
#include "util.h"

struct Receiver {
  /**@{*/
  thrd_t thrd;         /**< thread; */
  atomic_int running;  /**< running flag; */
  int sfd;             /**< socket file descriptor; */
  struct ip_mreq mreq; /**< multicast request; */
  int shutdown_fd[2];  /**< shutdown pipe; */
  mtx_t shutdown_mtx;  /**< shutdown mutex; */
  cnd_t shutdown_cnd;  /**< shutdown condition; */

  thrd_t rexmit_thrd;       /**< retransmission thread; */
  mtx_t mtx;                /**< mutex; */
  atomic_int valid;         /**< validity flag for rexmit thread; */
  struct sockaddr_in uaddr; /**< unicast address; */
  uint64_t first, last;     /**< first and last bytes; */
  size_t psize;             /**< packet size; */
  Bitset *bs;               /**< bitset; */
  /**@}*/
};

static int receiver_thread(void *);
static int rexmit_thread(void *);

char *discover_addr = "255.255.255.255";
uint16_t ctrl_port = 30000 + (438247 % 10000);
uint16_t ui_port = 10000 + (438247 % 10000);
size_t bsize = 65536;
size_t rtime = 250;
char *name = NULL;

Receiver *
receiver_create(uint16_t port, char *src_addr)
{
  Receiver *r = ecalloc(1, sizeof(Receiver));

  if (pipe(r->shutdown_fd) == -1) die("pipe:");

  if ((r->sfd = socket(AF_INET, SOCK_DGRAM, 0)) == -1) die("socket:");

  int opt = 1;
  size_t sopt = sizeof(opt), smreq = sizeof(r->mreq);
  if (setsockopt(r->sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sopt) == -1)
    die("setsockopt:");

  // TODO: check if this is necessary
  opt = 1;
  if (setsockopt(r->sfd, SOL_SOCKET, SO_REUSEPORT, &opt, sopt) == -1)
    die("setsockopt:");

  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(r->sfd, (struct sockaddr *)&addr, sizeof(addr)) == -1) die("bind:");

  struct addrinfo hints = {0}, *res;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  if (getaddrinfo(src_addr, NULL, &hints, &res) != 0)
    die("getaddrinfo: invalid address");

  r->mreq = (struct ip_mreq){0};
  r->mreq.imr_multiaddr.s_addr =
      ((struct sockaddr_in *)res->ai_addr)->sin_addr.s_addr;
  r->mreq.imr_interface.s_addr = htonl(INADDR_ANY);

  if (setsockopt(r->sfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &r->mreq, smreq) == -1)
    die("setsockopt:");

  freeaddrinfo(res);

  if (mtx_init(&r->shutdown_mtx, mtx_plain) != thrd_success) die("mtx_init:");
  if (cnd_init(&r->shutdown_cnd) != thrd_success) die("cnd_init:");

  if (mtx_init(&r->mtx, mtx_plain) != thrd_success) die("mtx_init:");
  r->bs = NULL;

  atomic_init(&r->running, 1);
  atomic_init(&r->valid, 0);

  if (thrd_create(&r->thrd, receiver_thread, r) != thrd_success)
    die("thrd_create:");

  if (thrd_create(&r->rexmit_thrd, rexmit_thread, r) != thrd_success)
    die("thrd_create:");

  return r;
}

void
receiver_destroy(Receiver *r)
{
  if (!r) return;
  atomic_store(&r->running, 0);
  if (write(r->shutdown_fd[1], "kys", 3) == -1) die("write:");
  if (thrd_join(r->thrd, NULL) != thrd_success) die("thrd_join:");

  if (mtx_lock(&r->shutdown_mtx) != thrd_success) die("mtx_lock:");
  if (cnd_signal(&r->shutdown_cnd) != thrd_success) die("cnd_signal:");
  if (mtx_unlock(&r->shutdown_mtx) != thrd_success) die("mtx_unlock:");

  if (thrd_join(r->rexmit_thrd, NULL) != thrd_success) die("thrd_join:");

  mtx_destroy(&r->mtx);
  mtx_destroy(&r->shutdown_mtx);
  cnd_destroy(&r->shutdown_cnd);

  bitset_destroy(r->bs);

  size_t smreq = sizeof(r->mreq);
  if (setsockopt(r->sfd, IPPROTO_IP, IP_DROP_MEMBERSHIP, &r->mreq, smreq) == -1)
    die("setsockopt:");

  if (close(r->sfd) == -1) die("close:");
  if (close(r->shutdown_fd[0]) == -1) die("close:");
  if (close(r->shutdown_fd[1]) == -1) die("close:");
  free(r);
}

int
receiver_thread(void *arg)
{
  Receiver *r = arg;

  char *msg = ecalloc(DATA_LENGTH_LIMIT, sizeof(char));
  Printer *p = printer_create();
  uint64_t session_id = 0;

  struct sockaddr_in client_addr;
  socklen_t client_addr_len = sizeof(client_addr);
  fd_set readfds;
  int maxfd = MAX(r->sfd, r->shutdown_fd[0]);

  while (atomic_load(&r->running)) {
    FD_ZERO(&readfds);
    FD_SET(r->sfd, &readfds);
    FD_SET(r->shutdown_fd[0], &readfds);

    if (select(maxfd + 1, &readfds, NULL, NULL, NULL) == -1) {
      if (errno == EINTR) continue;
      die("select:");
    }
    if (FD_ISSET(r->shutdown_fd[0], &readfds)) break;

    ssize_t len = recvfrom(r->sfd, msg, DATA_LENGTH_LIMIT, 0,
                           (struct sockaddr *)&client_addr, &client_addr_len);
    if (len == -1) {
      if (errno == EINTR) continue;
      die("recvfrom:");
    }

    AudioPacket pkg = {0};
    memcpy(&pkg.session_id, msg, sizeof(uint64_t));
    memcpy(&pkg.first_byte_num, msg + sizeof(uint64_t), sizeof(uint64_t));
    pkg.session_id = ntohll(pkg.session_id);
    pkg.first_byte_num = ntohll(pkg.first_byte_num);
    len -= sizeof(pkg.session_id) + sizeof(pkg.first_byte_num);

    if ((size_t)len > bsize) continue;

    if (session_id > pkg.session_id) continue;
    if (session_id < pkg.session_id) {
      session_id = pkg.session_id;
      printer_set_buffer(p, bsize, len, pkg.first_byte_num);

      if (mtx_lock(&r->mtx) != thrd_success) die("mtx_lock:");
      bitset_destroy(r->bs);
      r->bs = bitset_create(bsize / len);

      r->psize = len;
      r->first = pkg.first_byte_num, r->last = pkg.first_byte_num;

      memcpy(&r->uaddr, &client_addr, sizeof(client_addr));

      atomic_store(&r->valid, 1);

      if (mtx_unlock(&r->mtx) != thrd_success) die("mtx_unlock:");
    }

    printer_print(p, msg + sizeof(pkg.session_id) + sizeof(pkg.first_byte_num),
                  pkg.first_byte_num);

    if (mtx_lock(&r->mtx) != thrd_success) die("mtx_lock:");

    uint64_t cur = pkg.first_byte_num;
    if (cur >= r->first) {
      size_t size = bitset_size(r->bs);
      for (; (cur - r->first) / r->psize + 1 > size; r->first += r->psize)
        bitset_reset(r->bs, (r->first / r->psize) % size);
      bitset_set(r->bs, (cur / r->psize) % size);
      r->last = MAX(r->last, cur);
    }

    if (mtx_unlock(&r->mtx) != thrd_success) die("mtx_unlock:");
  }

  printer_destroy(p);
  free(msg);

  return thrd_success;
}

int
rexmit_thread(void *arg)
{
  Receiver *r = arg;

  int sfd, opt = 1;
  size_t sop = sizeof(opt);
  char *msg = ecalloc(DATA_LENGTH_LIMIT, sizeof(char));
  sprintf(msg, "LOUDER_PLEASE ");

  struct timespec ts, start, end;

  if ((sfd = socket(AF_INET, SOCK_DGRAM, 0)) == -1) die("socket:");
  if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sop) == -1)
    die("setsockopt:");

  struct sockaddr_in addr = {0};
  socklen_t addr_len = sizeof(addr);

  while (atomic_load(&r->running)) {
    if (clock_gettime(CLOCK_REALTIME, &start) == -1) die("clock_gettime:");

    if (mtx_lock(&r->mtx) != thrd_success) die("mtx_lock:");

    size_t offset = strlen("LOUDER_PLEASE ");

    if (!atomic_load(&r->running) || !atomic_load(&r->valid)) {
      if (mtx_unlock(&r->mtx) != thrd_success) die("mtx_unlock:");
      goto sleep;
    }

    for (size_t byte = r->first; byte != r->last; byte += r->psize) {
      size_t n = (byte / r->psize) % bitset_size(r->bs);
      if (bitset_test(r->bs, n)) continue;

      sprintf(msg + offset, "%zu,", byte);  // not a chance that it will fail
      offset += strlen(msg + offset);
    }

    if (atomic_load(&r->valid)) {
      memcpy(&addr, &r->uaddr, sizeof(addr));
      addr.sin_port = htons(ctrl_port);
      addr_len = sizeof(addr);
    }

    if (mtx_unlock(&r->mtx) != thrd_success) die("mtx_unlock:");

    if (offset > DATA_LENGTH_LIMIT) continue;  // TODO: split into multiple msgs

    if (offset > strlen("LOUDER_PLEASE ")) {
      msg[offset - 1] = '\n';
      if (sendto(sfd, msg, offset, 0, (struct sockaddr *)&addr, addr_len) == -1)
        warn("sendto:");  // not a big deal. we'll try again later... maybe
    }

  sleep:
    if (clock_gettime(CLOCK_REALTIME, &end) == -1) die("clock_gettime:");

    size_t msec = (end.tv_sec - start.tv_sec) * 1000 +
                  (end.tv_nsec - start.tv_nsec) / 1000000;
    if (msec >= rtime) continue;

    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) die("clock_gettime:");
    ts.tv_sec += (rtime - msec) / 1000;
    ts.tv_nsec += ((rtime - msec) % 1000) * 1000000;
    if (ts.tv_nsec >= 1000000000) {
      ts.tv_sec += 1;
      ts.tv_nsec -= 1000000000;
    }

    if (mtx_lock(&r->shutdown_mtx) != thrd_success) die("mtx_lock:");
    int ret = cnd_timedwait(&r->shutdown_cnd, &r->shutdown_mtx, &ts);
    if (ret != thrd_success && ret != thrd_timedout) die("cnd_timedwait:");
    if (mtx_unlock(&r->shutdown_mtx) != thrd_success) die("mtx_unlock:");
  }

  if (close(sfd) == -1) die("close:");
  free(msg);

  return thrd_success;
}
