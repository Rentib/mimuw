/* See LICENSE file for copyright and license details. */

#include <arpa/inet.h>
#include <errno.h>
#include <getopt.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <threads.h>
#include <time.h>
#include <unistd.h>

#include "cyclicbuffer.h"
#include "queue.h"
#include "util.h"

typedef struct Rexmit Rexmit;
struct Rexmit {
  struct sockaddr_in client;
  size_t len;
  uint64_t *bytes;
};

typedef struct Sender Sender;
struct Sender {
  int main_sfd;
  struct sockaddr_in mcast_group;
  int shutdown_fd[2];
  int ctrl_sfd;
  thrd_t ctrl_thrd;
  uint64_t session_id;
  uint64_t first, last;
  mtx_t mtx;
  CyclicBuffer *fcb;
  Queue *rexmit_q;
  thrd_t rexmit_thrd;
};

/* function declarations */
static void cleanup(Sender *);
static int ctrl_thread(void *);
static int rexmit_thread(void *);
static void run(Sender *);
static void setup(Sender *);
static void sigint_handler(int);
static void usage(void);

/* global variables */
static atomic_int running;
static char *mcast_addr = NULL;
static uint16_t data_port = 20000 + (438247 % 10000);
static uint16_t ctrl_port = 30000 + (438247 % 10000);
static size_t psize = 512;
static size_t fsize = 128 * 1024;  // 128kB
static size_t rtime = 250;
static char *name = "Nienazwany Nadajnik";

/* function definitions */
void
cleanup(Sender *s)
{
  atomic_store(&running, 0);

  if (close(s->main_sfd) == -1) die("close:");

  if (write(s->shutdown_fd[1], "kys", 3) == -1) die("write:");
  if (thrd_join(s->ctrl_thrd, NULL) != thrd_success) die("thrd_join:");
  if (close(s->ctrl_sfd) == -1) die("close:");
  if (close(s->shutdown_fd[0]) == -1) die("close:");
  if (close(s->shutdown_fd[1]) == -1) die("close:");
  if (thrd_join(s->rexmit_thrd, NULL) != thrd_success) die("thrd_join:");

  mtx_destroy(&s->mtx);
  cb_destroy(s->fcb);

  while (s->rexmit_q->size) {
    Rexmit *r = queue_pop(s->rexmit_q);
    free(r->bytes);
    free(r);
  }
  queue_destroy(s->rexmit_q);
}

int
ctrl_thread(void *arg)
{
  Sender *_Atomic s = arg;

  char *msg = ecalloc(DATA_LENGTH_LIMIT, sizeof(char));
  char *reply = ecalloc(DATA_LENGTH_LIMIT, sizeof(char));
  sprintf(reply, "BOREWICZ_HERE %s %d %s\n", mcast_addr, data_port, name);
  size_t reply_len = strlen(reply);
  reply = realloc(reply, reply_len + 1);  // it cannot fail

  struct sockaddr_in client;
  socklen_t client_len = sizeof(client);
  fd_set readfds;
  int maxfd = MAX(s->ctrl_sfd, s->shutdown_fd[0]);

  while (1) {
    FD_ZERO(&readfds);
    FD_SET(s->ctrl_sfd, &readfds);
    FD_SET(s->shutdown_fd[0], &readfds);

    if (select(maxfd + 1, &readfds, NULL, NULL, NULL) == -1) {
      if (errno == EINTR) continue;
      die("select:");
    }
    if (FD_ISSET(s->shutdown_fd[0], &readfds)) break;

    ssize_t len = recvfrom(s->ctrl_sfd, msg, DATA_LENGTH_LIMIT, 0,
                           (struct sockaddr *)&client, &client_len);
    if (len == -1) {
      if (errno == EINTR) continue;
      die("recvfrom:");
    }

    if (len == 19 && strncmp(msg, "ZERO_SEVEN_COME_IN\n", 19) == 0) {
      while (1) {
        len = sendto(s->ctrl_sfd, reply, reply_len, 0,
                     (struct sockaddr *)&client, client_len);
        if (len == -1 && errno != EINTR) die("sendto:");
        if ((size_t)len == reply_len) break;
      }
    } else if (len > 13 && strncmp(msg, "LOUDER_PLEASE ", 14) == 0) {
      Rexmit *r = ecalloc(1, sizeof(Rexmit));
      r->client = client;
      r->len = 1;

      for (size_t i = 13; i < (size_t)len; ++i) r->len += msg[i] == ',';
      r->bytes = ecalloc(r->len, sizeof(uint64_t));

      r->len = 0;
      msg[len - 1] = '\0';
      char *token = strtok(msg + 13, ",");
      while (token != NULL) {
        char *endptr;
        r->bytes[r->len++] = strtoull(token, &endptr, 10);
        if (errno || *endptr != '\0') {
          warn("received incorrect LOUDER_PLEASE message");
          free(r->bytes);
          free(r);
          r = NULL;
          break;
        }
        token = strtok(NULL, ",");
      }

      if (!r) continue;

      if (mtx_lock(&s->mtx) != thrd_success) die("mtx_lock:");
      queue_push(s->rexmit_q, r);
      if (mtx_unlock(&s->mtx) != thrd_success) die("mtx_unlock:");
    }
  }

  free(reply);
  free(msg);
  return thrd_success;
}

int
rexmit_thread(void *arg)
{
  Sender *_Atomic s = arg;
  int sfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sfd == -1) die("socket:");
  int optval = 1;
  if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) == -1)
    die("setsockopt:");
  if (setsockopt(sfd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval)) == -1)
    die("setsockopt:");

  char *msg = ecalloc(8 + 8 + psize, sizeof(char));

  struct timespec start, end;

  while (atomic_load(&running)) {
    if (mtx_lock(&s->mtx) != thrd_success) die("mtx_lock:");
    size_t qsize = s->rexmit_q->size;
    if (mtx_unlock(&s->mtx) != thrd_success) die("mtx_unlock:");

    if (clock_gettime(CLOCK_MONOTONIC, &start) == -1) die("clock_gettime:");

    while (qsize--) {
      if (mtx_lock(&s->mtx) != thrd_success) die("mtx_lock:");
      Rexmit *r = queue_pop(s->rexmit_q);
      if (mtx_unlock(&s->mtx) != thrd_success) die("mtx_unlock:");

      for (size_t i = 0; i < r->len; ++i) {
        if (mtx_lock(&s->mtx) != thrd_success) die("mtx_lock:");

        if (r->bytes[i] < s->first || s->last < r->bytes[i]) goto unlock;

        uint64_t session_id = htonll(s->session_id);
        uint64_t first_byte_num = htonll(r->bytes[i]);

        memcpy(msg, &session_id, 8);
        memcpy(msg + 8, &first_byte_num, 8);
        memcpy(msg + 16, s->fcb->buf + r->bytes[i] - s->first, psize);

        r->client.sin_port = htons(data_port);
        // TODO: move sendto outside of the lock
        ssize_t len = sendto(sfd, msg, 8 + 8 + psize, 0,
                             (struct sockaddr *)&r->client, sizeof(r->client));
        if (len == -1 && errno != EINTR) warn("sendto:");
        // retransmissions are not that big of a deal to end the program
        // or try until success

      unlock:
        if (mtx_unlock(&s->mtx) != thrd_success) die("mtx_unlock:");
      }

      free(r->bytes);
      free(r);
    }

    if (clock_gettime(CLOCK_MONOTONIC, &end) == -1) die("clock_gettime:");

    size_t msec = (end.tv_sec - start.tv_sec) * 1000 +
                  (end.tv_nsec - start.tv_nsec) / 1000000;
    if (msec >= rtime) continue;

    struct timespec sleep_time = {
        .tv_sec = (rtime - msec) / 1000,
        .tv_nsec = ((rtime - msec) % 1000) * 1000000,
    };

    if (thrd_sleep(&sleep_time, NULL) != thrd_success) warn("thrd_sleep:");
    // not a big deal if we sleep a bit more or less than rtime
  }

  free(msg);
  if (close(sfd) == -1) die("close:");

  return thrd_success;
}

void
run(Sender *s)
{
  AudioPacket pkg = {
      .session_id = (uint64_t)time(NULL),
      .first_byte_num = 0,
      .audio_data = ecalloc(psize, sizeof(char)),
  };
  s->session_id = pkg.session_id;

  int c;
  size_t cur = 0;
  char *msg = ecalloc(8 + 8 + psize, sizeof(char));

  while (atomic_load(&running) && (c = fgetc(stdin)) != EOF) {
    pkg.audio_data[cur++] = c;
    if (cur == psize) {
      uint64_t session_id = htonll(pkg.session_id);
      uint64_t first_byte_num = htonll(pkg.first_byte_num);

      memcpy(msg, &session_id, 8);
      memcpy(msg + 8, &first_byte_num, 8);
      memcpy(msg + 16, pkg.audio_data, psize);

      while (1) {
        ssize_t len =
            sendto(s->main_sfd, msg, 8 + 8 + psize, 0,
                   (struct sockaddr *)&s->mcast_group, sizeof(s->mcast_group));
        if (len == -1 && errno != EINTR) die("sendto:");
        if ((size_t)len == 8 + 8 + psize) break;
      }
      if (mtx_lock(&s->mtx) != thrd_success) die("mtx_lock:");
      cb_push(s->fcb, pkg.audio_data, pkg.first_byte_num - s->first);
      s->last = MAX(s->last, pkg.first_byte_num);
      if (s->last > s->fcb->cap)
        s->first = MAX(s->first, s->last - s->fcb->cap);
      if (mtx_unlock(&s->mtx) != thrd_success) die("mtx_unlock:");

      pkg.first_byte_num += psize;
      cur = 0;
    }
  }

  free(msg);
  free(pkg.audio_data);
}

void
setup(Sender *s)
{
  int opt;
  size_t sopt = sizeof(opt);

  // setup signal handling
  signal(SIGINT, sigint_handler);

  // disable buffering on stdin
  setvbuf(stdin, NULL, _IONBF, 0);

  s->mcast_group = (struct sockaddr_in){0};
  s->mcast_group.sin_family = AF_INET;
  s->mcast_group.sin_port = htons(data_port);
  struct addrinfo hints = {0}, *res;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  if (getaddrinfo(mcast_addr, NULL, &hints, &res) != 0)
    die("getaddrinfo: invalid address");
  s->mcast_group.sin_addr = ((struct sockaddr_in *)res->ai_addr)->sin_addr;
  freeaddrinfo(res);

  // if mcast_group is not a multicast address, die
  if ((ntohl(s->mcast_group.sin_addr.s_addr) & 0xF0000000) != 0xE0000000)
    die("invalid multicast address");

  // setup main socket
  if ((s->main_sfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    die("socket:");

  opt = 1;
  if (setsockopt(s->main_sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sopt) == -1)
    die("setsockopt:");
  opt = 4;
  if (setsockopt(s->main_sfd, IPPROTO_IP, IP_MULTICAST_TTL, &opt, sopt) == -1)
    die("setsockopt:");

  // setup control socket
  if (pipe(s->shutdown_fd) == -1) die("pipe:");

  if ((s->ctrl_sfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    die("socket:");

  opt = 1;
  if (setsockopt(s->ctrl_sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sopt) == -1)
    die("setsockopt:");

  struct sockaddr_in ctrl_addr = (struct sockaddr_in){0};
  ctrl_addr.sin_family = AF_INET;
  ctrl_addr.sin_port = htons(ctrl_port);
  ctrl_addr.sin_addr.s_addr = htonl(INADDR_ANY);

  if (bind(s->ctrl_sfd, (struct sockaddr *)&ctrl_addr, sizeof(ctrl_addr)) == -1)
    die("bind:");

  if (mtx_init(&s->mtx, mtx_plain) != thrd_success) die("mtx_init:");
  s->first = 0, s->last = 0;
  s->fcb = cb_create(fsize / psize * psize, psize);
  s->rexmit_q = queue_create();

  atomic_init(&running, 1);

  if (thrd_create(&s->ctrl_thrd, ctrl_thread, s) != thrd_success)
    die("thrd_create:");

  if (thrd_create(&s->rexmit_thrd, rexmit_thread, s) != thrd_success)
    die("thrd_create:");
}

void
sigint_handler(int sig)
{
  if (sig == SIGINT) atomic_store(&running, 0);
}

void
usage(void)
{
  fprintf(stderr,
          "usage: %s "
          "-a <adres multiemisji> "
          "[-P <port udp>] "
          "[-C <port kontrolny>] "
          "[-p <rozmiar pola audio_data>] "
          "[-f <rozmiar kolejki FIFO>] "
          "[-R <czas pomiędzy retransmisjami>] "
          "[-n <nazwa nadajnika>]\n",
          argv0);
  exit(EXIT_FAILURE);
}

int
main(int argc, char *argv[])
{
  argv0 = *argv;

  int option;
  while ((option = getopt(argc, argv, "a:P:C:p:f:R:n:")) != -1) {
    switch (option) {
    case 'a': {
      mcast_addr = optarg;
    } break;
    case 'P': {
      data_port = strtoport(optarg);
    } break;
    case 'C': {
      ctrl_port = strtoport(optarg);
    } break;
    case 'p': {
      psize = estrtoul(optarg, 10);
      if (psize > DATA_LENGTH_LIMIT - 16) die("invalid psize");
    } break;
    case 'f': {
      if ((fsize = estrtoul(optarg, 10)) < psize) die("invalid fsize");
    } break;
    case 'R': {
      rtime = estrtoul(optarg, 10);
    } break;
    case 'n': {
      name = optarg;
      if (strlen(name) == 0 || strlen(name) > 64 || name[0] == ' ' ||
          name[strlen(name) - 1] == ' ')
        die("invalid name");
      for (char *p = name; *p; ++p)
        if (*p < 32) die("invalid name");
      // chars above 127 are overflowing to negative values
    } break;
    default:
      usage();
    }
  }

  if (mcast_addr == NULL) usage();

  Sender s = {0};
  setup(&s);
  run(&s);
  cleanup(&s);

  return EXIT_SUCCESS;
}
