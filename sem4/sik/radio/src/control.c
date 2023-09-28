/* See LICENSE file for copyright and license details. */

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
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

#include "receiver.h"
#include "util.h"

typedef struct {
  char mcast_addr[64];
  uint16_t data_port;
  char name[128];  // 128 instead of 64 just to be sure
  uint64_t time;
} Sender;

typedef struct {
  atomic_size_t size;
  size_t cap;
  Sender *senders;
  size_t selected;
} SenderList;

/* function declarations */
static void cleanup(void);
static int discoverer_thread(void *);
static void get_sender_list(char *, size_t);
static void run(void);
static int scanner_thread(void *);
static int sender_cmp(const void *, const void *);
static void setup(void);
static void sigint_handler(int);
static int ui_thread(void *);
static void usage(void);

/* global variables */
static struct {
  int ctrl_sfd;
  int shutdown_fd[2];
  struct sockaddr_in addr;
  thrd_t discoverer;
  thrd_t scanner;

  int ui_fd[2];
  thrd_t ui;
} ctrl;
static atomic_int running = 1;
static mtx_t mtx;
static cnd_t cnd;
static cnd_t discoverer_cnd;
static SenderList senders = {0, 0, NULL, 0};

/* function definitions */
void
cleanup(void)
{
  if (write(ctrl.ui_fd[1], "kys", 3) == -1) die("write:");
  if (thrd_join(ctrl.ui, NULL) != thrd_success) die("thrd_join:");
  if (close(ctrl.ui_fd[0]) == -1) die("close:");
  if (close(ctrl.ui_fd[1]) == -1) die("close:");

  if (mtx_lock(&mtx) != thrd_success) die("mtx_lock:");
  if (cnd_signal(&discoverer_cnd) != thrd_success) die("cnd_signal:");
  if (mtx_unlock(&mtx) != thrd_success) die("mtx_unlock:");
  if (thrd_join(ctrl.discoverer, NULL) != thrd_success) die("thrd_join:");

  if (write(ctrl.shutdown_fd[1], "kys", 3) == -1) die("write:");
  if (thrd_join(ctrl.scanner, NULL) != thrd_success) die("thrd_join:");
  if (close(ctrl.shutdown_fd[0]) == -1) die("close:");
  if (close(ctrl.shutdown_fd[1]) == -1) die("close:");
  free(senders.senders);

  cnd_destroy(&discoverer_cnd);
  cnd_destroy(&cnd);
  mtx_destroy(&mtx);

  if (close(ctrl.ctrl_sfd) == -1) die("close:");
}

int
discoverer_thread(void *arg)
{
  // TODO: pass socket and address as arguments
  (void)arg;

  int sfd = ctrl.ctrl_sfd;
  struct sockaddr_in addr = ctrl.addr;
  (void)addr;

  char msg[] = "ZERO_SEVEN_COME_IN\n";
  size_t smsg = sizeof(msg) - 1, saddr = sizeof(addr);

  while (running) {
    while (1) {
      ssize_t len = sendto(sfd, msg, smsg, 0, (struct sockaddr *)&addr, saddr);
      if (len == -1 && errno != EINTR) die("sendto:");
      if ((size_t)len == smsg) break;
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 5;
    if (mtx_lock(&mtx) != thrd_success) die("mtx_lock:");
    int ret = cnd_timedwait(&discoverer_cnd, &mtx, &ts);
    if (ret != thrd_success && ret != thrd_timedout) die("cnd_timedwait:");
    if (mtx_unlock(&mtx) != thrd_success) die("mtx_unlock:");
  }

  return thrd_success;
}

void
get_sender_list(char *buf, size_t len)
{
  memset(buf, 0, len);

  char *header =
      "\033[H\033[2J"
      "------------------------------------------------------------------------"
      "\r\n\r\n SIK "
      "Radio\r\n\r\n-----------------------------------------------------------"
      "-------------\r\n\r\n";
  char *footer =
      "\r\n--------------------------------------------------------------------"
      "----\r\n\r\n";
  size_t header_len = strlen(header);
  size_t footer_len = strlen(footer);

  if (mtx_lock(&mtx) != thrd_success) die("mtx_lock:");

  size_t pos = 0;
  memcpy(buf + pos, header, header_len);
  pos += header_len;
  for (size_t i = 0; i < senders.size; ++i) {
    if (i == senders.selected) {
      memcpy(buf + pos, " > ", 3);
      pos += 3;
    }
    memcpy(buf + pos, senders.senders[i].name, strlen(senders.senders[i].name));
    pos += strlen(senders.senders[i].name);
    memcpy(buf + pos, "\r\n", 2);
    pos += 2;
  }
  memcpy(buf + pos, footer, footer_len);

  if (mtx_unlock(&mtx) != thrd_success) die("mtx_unlock:");
}

void
run(void)
{
  Receiver *receiver = NULL;
  uint16_t cur_port = 0;
  char cur_addr[64] = {0};

  while (running) {
    if (mtx_lock(&mtx) != thrd_success) die("mtx_lock:");
    if (cnd_wait(&cnd, &mtx) != thrd_success) die("cnd_wait:");

    if (!running || !senders.size) {
      receiver_destroy(receiver);
      receiver = NULL;
      cur_port = 0;
      goto unlock;
    }

#if 0
    // NOTE: this is just for debugging
    fprintf(stderr, "\033[1mSender list\033[0m (%zu):\n", time(NULL));
    for (size_t i = 0; i < senders.size; ++i) {
      fprintf(stderr,
              "\033[1m%zu)\033[0m \033[32m%s\033[0m -- "
              "\033[31m%s\033[0m:\033[34m%hu\033[0m\n",
              i + 1, senders.senders[i].name, senders.senders[i].mcast_addr,
              senders.senders[i].data_port);
    }
#endif

    Sender *s = &senders.senders[senders.selected];
    if (cur_port == s->data_port && strcmp(cur_addr, s->mcast_addr) == 0)
      goto unlock;

    cur_port = s->data_port;
    strcpy(cur_addr, s->mcast_addr);
    receiver_destroy(receiver);
    receiver = receiver_create(cur_port, cur_addr);

  unlock:
    if (mtx_unlock(&mtx) != thrd_success) die("mtx_unlock:");
  }
}

int
scanner_thread(void *arg)
{
  (void)arg;

  int sfd = ctrl.ctrl_sfd;
  struct sockaddr_in addr = ctrl.addr;
  (void)addr;

  char msg[256];

  struct sockaddr_in sender;
  socklen_t sender_len = sizeof(sender);
  fd_set readfds;
  int maxfd = MAX(sfd, ctrl.shutdown_fd[0]);
  Sender s = {0};
  struct timeval timeval = {20, 0};

  while (running) {
    FD_ZERO(&readfds);
    FD_SET(sfd, &readfds);
    FD_SET(ctrl.shutdown_fd[0], &readfds);

    int locked = 0, notify = 0;

    switch (select(maxfd + 1, &readfds, NULL, NULL, &timeval)) {
    case 0: {
      s.time = time(NULL);
      goto update_senders;
    } break;
    case -1:
      if (errno == EINTR) continue;
      die("select:");
    }
    if (FD_ISSET(ctrl.shutdown_fd[0], &readfds)) break;

    ssize_t len = recvfrom(sfd, msg, sizeof(msg), 0, (struct sockaddr *)&sender,
                           &sender_len);
    if (len == -1) {
      if (errno == EINTR) continue;
      die("recvfrom:");
    }

    s.time = time(NULL);
    if (strncmp(msg, "BOREWICZ_HERE ", 14) != 0) goto update_senders;

    size_t port;
    if (sscanf(msg + 14, "%s %zu %64[^\n]", s.mcast_addr, &port, s.name) != 3)
      goto update_senders;

    // check whether port is valid
    if (!port || port > 65535) goto update_senders;
    s.data_port = (uint16_t)port;

    // check whether mcast_addr is valid
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    if (getaddrinfo(s.mcast_addr, NULL, &hints, &res) != 0) goto update_senders;
    freeaddrinfo(res);

    if (mtx_lock(&mtx) != thrd_success) die("mtx_lock:");
    locked = 1;
    for (size_t i = 0; i < senders.size; ++i) {
      if (strcmp(senders.senders[i].name, s.name) == 0) {
        senders.senders[i].time = s.time;
        goto update_senders;
      }
    }

    // we have a new sender
    notify = 1;
    if (senders.size == senders.cap) {
      senders.cap = MAX(1, senders.cap * 2);
      senders.senders = realloc(senders.senders, senders.cap * sizeof(Sender));
      if (!senders.senders) die("realloc:");
    }
    senders.senders[senders.size++] = s;
    senders.selected += sender_cmp(&s, &senders.senders[senders.selected]) < 0;

  update_senders:
    timeval = (struct timeval){20, 0};
    if (!locked && mtx_lock(&mtx) != thrd_success) die("mtx_lock:");

    for (size_t i = 0; i < senders.size; ++i) {
      if (s.time - senders.senders[i].time >= 20) {
        if (i == senders.selected) {
          senders.selected = 0;
          for (size_t j = 0; j < senders.size - 1; ++j)
            if (!strcmp(senders.senders[j].name, name)) senders.selected = j;
        } else if (i < senders.selected) {
          senders.selected--;
        }
        senders.senders[i--] = senders.senders[--senders.size];
        notify = 1;
      } else {
        // we want to wake up after the first sender dies
        timeval.tv_sec = MIN((uint64_t)timeval.tv_sec,
                             20 - (s.time - senders.senders[i].time));
      }
    }

    qsort(senders.senders, senders.size, sizeof(Sender), sender_cmp);

    cnd_signal(&cnd);
    mtx_unlock(&mtx);

    if (notify && write(ctrl.ui_fd[1], "upd", 3) == -1)
      warn("write:");  // not a big deal
  }

  return thrd_success;
}

int
sender_cmp(const void *a, const void *b)
{
  const Sender *s1 = a, *s2 = b;
  return strcmp(s1->name, s2->name);
}

void
setup(void)
{
  signal(SIGINT, sigint_handler);

  int opt;
  size_t sopt = sizeof(opt);
  if ((ctrl.ctrl_sfd = socket(AF_INET, SOCK_DGRAM, 0)) == -1) die("socket:");

  opt = 1;
  if (setsockopt(ctrl.ctrl_sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sopt) == -1)
    die("setsockopt:");
  opt = 1;
  if (setsockopt(ctrl.ctrl_sfd, SOL_SOCKET, SO_BROADCAST, &opt, sopt) == -1)
    die("setsockopt:");

  ctrl.addr.sin_family = AF_INET;
  ctrl.addr.sin_port = htons(ctrl_port);

  struct addrinfo hints = {0}, *res;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  if (getaddrinfo(discover_addr, NULL, &hints, &res) != 0)
    die("getaddrinfo: invalid address");
  ctrl.addr.sin_addr = ((struct sockaddr_in *)res->ai_addr)->sin_addr;
  freeaddrinfo(res);

  // research why bind breaks stuff

  if (mtx_init(&mtx, mtx_plain) != thrd_success) die("mtx_init:");
  if (cnd_init(&cnd) != thrd_success) die("cnd_init:");
  if (cnd_init(&discoverer_cnd) != thrd_success) die("cnd_init:");

  if (pipe(ctrl.shutdown_fd) == -1) die("pipe:");
  if (pipe(ctrl.ui_fd) == -1) die("pipe:");

  if (thrd_create(&ctrl.scanner, scanner_thread, NULL) != thrd_success)
    die("thrd_create:");

  if (thrd_create(&ctrl.discoverer, discoverer_thread, NULL) != thrd_success)
    die("thrd_create:");

  if (thrd_create(&ctrl.ui, ui_thread, NULL) != thrd_success)
    die("thrd_create:");
}

void
sigint_handler(int sig)
{
  if (sig != SIGINT) return;
  running = 0;
  // TODO: check if it is safe
  if (mtx_lock(&mtx) != thrd_success) die("mtx_lock:");
  if (cnd_signal(&cnd) != thrd_success) die("cnd_signal:");
  if (mtx_unlock(&mtx) != thrd_success) die("mtx_unlock:");
}

int
ui_thread(void *arg)
{
  (void)arg;

  int sfd, opt;
  size_t sopt = sizeof(opt);

  if ((sfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) die("socket:");

  opt = 1;
  if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sopt) == -1)
    die("setsockopt:");
  opt = 1;
  if (setsockopt(sfd, IPPROTO_TCP, TCP_NODELAY, &opt, sopt) == -1)
    die("setsockopt:");

  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(ui_port);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);

  if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) == -1) die("bind:");

  char *buf = ecalloc(BUFSIZ, sizeof(char));
  ssize_t len;

  size_t fds_size = 4;
  struct pollfd *fds = ecalloc(fds_size, sizeof(struct pollfd));
  fds[0].fd = sfd;
  fds[0].events = POLLIN;
  fds[1].fd = ctrl.ui_fd[0];
  fds[1].events = POLLIN;
  for (nfds_t i = 2; i < fds_size; ++i) fds[i] = (struct pollfd){-1, 0, 0};

  nfds_t nfds = 2;
  int timeout = -1;

  if (listen(sfd, 5) == -1) die("listen:");

  while (running) {
    for (nfds_t i = 0; i < nfds; ++i) fds[i].revents = 0;

    int ret = poll(fds, nfds, timeout);
    if (ret == -1) {
      if (errno == EINTR) continue;
      die("poll:");
    }

    if (ret == 0) continue;

    if (running && fds[0].revents & POLLIN) {
      int fd = accept(fds[0].fd, NULL, NULL);
      if (fd == -1) {
        warn("accept:");  // not a big deal
      } else if (fcntl(fd, F_SETFL, O_NONBLOCK) == -1) {
        warn("fcntl:");  // not a big deal
        close(fd);
      } else {
        if (nfds == fds_size) {
          fds_size *= 2;
          fds = realloc(fds, fds_size * sizeof(struct pollfd));
          if (fds == NULL) die("realloc:");  // big deal
          for (nfds_t i = nfds; i < fds_size; ++i)
            fds[i] = (struct pollfd){-1, 0, 0};
        }

        fds[nfds].fd = fd;
        fds[nfds++].events = POLLIN;

        uint8_t mode_character[] = "\377\375\042\377\373\001";
        if (send(fd, mode_character, 6, 0) == -1)
          warn("send:");  // not a big deal

        get_sender_list(buf, BUFSIZ);
        if (send(fd, buf, strlen(buf), 0) == -1)
          warn("send:");  // not a big deal
      }
    }

    int updated = 0;

    if (running && fds[1].revents & POLLIN) {
      if (read(fds[1].fd, buf, 3) == -1 && strncmp(buf, "upd", 3) == 0)
        warn("read:");  // not a big deal
      updated = 1;
    }

    for (nfds_t i = 2; i < nfds; ++i) {
      if (fds[i].fd == -1 || !(fds[i].revents & POLLIN)) continue;

      len = recv(fds[i].fd, buf, BUFSIZ, 0);
      if (len == -1) {
        warn("recv:");  // not a big deal
        continue;
      } else if (len == 0) {
        if (close(fds[i].fd) == -1) die("close:");
        fds[i] = fds[--nfds];
        fds[nfds].fd = -1;
        i--;
        continue;
      }

      if (senders.size < 2) continue;

      if (buf[0] == 27 && buf[1] == 91 && buf[2] == 65) {
        mtx_lock(&mtx);
        senders.selected = (senders.selected + senders.size - 1) % senders.size;
        mtx_unlock(&mtx);
        updated = 1;
      } else if (buf[0] == 27 && buf[1] == 91 && buf[2] == 66) {
        mtx_lock(&mtx);
        senders.selected = (senders.selected + 1) % senders.size;
        mtx_unlock(&mtx);
        updated = 1;
      }
    }

    if (!updated) continue;

    get_sender_list(buf, BUFSIZ);
    for (nfds_t i = 2; i < nfds; ++i) {
      if (fds[i].fd == -1) continue;
      if (send(fds[i].fd, buf, strlen(buf), 0) == -1)
        warn("send:");  // not a big deal
    }

    if (mtx_lock(&mtx) != thrd_success) die("mtx_lock:");
    if (cnd_signal(&cnd) != thrd_success) die("cnd_signal:");
    if (mtx_unlock(&mtx) != thrd_success) die("mtx_unlock:");
  }

  if (close(fds[0].fd) == -1) die("close:");
  for (nfds_t i = 2; i < nfds; ++i) {
    if (fds[i].fd != -1)
      if (close(fds[i].fd) == -1) die("close:");
  }

  free(fds);
  free(buf);
  return thrd_success;
}

void
usage(void)
{
  fprintf(stderr,
          "usage: %s "
          "[-d <adres odkrywania>] "
          "[-C <port kontrolny>] "
          "[-U <port ui>] "
          "[-b <rozmiar bufora>] "
          "[-R <czas pomiędzy retransmisjami>] "
          "[-n <nazwa preferowanego nadajnika>]\n",
          argv0);
  exit(EXIT_FAILURE);
}

int
main(int argc, char *argv[])
{
  argv0 = *argv;

  int option;
  while ((option = getopt(argc, argv, "d:C:U:b:R:n:")) != -1) {
    switch (option) {
    case 'd': {
      discover_addr = optarg;
      // TODO: check if we need to ensure that it is a valid broadcast address
    } break;
    case 'C': {
      ctrl_port = strtoport(optarg);
    } break;
    case 'U': {
      ui_port = strtoport(optarg);
    } break;
    case 'b': {
      bsize = estrtoul(optarg, 10);
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

  setup();
  run();
  cleanup();

  return EXIT_SUCCESS;
}
