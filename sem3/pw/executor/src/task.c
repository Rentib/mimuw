/*
  Executor, homework assignment for the course "Concurrent Programming".
  Copyright (C) 2022 Stanisław Bitner <sb438247>

  Executor is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Executor is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "task.h"
#include "util.h"

typedef union Arg Arg;
typedef struct Task Task;

union Arg {
  struct {
    Task *t;
    char **args;
    pthread_barrier_t *brr;
  };
  struct {
    int fd;
    char *buf;
    pthread_mutex_t *mtx;
  };
};

struct Task {
  _Atomic pid_t pid;
  _Atomic _Bool running;
  pthread_t thrd;
  char buf[2][MAX_LINE];  /* 0 - STDOUT, 1 - STDERR */
  pthread_mutex_t mtx[2]; /* 0 - STDOUT, 1 - STDERR */
};

static Task tasks[MAX_TASKS];

static pthread_mutex_t queue_mtx = PTHREAD_MUTEX_INITIALIZER;
static _Bool enqueue;
static char queue[MAX_TASKS * 64]; /* cuz im dumb */

static void
safe_printf(const char *fmt, ...)
{
  static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
  va_list ap;
  va_start(ap, fmt);
  CHECK_PTH(pthread_mutex_lock(&mtx));
  vprintf(fmt, ap);
  CHECK_PTH(pthread_mutex_unlock(&mtx));
  va_end(ap);
}

static void *
read_pipe(void *arg)
{
  Arg *a = (Arg *)arg;
  char buf[MAX_LINE] = {0};
  FILE *fp;
  _Bool running = 1;

  if (!(fp = fdopen(a->fd, "r")))
    die("fdopen");

  while (running) {
    running = fgets(buf, MAX_LINE, fp) != NULL;
    CHECK_PTH(pthread_mutex_lock(a->mtx));
    strcpy(a->buf, buf);
    a->buf[strcspn(a->buf, "\n")] = '\0';
    CHECK_PTH(pthread_mutex_unlock(a->mtx));
  }

  fclose(fp);

  return NULL;
}

static void *
spawn_task(void *arg)
{
  Task *t = ((Arg *)arg)->t;
  char **args = ((Arg *)arg)->args, buf[64] = {0};
  int fdout[2], fderr[2], status, errno;
  size_t i;
  pthread_t thrd_out, thrd_err;

  for (i = 0; i < 2; i++) {
    memset(t->buf[i], 0, MAX_LINE);
    CHECK_PTH(pthread_mutex_init(&t->mtx[i], NULL));
  }

  CHECK_SYS(pipe(fdout));
  CHECK_SYS(pipe(fderr));
  set_close_on_exec(fdout[0]);
  set_close_on_exec(fderr[0]);

  t->running = 1;

  CHECK_SYS(t->pid = fork());
  if (t->pid == 0) {
    CHECK_SYS(close(fdout[0]));
    CHECK_SYS(close(fderr[0]));
    CHECK_SYS(dup2(fdout[1], STDOUT_FILENO));
    CHECK_SYS(dup2(fderr[1], STDERR_FILENO));
    CHECK_SYS(close(fdout[1]));
    CHECK_SYS(close(fderr[1]));

    CHECK_SYS(execvp(args[0], args)); /* it writes to task stderr... */
  }

  CHECK_SYS(close(fdout[1]));
  CHECK_SYS(close(fderr[1]));

  errno = pthread_create(&thrd_out, NULL, read_pipe,
                 &(Arg){.fd = fdout[0], .buf = t->buf[0], .mtx = &t->mtx[0]});
  CHECK_PTH(errno);
  errno = pthread_create(&thrd_err, NULL, read_pipe,
                 &(Arg){.fd = fderr[0], .buf = t->buf[1], .mtx = &t->mtx[1]});
  CHECK_PTH(errno);

  CHECK_PTH_BAR(pthread_barrier_wait(((Arg *)arg)->brr));
  CHECK_SYS(waitpid(t->pid, &status, 0));

  t->running = 0;

  CHECK_PTH(pthread_join(thrd_out, NULL));
  CHECK_PTH(pthread_join(thrd_err, NULL));

  CHECK_PTH(pthread_mutex_lock(&queue_mtx));

  if (WIFEXITED(status))
    sprintf(buf, "Task %ld ended: status %d.\n", t - tasks, WEXITSTATUS(status));
  else if (WIFSIGNALED(status))
    sprintf(buf, "Task %ld ended: signalled.\n", t - tasks);

  if (enqueue)
    strcat(queue, buf);
  else
    safe_printf("%s", buf);

  CHECK_PTH(pthread_mutex_unlock(&queue_mtx));

  return NULL;
}

void
task_setup(void)
{
  Task *t;
  for (t = tasks; t < tasks + MAX_TASKS; t++) {
    t->pid = -1;
    t->running = 0;
  }
}

void
task_cleanup(void)
{
  Task *t;
  for (t = tasks; t < tasks + MAX_TASKS; t++) {
    if (t->pid != -1) {
      if (t->running == 1)
        task_kill(t - tasks, SIGKILL);
      CHECK_PTH(pthread_join(t->thrd, NULL));
      CHECK_PTH(pthread_mutex_destroy(&t->mtx[0]));
      CHECK_PTH(pthread_mutex_destroy(&t->mtx[1]));
    }
  }
}

void
task_create(size_t T, char **args)
{
  Task *t = &tasks[T];
  pthread_barrier_t brr;
  CHECK_PTH(pthread_barrier_init(&brr, NULL, 2));

  CHECK_PTH(pthread_create(&t->thrd, NULL, spawn_task,
                           &(Arg){.t = t, .args = args, .brr = &brr}));

  CHECK_PTH_BAR(pthread_barrier_wait(&brr));
  CHECK_PTH(pthread_barrier_destroy(&brr));

  safe_printf("Task %ld started: pid %d.\n", T, t->pid);
}

void
task_kill(size_t T, int sig)
{
  Task *t = &tasks[T];

  if (t->running)
    kill(t->pid, sig);
}


void
task_out(size_t T, int fd)
{
  Task *t = &tasks[T];
  pthread_mutex_t *mtx = &t->mtx[fd == STDOUT_FILENO ? 0 : 1];

  CHECK_PTH(pthread_mutex_lock(mtx));
  safe_printf("Task %ld %s: \'%s\'.\n", T,
              fd == STDOUT_FILENO ? "stdout" : "stderr",
              t->buf[fd == STDOUT_FILENO ? 0 : 1]);
  CHECK_PTH(pthread_mutex_unlock(mtx));
}

void
task_block(void)
{
  CHECK_PTH(pthread_mutex_lock(&queue_mtx));
  enqueue = 1;
  memset(queue, 0, sizeof(queue));
  CHECK_PTH(pthread_mutex_unlock(&queue_mtx));
}

void
task_unblock(void)
{
  CHECK_PTH(pthread_mutex_lock(&queue_mtx));
  enqueue = 0;
  safe_printf("%s", queue);
  CHECK_PTH(pthread_mutex_unlock(&queue_mtx));
}
