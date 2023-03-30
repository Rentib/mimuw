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

#include <ctype.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ui.h"
#include "util.h"
#include "task.h"

typedef struct {
  const char *cmd;
  void (*func)(char *);
} Parser;

static void ui_run(char *);
static void ui_out(char *);
static void ui_err(char *);
static void ui_kill(char *);
static void ui_sleep(char *);
static void ui_quit(char *);

static int running = 1;
static size_t T = 0; /* new task number */

static const Parser parsers[] = {
  { "run",   ui_run },
  { "out",   ui_out },
  { "err",   ui_err },
  { "kill",  ui_kill },
  { "sleep", ui_sleep },
  { "quit",  ui_quit },
};

void
ui_run(char *input)
{
  size_t i = 0;
  char *args[512];

  args[i] = strtok(input, " ");
  while (args[i] != NULL)
    args[++i] = strtok(NULL, " ");

  task_create(T++, args);
}

void
ui_out(char *input)
{
  size_t tid = atoi(input);
  if (tid < T)
    task_out(tid, STDOUT_FILENO);
  else
    printf("Task %ld does not exist.\n", tid);
}

void
ui_err(char *input)
{
  size_t tid = atoi(input);
  if (tid < T)
    task_out(tid, STDERR_FILENO);
  else
    printf("Task %ld does not exist.\n", tid);
}

void
ui_kill(char *input)
{
  size_t tid = atoi(input);
  if (tid < T)
    task_kill(tid, SIGINT);
  else
    printf("Task %ld does not exist.\n", tid);
}

void
ui_sleep(char *input)
{
  usleep(atoi(input) * 1000);
}

void
ui_quit(char *input)
{
  (void) input; /* silence warnings */
  running = 0;
}

void
ui_loop(void)
{
  char input[MAX_LINE];
  const Parser *p;
  size_t i, len;

  while (running) {
    memset(input, 0, sizeof(input));
    readline(input);

    task_block();
    if (feof(stdin)) {
      ui_quit(NULL);
      task_unblock();
      continue;
    }

    for (i = 0, p = parsers; i < LENGTH(parsers); i++, p++) {
      len = strlen(p->cmd);
      if ((!input[len] || isspace(input[len]))
      &&  !strncmp(input, p->cmd, len)) {
        p->func(input + len + 1); /* add 1 for space */
        break;
      }
    }
    if (i == LENGTH(parsers))
      printf("Unknown command: \'%s\'.\n", input);
    task_unblock();
  }
}
