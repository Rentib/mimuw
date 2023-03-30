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
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"

_Noreturn void
die(const char *fmt, ...)
{
  va_list ap;

  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);

  exit(EXIT_FAILURE);
}

void
readline(char *input)
{
  int c;
  while (isspace(c = getchar()));
  for (; c != '\n' && c != EOF; c = getchar())
    *input++ = c;
  *input = '\0';
}


void
set_close_on_exec(int fd)
{
  int flags = fcntl(fd, F_GETFD);
  CHECK_SYS(flags);
  flags |= FD_CLOEXEC;
  CHECK_SYS(fcntl(fd, F_SETFD, flags));
}
