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

#ifndef EXECUTOR_UTIL_
#define EXECUTOR_UTIL_

#define MAX_TASKS 4096
#define MAX_LINE 1024
#define LENGTH(X) (sizeof X / sizeof X[0])
#define CHECK_SYS(X) do { if ((X) == -1) die(#X); } while (0)
#define CHECK_PTH(X)                                                           \
  do {                                                                         \
    int pth_err = (X);                                                         \
    if (pth_err != 0)                                                          \
      die(#X);                                                                 \
  } while (0)
#define CHECK_PTH_BAR(X)                                                       \
  do {                                                                         \
    int pth_err = (X);                                                         \
    if (pth_err != 0 && pth_err != PTHREAD_BARRIER_SERIAL_THREAD)              \
      die(#X);                                                                 \
  } while (0)

extern _Noreturn void die(const char *fmt, ...);
extern void readline(char *input);
extern void set_close_on_exec(int fd);

#endif /* EXECUTOR_UTIL_ */
