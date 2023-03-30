cmake_minimum_required(VERSION 3.1)
project(MIMUW-FORK C)

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_C_STANDARD "11")
set(CMAKE_C_FLAGS "-pedantic -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g")

set(SOURCE_FILES
  src/executor.c
  src/util.h
  src/util.c
  src/ui.h
  src/ui.c
  src/task.h
  src/task.c)

add_executable(executor ${SOURCE_FILES})

target_link_libraries(executor pthread)
