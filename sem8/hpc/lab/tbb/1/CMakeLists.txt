cmake_minimum_required (VERSION 3.5)
project (MIMUW-tbb-lab1 CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")

find_package ( TBB REQUIRED )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-g -Wall")

add_executable(parfor parfor.cpp)
target_link_libraries(parfor tbb)

#add_executable(parfor-counter parfor-counter.cpp)
#target_link_libraries(parfor-counter tbb)

add_executable(tasks-fib tasks-fib.cpp)
target_link_libraries(tasks-fib tbb)

add_executable(nqueens nqueens.cpp)
target_link_libraries(nqueens tbb)

install(TARGETS DESTINATION .)
