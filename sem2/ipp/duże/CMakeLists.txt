cmake_minimum_required(VERSION 3.0)
project(phone_numbers C)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif ()

# Nie chcemy oglądać komunikatów o procentowym postępie kompilacji.
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)

# Chcemy zobaczyć polecenia wywołane przez make.
set(CMAKE_VERBOSE_MAKEFILE ON)

# Ustawiamy wspólne opcje kompilowania dla wszystkich wariantów projektu.
set(CMAKE_C_FLAGS "-std=c17 -Wall -Wextra -Wno-implicit-fallthrough")
# Domyślne opcje dla wariantów Release i Debug są sensowne.
# Jeśli to konieczne, ustawiamy tu inne.
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
# set(CMAKE_C_FLAGS_DEBUG "-g")

# Wskazujemy pliki źródłowe.
set(SOURCE_FILES
    src/number.h
    src/number.c
    src/phone_forward.h
    src/phone_forward.c
    src/trie.h
    src/trie.c
    src/circular_list.h
    src/circular_list.c
    src/vector.h
    src/vector.c
    src/phone_forward_example.c)
set(SOURCE_FILES_TEST
    src/number.h
    src/number.c
    src/phone_forward.h
    src/phone_forward.c
    src/trie.h
    src/trie.c
    src/circular_list.h
    src/circular_list.c
    src/vector.h
    src/vector.c
    src/phone_forward_tests.c)

# Wskazujemy plik wykonywalny.
add_executable(phone_forward ${SOURCE_FILES})
add_executable(phone_forward_test ${SOURCE_FILES_TEST})
add_executable(phone_forward_instrumented ${SOURCE_FILES_TEST})

target_link_options(phone_forward_instrumented PUBLIC -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=reallocarray -Wl,--wrap=free -Wl,--wrap=strdup -Wl,--wrap=strndup)

# Dodajemy obsługę Doxygena: sprawdzamy, czy jest zainstalowany i jeśli tak to:
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # Wskazujemy lokalizacją pliku konfiguracyjnego i podajemy jego docelową lokalizację w folderze, gdzie następuje kompilacja.
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    # Dodajemy cel doc: użycie make doc spowoduje wywołanie doxygena z odpowiednim plikiem konfiguracyjnym w folderze kompilacji.
    # Na wyjście zostanie wypisany odpowiedni komentarz.
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
    )
endif (DOXYGEN_FOUND)
