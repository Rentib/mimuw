#!/bin/sh

PROG="$1"
DIR="$2"

[ -z "$PROG" ] && printf "Usage: test.sh program directory\n" && exit 1
[ -z "$DIR" ] && printf "Usage: test.sh program directory\n" && exit 1
[ ! -f "$PROG" ] && printf "\e[92;1m%s\e[0m does not exist.\n" "$PROG" && exit 1
[ ! -d "$DIR" ] && printf "\e[34;1m%s\e[0m does not exist.\n" "$DIR" && exit 1

ansok=0
errok=0
memok=0
tests=0

printf "Checking \e[92;1m%s\e[0m.\n" "$PROG"
for test in $DIR/*.in; do
  tests=$((tests + 1))
  test="${test%.in}"
  printf "%s\n" "$test"

  valgrind --error-exitcode=69 \
           --leak-check=full \
           --show-leak-kinds=all \
           --errors-for-leak-kinds=all \
           --quiet \
           --log-fd=3 \
           ./"$PROG" < "$test.in" 1>prog.out 2>prog.err 3>val.err
  valgrindval=$?

  printf "ANS: "
  diff prog.out "$test.out" >/dev/null 2>&1 \
    && printf "\e[92mOK\e[0m\n" && ansok=$((ansok + 1)) \
    || printf "\e[31mWA\e[0m\n"
  
  printf "ERR: "
  diff prog.err "$test.err" >/dev/null \
    && printf "\e[92mOK\e[0m\n" && errok=$((errok + 1)) \
    || printf "\e[31mWA\e[0m\n"

  printf "MEM: "
  [ "$valgrindval" -ne "69" ] \
    && printf "\e[92mOK\e[0m\n" && memok=$((memok + 1)) \
    || printf "\e[31mME\e[0m\n"

done

printf "\nTotal results:\n"
printf "\e[37;1m%d/%d\e[0m tests with correct answer.\n" "$ansok" "$tests"
printf "\e[37;1m%d/%d\e[0m tests with correct error.\n" "$errok" "$tests"
printf "\e[37;1m%d/%d\e[0m tests without memory errors.\n" "$memok" "$tests"

rm prog.out prog.err val.err
