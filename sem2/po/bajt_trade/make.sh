#!/bin/sh
question() {
  read -p "$1 [y/N]" -n 1 -r
  printf "\n"
  if [[ !($REPLY =~ ^[Yy]$) ]]; then
    printf "Nie można kontynuować.\n"
    exit 0
  fi
}

question "Czy gradle jest zainstalowany?"

./gradlew build
./gradlew shadowJar
printf "\nProszę podać ścieżkę do pliku wejściowego.\n"
read path

java -jar build/libs/bajt_trade-1.0-all.jar "$path"
