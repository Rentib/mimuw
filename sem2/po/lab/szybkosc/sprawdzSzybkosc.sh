#!/bin/sh
generujTesty() {
  mkdir testy
  g++ generatorka.cpp -o generatorka
  printf "Generowanie testow...\n"
  for ((i = 1; i <= "$1"; i++)); do
    ./generatorka "$i" > testy/"$i".in
  done
  printf "Generowanie testow zakonczone.\n"
  rm generatorka
}

runOnTests() {
  for f in testy/*.in; do
    $@ < $f > /dev/null
  done
}

[ -d testy ] || generujTesty 1000

javac Main.java
g++ main.cpp -o main

echo "java"
time runOnTests java Main
echo "java -Xint"
time runOnTests java -Xint Main
echo "cpp"
time runOnTests ./main

rm *.class main
