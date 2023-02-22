Szybkość programów w Javie - Otoczka Wypukła
--------------------------------------------
Otoczka wypukła obliczana algorytmem [*Monotone Chain (Andrew's Algorithm)*](https://en.wikipedia.org/wiki/Convex_hull_algorithms).

Informacje
----------
Języki: java / cpp
Środowisko: Linux 5.17.1-arch1-1 x86_64
Wersja javy: openjdk 11.0.15 2022-04-19
             OpenJDK Runtime Environment (build 11.0.15+3)
             OpenJDK 64-Bit Server VM (build 11.0.15+3, mixed mode)
Wersja g++: g++ (GCC) 11.2.0

Czasy mierzone komendą *time*:

java: 
real 3m5.558s
user 5m46.781s
sys 0m21.025s

java -Xint:
real 11m54.133s
user 11m28.827s
sys 0m15.145s

cpp:
real 0m43.998s
user 0m42.477s
sys 0m1.433s

Powtarzanie
-----------
Aby powtórzyć pomiar wystarczy wpisać komendę:
  
    $ ./sprawdzSzybkosc.sh

Testy zostaną wtedy wygenerowane i umieszczone w folderze o nazwie *testy*
