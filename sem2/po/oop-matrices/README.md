# Reprezentacja macierzy

## Uruchamianie projektu

Program może zostać skompilowany do pojedynczego pliku `jar` z wykorzystaniem polecenia
```shell
./gradlew shadowJar
```
wynik budowania znajduje się wówczas w katalogu `build/libs` i można go uruchomić z konsoli poprzez polecenie
```shell
java -jar build/libs/oop-2022-all.jar
```
Zalecane jest jednak, na etapie tworzenia rozwiązania, korzystanie z konfiguracji przygotowywanych
przez środowiska programistyczne, takie jak Intellij, które umożliwiają
uruchomienie funkcji `main` z poziomu interfejsu użytkownika oraz łatwe debugowanie
naszego programu.

## Testowanie

Testy z katalogu `src/test` mogą zostać uruchomione poleceniem
```shell
./gradlew test
```
które powoduje uruchomienie wszystkich funkcji oznaczonych adnotacjami dostarczonymi
przez JUnit, np. `@Test`.
