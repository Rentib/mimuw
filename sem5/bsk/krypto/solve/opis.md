<!-- Numer indeksu: 438247                                                     -->
<!-- Nazwa zadania: Zadanie zaliczeniowe - kryptografia                        -->
<!-- Flaga1:        flag{still-not-a-csprng}                                   --> 
<!-- Flaga2:        flag{sh0rt-fl4G}                                           -->
<!-- Flaga3:        flag{p4dding-1s-h4rd-but-re4ly-just-s1gn-y0ur-c1phert3xts} -->
<!-- opis:          opis.md                                                    -->

# Flaga1
Mamy dane m=1<<64, s1, s2, s3, s4, s5
szukamy pary (a,c) takiej, że
s_{i+1} mod m = (s_i * a ^ c) mod m
m = 1<<64, więc równania powinny działać dla każdego k=0..64 mod 1<<k
najpierw ma się zgadzać dla k=0, czyli
s_{i+1} mod 1<<k = (s_i * a ^ c) mod 1<<k
0 = 0
jest tylko jedna możliwość dla (a,c) czyli (0,0)
jak chcemy zwiększyć k, to dokładamy każdą możliwą kombinację wyższych
bitów do każdej dotychczasowej pary (a,c)

    ((a | (0<<k), c | (0<<k)))
    ((a | (0<<k), c | (1<<k)))
    ((a | (1<<k), c | (0<<k)))
    ((a | (1<<k), c | (1<<k)))

Potem każdą tak otrzymaną parę sprawdzamy z czterema równaniami
i zostawiamy tylko te pary, które spełniają wszystkie. Okazuje się, że
par jest w każdym momencie dość mało ~ O(1), więc złożoność to
O(1 * 4 * 64) = O(1)
*  4, bo każda para daje 4 nowe pary
* 64, bo liczba jest 64bitowa
Ostatecznie mamy propozycje liczb (a,c), z których wybieramy dowolną
i podstawiamy do wzoru żeby otrzymać s6. Czasami to nie działa (niektóre
pary są błędne) ale to nie jest problem, bo wystarczy całość powtórzyć.
I tak to jest po prostu łączenie się z serwerem, wybranie zadania
i wysłanie jednej liczby, więc nikt się nie zorientuje.

# Flaga2

Mamy szyfr blokowy.
Znamy zaszyfrowane hello, na podstawie tego generujemy zaszyfrowane 'flag?' i wysyłamy to do serwera.
Otrzymujemy zaszyfrowaną pierwszą flagę.
Wiemy, że flagi zaczynają się na 'flag{'.
To pozwala nam bruteforcować kolejne bajty.
Funkcja w kodzie to *get_block_based_on_5_bytes()*

# Flaga3

Tutaj jest tak samo, ale flaga może być dłuższa niż 1 block, więc
nie możemy jej znaleźć dokładnie tak samo, bo możemy odgadywać tylko
pojedyncze bloki na podstawie 5 bajtów bloku.
Pierwszy blok robimy tak samo jak w fladze 2, bo wiemy, że zaczyna się na 'flag{'
tylko używamy zapytania 'FLAG!'.
Potem używamy zapytania 'hash?' na sufiksie, aby znaleźć pierwsze
5 bajtów kolejnego bloku wiedząc, że jego IV to CT poprzedniego bloku. Dokładny opis to kod.
Jak mamy 5 bajtów na prefiksie, to używamy funkcji
*get_block_based_on_5_bytes()* i powtarzamy aż trafimy na znak '}'
