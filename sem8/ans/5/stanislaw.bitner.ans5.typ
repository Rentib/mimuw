#set page(paper: "a4", margin: 3cm)

#align(center)[
  #text(size: 17pt)[ *_Algorytmy najkrótszych ścieżek_* – Praca domowa 5 ]\
  #text(size: 15pt)[ Stanisław Bitner ]\
  #text(size: 14pt)[ 15 czerwca 2025 ]
]

= 1

Zaaplikujmy potencjał. Po takiej operacji wszystkie krawędzie mają nieujemne
wagi. Długość najkrótszych ścieżek z $s$ wynosi wtedy $0$, a co za tym idzie,
wszystkie krawędzie należące do najkrótszych ścieżek zaczynających się w $s$
mają wagi zero. Zauważmy zatem, że do zbioru $F$ należą tylko krawędzie o wadze
zero. Jak teraz odwrócimy te krawędzi i zmienimy ich wagi $w$ na $-w$, to nadal
mamy tylko krawędzi nieujemne, bo $-w = -0 = 0$. A skoro wcześniej ścieżki
miały długość zero, to z nieujemnych krawędzi nie dostaniemy mniej. Dlatego też
$delta_G(s,t) <= delta_(G')(s,t)$.

= 2

Przyjmuję oznaczenia $W$ -- praca, $T$ -- głębokość.

== (a)

Dzielimy wierzchołki na $n^(2 slash 3)$ kubełków po $n^(1 slash 3)$
wierzchołków w taki sposób, że
$
  K_l = { v in V | l <= tau(v) / n^(1 slash 3) < l + 1 }.
$
Wewnątrz każdego kubełka liczymy APSP mnożąc macierze jak na
wykładzie. Następnie dodajemy do grafu krawędzie będące odległościami między
parami wierzchołków z jednego kubełka. Takich krawędzi będzie $O(n^(2 slash
  3))$ w jednym kubełku. Kosztuje nas to:
$
  W = tilde(O)(n^(2 slash 3) dot (n^(1 slash 3))^3) = tilde(O)(n^(5 slash 3)), T = tilde(O)(1).
$
Aby obliczyć SSSP dla ustalonego $s$, możemy teraz odpalić Bellmana-Forda
z ograniczeniem na długość ścieżki $n^(1 slash 3)$. W taki sposób, że w $k$-tym
kroku relaksujemy krawędzie oryginalnego grafu oraz skróty z $k$-tego kubełka.
To daje:
$
  W = tilde(O)((m + n^(2 slash 3)) dot n^(2 slash 3)) = tilde(O)(m n^(2 slash 3)), T = tilde(O)(n^(2 slash 3)).
$
Taki sposób relaksacji krawędzi (nie wszystkie na raz) zmniejsza potrzebną
pracę algorytmu. Jest to poprawne, ponieważ po $k$ iteracjach, $k$ pierwszych
kubełków ma już swoje optymalne ścieżki, gdyż z kubełka o większym numerze nie
da się dojść do kubełka o mniejszym numerze (DAG).

== (b)

Dzielimy wierzchołki na $sqrt(n)$ kubełków po $sqrt(n)$ wierzchołków w taki
sposób, że
$
  K_l = { v in V | l <= tau(v) / sqrt(n) < l + 1 }.
$
Wewnątrz każdego kubełka liczymy APSP mnożąc macierze jak na
wykładzie. Następnie dodajemy do grafu krawędzie będące odległościami między
parami wierzchołków z jednego kubełka. Takich krawędzi będzie
$O(n^(3 slash 2))$. Kosztuje nas to:
$
  W = tilde(O)(sqrt(n) dot sqrt(n)^3) = tilde(O)(n^2), T = tilde(O)(1).
$
Aby obliczyć SSSP dla ustalonego $s$, możemy teraz na zmodyfikowanym grafie
odpalić Bellmana-Forda z ograniczeniem na długość ścieżki $sqrt(n)$. To daje:
$
  W = tilde(O)((m + n^(3 slash 2)) dot sqrt(n)) = tilde(O)(m sqrt(n) + n^2), T = tilde(O)(sqrt(n)).
$
Każda najkrótsza ścieżka ma co najwyżej $sqrt(n)$ krawędzi, ponieważ numer
kubełka może się tylko zwiększać, a wewnątrz kubełka nie ma sensu chodzić po
więcej niż dwóch wierzchołkach -- wejście do kubełka, przejście jednym
"skrótem" i wyjście z kubełka.

= 3

== (a)

Dzielimy wierzchołki na kubełki
$
  B_i = { v | k i <= pi(v) < (k+1) i }.
$
Na tych przedziałach budujemy drzewo binarne w taki sposób, że węzeł drzewa $X$
będący przodkiem liści pokrywających łączny przedział $P_X = [l,r]$ ma zdefiniowane
$
  S_X = { v | (l+r)/2-k/2 <= pi(v) <= (l+r)/2+k/2 }.
$

Dla każdego węzła $X$ odpalamy $O(k)$ Dijkstr ze źródłami z $S_X$ na podgrafie
indukowanym przez $P_X$ normalnie oraz z odwróconymi krawędziami. Wyniki
zapisujemy w $delta_X$ i $delta^R_X$. Zauważmy, że każda krawędź pojawia się w takim
indukowanym grafie $tilde(O)(1)$ razy. Zajmuje to łącznie czas
$
  tilde(O)(sum_X |S_X| dot E(X)) = tilde(O)(m k).
$

Aby znaleźć $delta(s,t)$ wystarczy teraz znaleźć wpierw węzły drzewa $L,R$
takie, że $pi(s) in P_L, pi(t) in P_R$. Następnie wyznaczamy LCA tych węzłów --
$X$. Wówczas
$
  delta(s,t) = min_(v in S_X) { delta^R_X(v,s) + delta(v,t) }.
$
Ścieżka $s ~> t$ musi przechodzić przez któryś z wierzchołków należących do
$S_X$, gdyż krawędzie mogą zmienić $pi$ o co najwyżej $k$, a wierzchołki
z $S_X$ pokrywają spójny przedział $k$ wartości przypisania $pi$ i mają swoje
wartości między wartościami $s$ i $t$.

To daje konstrukcję w czasie $tilde(O)(m k)$ i odpowiedzi w czasie
$tilde(O)(k)$.

== (b)

Zauważmy, że $m <= n k, k m <= n k^2$, więc ten podpunkt wynika z (a). Wynika
to z tego, że każdy wierzchołek może mieć stopień co najwyżej $k$, inaczej $k$
nie było by maksymalne. To daje: $m <= n dot max { "deg"(v) | v in V } <= n k$.
