#import "@preview/fletcher:0.5.7" as fletcher

#set page(paper: "a4", margin: 3cm)

#align(center)[
    #text(size: 17pt)[ *_Algorytmy najkrótszych ścieżek_* – Praca domowa 3 ]\
    #text(size: 15pt)[ Stanisław Bitner ]\
    #text(size: 14pt)[ 28 kwietnia 2025 ]
]

#align(center)[ ]

= 1

== (a)

Dla każdego wierzchołka sortujemy jego sąsiadów rosnąco według wag krawędzi
i po drugie po numerach wierzchołków (para: waga, indeks). To zajmuje czas
$tilde(O)(m)$.

Zaczynamy od skonstruwania zbioru $N_1(u) = {u}$ dla każdego $u in V$. Każdy
element każdego ze zbiorów dodatkowo będzie miał wskaźnik na następnego
sąsiada. Nazwijmy go następnikiem, bądź też Next.

Mając obliczone $N_k$ możemy znaleźć $N_(k+1)$ w następujący sposób:
$
    N_(k+1) (u) = N_k (u) union { v },
$
gdzie $v in { N e x t(w) | w in N_k (u) and w(w v) = min { w(a b) | (a,b) in
E and b in.not N_k (u) and a in N_k (u) } }$, czyli słownie -- wierzchołek $v$
ma być takim spośród następników wierzchołków już zawartych w $N_k (u)$,
którego dołączenie będzie najtańsze. W przypadku remisów oczywiście wybieramy
wierzchołek o niższym numerze.

Przed przystąpieniem do znajdowania następnego wierzchołka $v$, należy przejść
po wszystkich $w$ już należących do zbioru $N_k (u)$ wierzchołków
i w przypadku, w którym następnik $N e x t(w) in N_k (u)$, należy takiego
następnika przesunąć na kolejnego z posortowanych wcześniej sąsiadów tegoż
wierzchołka.

Przesuwanie wskaźników łącznie może zająć co najwyżej $O(k^2)$ dla pojedynczego
zbioru, gdyż każdy ze wskaźników, może być przesunięty jedynie $k$-krotnie
(tyle ile jest wierzchołków w zbiorze), co dla wszystkich zbiorów daje
złożoność $O(n k^2)$. Wybór kolejnego wierzchołka trywialnie zajmuje czas $O(k
    n k) = O(n k^2)$, bo w jednej iteracji musimy dla każdego zbioru przejść po
wszystkich jego elementach.

Łącznie dostajemy złożoność $tilde(O)(m + n k^2)$, co jest złożonością
oczekiwaną.

Algorytm oczywiście jest poprawny, a argument jest taki sam jak do algorytmu
Prima do konstrukcji minimalnego drzewa rozpinającego.

== (b)

Widać, że działa :). A graf musi być nieskierowany, bo dla skierowanego by nie
działało na przykład dla $G = angle.l V, E angle.r, V = {1..n},
E = { u v | u > v or u + 1 = v }$ i wtedy by nie działało.

= 2

Użyjemy algorytmu do aproksymacji długości ścieżek w nieważonych grafach
nieskierowanych, jako zamiennika do algorytmu mnożenia macierzy.

Rozważmy grafy mające trzy warstwy wierzchołków: $A, B, C$. Grafy te będą
skonstruowane w taki sposób, że krawędzie będą mogły prowadzić jedynie
z wierzchołków warstwy $A$ do $B$ oraz z $B$ do $C$.

#align(center)[
    #import fletcher: diagram, node, edge
    #fletcher.diagram($
        s edge(->) &
        circle edge(->) &
        t_1 \
        circle edge(->) edge("ur", ->) &
        circle edge(->) edge("ur", ->) &
        t_2
    $)
]

Jak zmienimy wszystkie krawędzie na nieskierowane, to dla dowolnych dwóch
wierzchołków $s in A, t in C$ istnieje ścieżka $s ~> t$ wtedy i tylko wtedy gdy
$d(s,t) < 4$. Na rysunku odległość między $s$ i $t_1$ to
$2$ dlatego $d(s,t_1) <= 1.9*2 = 3.8 < 4$.

Jeżeli $d(s,t) >= 4$ (na rysunku $s$ i $t_2$), to oznacza to, że odległość
między $s$ i $t$ musi być dłuższa niż $2$. Skoro między wierzchołkami w jednej
warstwie nie ma krawędzi, to na ścieżce musi istnieć krawędź zawracająca
z warstwy późniejszej do warsty wcześniejszej. No ale graf był skonstruowany
tak żeby krawędzie szły jedynie z wcześniejszych do późniejszych warstw, czyli
taka ścieżka musi przechodzić po krawędzi, która oryginalnie była źle
skierowana.

Tej obserwacji możemy użyć jako alternatywy do mnożenia macierzy: tworzymy nowy
graf nieskierowany $G'=({ v_A, v_B, v_C | v in V }, { u_A v_B, u_B v_C | u v in
        E })$. W ten sposób dostajemy graf składający się z właśnie trzech warstw
$A,B,C$ taki, że jeśli isnieje w nim ścieżka $s_A ~> t_C$ długości $2$, to
w oryginalnym grafie isnieje ścieżka $s ~> t$.

Na tak powstałym grafie uruchamiamy dany algorytm i tworzymy macierz $M in R^(n
times n), M_(s t) = [s_A t_B in E(G') or d(s_A, t_C) < 4]$. Jeśli $M_(s t) = 1$
i $s_A t_B in.not E(G')$, to do krawędzi grafu $G'$ dodajemy dwie nowe
krawędzie $s_A t_B$ oraz $s_B t_C$. Ten krok powtarzamy $ceil(log_2(n))$ razy.

Na końcu otrzymujemy macierz osiągalności oryginalnego grafu. Algorytm łącznie
działa w złożoności $O((T(n) + n^2)log n) = tilde(O)(T(n))$.

= 3

Do rozwiązania użyjemy algorytmu Aarona Bernsteina z 2013 roku do utrzymywania
dekrementalnego statycznego APSP. Oznaczmy przez $R$ stosunek największej
możliwej wagi, do najmniejszej: $R = (max { w(e) | e in E }) / (min { w(e)
    | e in E }) = W / 1 = W$. Algorytm ten działa w złożoności $tilde(O)(m n log(
        n
        R
    ) slash epsilon + Delta) = tilde(O)(m n log (n W) slash epsilon + Delta)$.

Algorytm ten pozwala na proste sprawdzenie, które z najkrótszych ścieżek
zostały zmienione po ostatniej aktualizacji. Jako że, odległości mogą się
jedynie zwiększać, to dla dowolnej ścieżki $s ~> t$, jej przybliżona do potęgi
$(1 + epsilon)$ długość może się zwiększyć jedynie $O(log(n R) slash epsilon)
= O(log(n W) slash epsilon)$ razy. To dla wszystkich par daje $O(n^2 log(n W)
    slash epsilon)$ aktualizacji.

Aby znajdować wartość $alpha(G)$ możemy utrzymywać min-drzewo przedziałowe dla
odległości między wszystkimi parami $s,t in P$ i aktualizować jego liście przy
każdej zmianie odległości dla tych par. To daje złożoność $O(n^2 log(n W) slash
    epsilon dot log(n))$.

Sumaryczna złożoność będzie wynosić $tilde(O)(m n log(n W) slash epsilon
    + Delta + n^2 log(n W) slash epsilon dot log(n)) = tilde(O)((m+n) n log(W)
    slash epsilon + Delta) = tilde(O)((n+m)^2 log(W) slash epsilon + Delta)$, co
jest oczekiwaną złożonością.
