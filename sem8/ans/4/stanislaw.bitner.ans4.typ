#import "@preview/fletcher:0.5.7" as fletcher

#set page(paper: "a4", margin: 3cm)

#align(center)[
    #text(size: 17pt)[ *_Algorytmy najkrótszych ścieżek_* – Praca domowa 4 ]\
    #text(size: 15pt)[ Stanisław Bitner ]\
    #text(size: 14pt)[ 25 maja 2025 ]
]

= 1

Niech $k = O(n)$. Najpierw tworzymy dwie ścieżki:

#align(center)[
    #import fletcher: diagram, node, edge
    #fletcher.diagram($
        u_1 edge() &
        u_2 edge() &
        ... edge() &
        u_(k-1) edge() &
        u_k
        \
        v_1 edge() &
        v_2 edge() &
        ... edge() &
        v_(k-1) edge() &
        v_k
    $)
]

Następnie dodajemy $k$ krawędzie kolejno łącząc wierzchołki $u_i, v_i$.

#align(center)[
    #import fletcher: diagram, node, edge
    #fletcher.diagram($
        u_1 edge() edge("d", e_1) &
        u_2 edge() edge("d", e_2) &
        ... edge() edge("d", "--") &
        u_(k-1) edge() edge("d", e_(k-1)) &
        u_k edge("d", e_k)
        \
        v_1 edge() &
        v_2 edge() &
        ... edge() &
        v_(k-1) edge() &
        v_k
    $)
]

Dodanie krawędzi $e_i$ zmniejsza odległości między każdą parą wierzchołków $u
in { u_j | j >= i }, v in { v_j | j >= i }$. Takich par jest $(k-i+1)^2$.

To daje łącznie $sum_(i=1)^k (k-i+1)^2 = 1/6 k(2k^2+3k+1) = Theta(k^3)
= Theta(n^3)$ zmian.

= 2

== (a)

#align(center)[
    #import fletcher: diagram, node, edge
    #fletcher.diagram($
      circle edge(e_1) &
      circle edge(e_2) &
      ...    edge(e_(n-2)) &
      circle edge(e_(n-1)) &
      circle
    $)
]

$F$ konstruujemy zachłannie przechodząc po krawędziach $e_1, e_2, ...$ tak, że
trzymamy dotychczasową sumę wag krawędzi i jeśli przekroczy ona $D$, to usuwamy
najnowszą krawędź i zerujemy sumę. To oczywiście spowoduje usunięcie co
najwyżej $W / D$ krawędzi, a co za tym idzie $|F| <= W/D$.

== (b)

Dowód przez indukcje. Zauważmy, że dla $|V| = 1$, $|F| = 0 = 2 W/D$.
Przypuśćmy, że dla każdego $|V| <= n$ da się zrobić LDD spełniające warunki.
Rozważmy drzewo o $n+1$ wierzchołkach.

Niech $u$ będzie takim wierzchołkiem, że
$forall_(v in "son"(u)) "diameter"("subtree"(v)) <= D < "diameter"("subtree"(u))$.

Niech $W_v = W("subtree"(v)), D_v = "diameter"("subtree"(v)), H_v
= "height"("subtree"(v))$.

Jako że, dla każdego syna $u$ średnica jego poddrzewa jest mniejsza niż $D$,
a $D_u >= D$, to dla pewnych par synów wierzchołka $u$: $v_1, v_2$ zachodzi $D
< H_v_1 + w({v_1,u}) + w({u, v_2}) + H_v_2$. Usuwamy $k$ najcięższych krawędzi
$u$ z synami tak, aby ostatecznie $D_u <= D$. Usunięte krawędzie oznaczmy jako
${u,v_1},...,{u,v_k}$. Zauważmy, że $forall_i w({u,v_i}) + H_v_1 >= D/2 and
D_v_i <= D$. Potem robimy LDD na pozostałym grafie.
$
|F|
&<= k + 2 (W - (W_v_1 + w({u,v_1}) + ... + W_v_k + w({u,v_k})))/D \
&<= k + 2 (W - (H_v_1 + w({u,v_1}) + ... + H_v_k + w({u,v_k})))/D \
&<= k + 2(W - k dot D/2)/D = k + 2W/D - k = 2W/D
$

Zatem na mocy zasady indukcji matematycznej własność zachodzi.

= 3

O ile nie wiem jak zrobić w $O(n^(3-epsilon))$, to w $O(n^3)$ jest dość
łatwo. Zauważmy, że odległość od każdego wierzchołka może się jedynie zwiększać
i nie przekekracza ona $n$. To oznacza, że dla każdego wierzchołka mamy $O(n)$
zmian.

Przy każdym dodaniu krawędzi $u v$ odpalamy bfs z wierzchołka $v$ z odległością
odpaloną na mniejszą z dotychczasowej oraz z odległości od $u$ zwiększonej
o $1$. Potem odwiedzamy jedynie wierzchołki, których odległości zmniejszamy. To
daje nam złożoność $O(n dot n dot "deg"(n)) = O(n^3)$.

Wszystkie najkrótsze ścieżki przechowujemy w tablicach jako pary elementów
$("czas", "odległość")$, co pozwala na odpowiadanie na zapytania o $delta(v, t)$
w czasie $O(log(n))$ wyszukując binarnie ostatnią parę $(t', d)$ w liście
wierzchołka $v$ taką, że $t' <= t$.
