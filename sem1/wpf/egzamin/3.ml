(* niech m oznacza dlugosc danego napisu *)
(* niech n oznacza rozmiar danego drzewa *)
(* zlozonosc pamieciowa i czasowa wynikaja ze zlozonosci funkcji pref oraz dfsa *)
(* zlozonosc pamieciowa O(m + n) *)
(* zlozonosc czasowa    O(m + n) *)

(* na egzaminie bylo powiedziane ze tak drzewo jak i dany wzorzec skladaja sie jedynie z liter *)
(* robimy kmp na dynamicznie tworzonych napisach, zlozonosc i tak sie zamortyzuje do O(n) *)
(* innym i znacznie prostszym do wymyslenie sposobem byloby haszowanie 
   ale ocaml nie jest zbyt przyjazny jesli chodzi o takie metody *)

type tree = Node of char * tree * tree | Null

(* algorytm ze smurfa *)
let pref t =
    let p = Array.make (Array.length(t) + 1) 0
    and pj = ref 0 in
    begin
        for i = 2 to Array.length t do
            while (!pj > 0) && (t.(!pj) <> t.(i - 1)) do
                pj := p.(!pj)
            done;
            if t.(!pj) = t.(i - 1) then pj := !pj + 1;
            p.(i) <- !pj
        done;
        p
    end
;;

let napis given_pattern root = 
  let m = Array.length given_pattern in
  let pattern = Array.append given_pattern (Array.make 1 '#') in (* dodajemy wartownika *)
  let p = pref pattern in (* prefikso sufiksy *)
  (* warto wspomniec ze nie potrzebujemy wiekszej tablicy gdyz mamy wartownika, wiec bedziemy sprawdzac indexy <= m *)
  let res = ref 0 in
  let rec dfs node i = 
    let j = ref i in (* deklaracja tutaj a nie pozniej, bo ocaml moze cos popsuc *)
    match node with
    | Null -> ()
    | Node(x, l, r) -> (
        (* let j = ref i in *) (* deklaracja przeniesiona - tutaj sie kompiluje ale nie jestem pewny czy dziala tak wlasciwie *)
        while !j > 0 && x <> pattern.(!j) do (
          j := p.(!j)
        ) done;
        if x = pattern.(!j) then j := !j + 1;
        (* nie trzeba zapisywac j w tablicy p z powodu juz opisanego powyzej *)
        res := !res + (if !j = m then 1 else 0);
        dfs l !j;
        dfs r !j;
      )
  in
  dfs root 0; (* po prostu funkcja pref na dynamicznie robionej tablicy *)
  !res
;;

(* niedokonczone rozwiozanie uzywajace haszy *)

(* let napis give_pattern root = *) 
(*   let m = Array.length given_pattern in *)
(*   let q = 1000000007 *)
(*   and p = 997 in *)
(*   let top = ref 0 in *)

(*   let coP = Array.make (size root) '#' in *)
  
(*   let rec dfs node depth phasz nodehasz plength cfP = *) 
(*     match node with | Null -> 0 | Node(x, l, p) -> ( *)
(*       coP.(!top) <- cfP; *)
(*       top := !top + 1; *)
(*       let patternHash = ref (phasz * p + (Char.code cfP)) % mod in *)
(*       if !top > patt then ( *)
(*         patternHash := (!patternHash - coP.(top - plength - 1) * ModPow(B, plength) % mod + mod) % mod; *)
(*       ); *)
(*       let res = ref 0 in *)
(*       if top >= plength && patternHash = nodehasz then result := !result + 1; *)
(*       !result = !result + (dfs *) 
(*     ) *)
(*   in *)
(* ;; *)


(* char charactersOnPath[MAX_N]; *)
(* int top = 0; *)

(* //root is at depth = 0 *)
(* int DFS(Node *n, int depth, int patternHash, int nodeHash, int patternLenght, char charFromParent) { *)
(*     charactersOnPath[top++] = charFromParent; *)
(*     patternHash = (patternHash * B + charFromParent) % mod; *)
(*     if(top > patternLength) { *)
(*         patternHash = (patternHash - characterOnPath[top - patternLength - 1] * ModPow(B, patternLength) % mod + mod) % mod; *)
(*     } *)
(*     int result = 0; *)
(*     if(top >= patternLength && patternHash == nodeHash) *)
(*         result++; *)
(*     for(Edge e : n -> children) { *)
(*         result += DFS(e.to, depth + 1, patternHash, nodeHash, patternLength, e.character); *)
(*     } *)
(*     top--; *)
(*     return result; *)
(* } *)
