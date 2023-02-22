(* przejdz po drzewie, 
   sprawdz ksztalt (wierzcholki puste musza byc w tych samych miejscach),
   sprawdz czy funkcja dla danego x zwraca dokladnie 1 y *)

(* niech n oznacza mniejszy z rozmiarow obu drzew *)
(* zlozonosc pamieciowa O(n) *)
(* zlozonosc czasowa    O(n) *)

type tree = Node of int * tree * tree | Null;;
let rec map_tree f = function
    | Null -> Null
    | Node (x, l, p) -> Node (f x, map_tree f l, map_tree f p) ;;

let podobne root1 root2 = 
  let function_res = Hashtbl.create 424242 in
  let rec dfs node1 node2 = 
    match (node1, node2) with
    | (Null, Null) -> true
    | (Null, _)    -> false (* inny ksztalt *)
    | (_, Null)    -> false (* inny ksztalt *)
    | (Node(x, l1, p1), Node(y, l2, p2)) ->
        if Hashtbl.mem function_res x = false then ( (* jesli nie bylo wczesniej x to wartosc funkcji dla x bedzie y *)
          Hashtbl.add function_res x y;
        );
        if Hashtbl.find function_res x = y then (* sprawdzamy czy wartosc funkcji w x zgadza sie z y *)
          (dfs l1 l2) && (dfs p1 p2)
        else
          false
  in
  dfs root1 root2
;;

