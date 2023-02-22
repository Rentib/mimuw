(* Origami      - Stanislaw Bitner *)
(* Sprawdzajacy - Paweł Pilarski   *)

type point = float * float;;
type kartka = point -> int;;

let sq t = t *. t;; (* funkcja pomocnicza, oblicza kwadrat floata *)

(*
   P2 (lewo)
  /  \ 
 /    \
A------B----P3 (srodek)
 \    /
  \  /
   P1 (prawo)
*)

type orientacja = Lewo | Srodek | Prawo;;
(* otrzymuje jako argumenty 3 punkty (a <> b), zwraca: *)
(* ujemna liczba -> lewo, 0 -> srodek, dodatnia liczba -> prawo *)
let oblicz_orientacje ((ax, ay) : point) ((px, py) : point) ((bx, by) : point) = 
  let wartosc = (bx -. ax) *. (py -. by) -. (by -. ay) *. (px -. bx) in
  if wartosc < 0. then Lewo   else
  if wartosc = 0. then Srodek else
                       Prawo
;;

(*
     O
     |
     |
P------------Q
     |
     |
     O' (obraz punktu O względem prostej zdefiniowanej przez PQ)
*)

(* otrzymuje jako argumenty 3 punkty (P <> Q), *)
(* oblicza obraz punktu O względem prostej zdefiniowanej przez punkty P oraz Q *)
let obraz ((px, py) : point) ((qx, qy) : point) ((ox, oy) : point) = 
  (* prosta PQ jest zdefiniowana poprzez rownanie Ax + By + C = 0 *)
  let a = px -. qx in
  let b = qy -. py in
  let c = -.a *. py -. b *. px in
  let x = (ox *. (sq a -. sq b) -. 2. *. b *. (a *. oy +. c)) /. (sq a +. sq b) in
  let y = (oy *. (sq b -. sq a) -. 2. *. a *. (b *. ox +. c)) /. (sq a +. sq b) in
  ((x, y) : point)
;;

(*
        Q1

    +-----------------P2
    |                 |
Q4  |      Q          |   Q2
    |                 |
    P1----------------+

              Q3
*)

let prostokat ((p1x, p1y) : point) ((p2x, p2y) : point) =
  (function ((qx, qy) : point) -> if (p1x <= qx && qx <= p2x) && (p1y <= qy && qy <= p2y) then 1 else 0 : kartka)
;;

(*
         , - ~ ~ ~ - ,
     , '               ' ,
   ,                       ,
  ,                         ,
 ,                           ,
 ,             P             ,
 ,                           ,
  ,                  Q      ,
   ,                       ,
     ,                  , '
       ' - , _ _ _ ,  '      Q1
*)

(* niestety trzeba użyć sqrt, gdyż jeśli by się po prostu brało kwadrat tego równania to floaty mogą wyjść poza zakres *)
let kolko ((px, py) : point) (r : float) = 
  (function ((qx, qy) : point) -> if sqrt(sq (px -. qx) +. sq (py -. qy)) <= r then 1 else 0 : kartka)
;;

let zloz (a : point) (b : point) (k : kartka) = 
  (function (p : point) ->
    let (q : point) = obraz a b p in
    match (oblicz_orientacje a p b) with
    | Lewo   -> 0
    | Srodek -> k p
    | Prawo  -> k p + k q
  )
;;

let skladaj l (k : kartka) =
  List.fold_left (fun w (p1, p2) -> zloz p1 p2 w) k l
;;
