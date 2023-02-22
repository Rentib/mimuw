(*     Arytmetyka  - Stanislaw Bitner *)
(* 1. Sprawdzajacy - Kamil Pilkiewicz *)
(* 2. Sprawdzajacy - Marek Zbysinski  *)
(* Jestesmy w grupie trzyosobowej poniewaz na cwiczeniach jest 15 osob, zostala wydana zgoda prowadzacego laboratorium - Doktora Piotra Skowrona *)
(* Nie umiem matematyki ale znalazlem ten artykul      - https://en.wikipedia.org/wiki/Interval_arithmetic              *)
(* Znalazlem takze bardzo ladny pdf zawierajacy dowody - http://fab.cba.mit.edu/classes/S62.12/docs/Hickey_interval.pdf *)

(* typ *)
type wartosc = Z of float * float | P of float * float | Pusty;; (* Z - zwykly [a, b], P - podwojny (-inf, a] U [b, +inf), Pusty - pusty *)

(* konstruktory *)

let wartosc_dokladnosc x p =
  let dolne = x *. ((100. -. p) /. 100.) in
  let gorne = x *. ((100. +. p) /. 100.) in
  Z(min dolne gorne, max dolne gorne)
;;

let wartosc_od_do  x y = Z(x, y);;

let wartosc_dokladna x = Z(x, x);;

(* selektory *)

let in_wartosc x y = 
  match x with
  | Z(a, b) -> a <= y && y <= b
  | P(a, b) -> y <= a || b <= y
  | _       -> false
;;

let min_wartosc = function
  | Z(a, _) -> a
  | P(_, _) -> neg_infinity
  | _       -> nan
;;

let max_wartosc = function
  | Z(_, b) -> b
  | P(_, _) -> infinity
  | _       -> nan
;;

let sr_wartosc x = (min_wartosc x +. max_wartosc x) /. 2.;;
(* dla x = P(_, _) bedzie nan bo ocaml zwraca (inf + neg_inf = nan) *)

(* funkcje pomocnicze *)

let nie_nan funkcja w1 w2 = 
  if classify_float w1 = FP_nan then w2 else
  if classify_float w2 = FP_nan then w1 else
  funkcja w1 w2
;;

let mnozenie a b c d = (* przyjmuje 4 iloczyny i zwraca najmniejszy oraz najwiekszy z nich *)
  let extremum funkcja a b c d = nie_nan funkcja (nie_nan funkcja a b) (nie_nan funkcja c d) in
  let minimum = extremum min a b c d in
  let maximum = extremum max a b c d in
  Z(minimum, maximum)
;;

let rec zlacz = function (* laczenie przedzialow [a, b] [c, d] -> [a, b] U [c, d] *)
  (* dla uproszczenia chcemy aby przedzial [a, b] byl "na lewo" od [c, d] *)
  | (Z(a, b), Z(c, d))                      when c < a            -> zlacz ((Z(c, d), Z(a, b)))
  | (P(a, b), P(c, d))                      when c < a            -> zlacz ((P(c, d), P(a, b)))
  | (Z(a, b), Z(c, d))                      when b < c            -> P(b, c)
  | (Z(a, b), Z(c, d))                                            -> Z(min a c, max b d)
  | (Z(a, b), P(c, d)) | (P(c, d), Z(a, b)) when c >= a && b >= d -> Z(neg_infinity, infinity)
  | (Z(a, b), P(c, d)) | (P(c, d), Z(a, b)) when c >= a           -> P(max c b, d)
  | (Z(a, b), P(c, d)) | (P(c, d), Z(a, b)) when b >= d           -> P(min d a, c)
  | (P(a, b), P(c, d))                      when c >= b           -> Z(neg_infinity, infinity)
  | (P(a, b), P(c, d))                                            -> P(max a c, min b d)
  | _                                                             -> Pusty
;;

let rec odwrotnosc = function (* dla przedzialu [a, b] zwraca (1 / [a, b]) *)
  (* przez to ze trzeba pisac kropki, kod jest nierowny i brzydki... *)
  | Z(a, 0.)                        -> Z(neg_infinity, 1. /. a)
  | Z(0., b)                        -> Z(1. /. b, infinity)
  | Z(a, b) when a <= 0. && 0. <= b -> P(1. /. a, 1. /. b)
  | Z(a, b)                         -> Z(1. /. b, 1. /. a)
  | P(a, b)                         -> zlacz (odwrotnosc (Z(neg_infinity, a)), odwrotnosc (Z(b, infinity))) 
  | _                               -> Pusty
;;

(* modyfikatory *)

let plus w1 w2 = (* suma przedzialow w1 w2 -> w1 U w2 *)
  match (w1, w2) with
  | (Z(a, b), Z(c, d))                       -> Z(a +. c, b +. d)
  | (Z(a, b), P(c, d)) when b +. c >= a +. d -> Z(neg_infinity, infinity)
  | (Z(a, b), P(c, d))                       -> P(c +. b, d +. a)
  | (P(a, b), Z(c, d)) when d +. a >= c +. b -> Z(neg_infinity, infinity)
  | (P(a, b), Z(c, d))                       -> P(a +. d, b +. c)
  | (P(_, _), P(_, _))                       -> Z(neg_infinity, infinity)
  | _                                        -> Pusty
;;

let minus w1 w2 = (* roznica przedzialow w1 w2 -> w1 - w2 *)
  match (w1, w2) with
  | (Z(a, b), Z(c, d)) -> Z(a -. d, b -. c)
  | (Z(a, b), P(c, d)) -> P(b -. d, a -. c)
  | (P(a, b), Z(c, d)) -> P(a -. c, b -. d)
  | (P(_, _), P(_, _)) -> Z(neg_infinity, infinity)
  | _                  -> Pusty
;;

let rec razy w1 w2 = (* iloczyn przedzialow w1 w2 -> w1 * w2 *)
  match (w1, w2) with
  | (Pusty, _) | (_, Pusty) -> Pusty      (* jesli dowolny z przedzialow jest pusty to mnozenie nie ma sensu, warunek ten jest na poczatku aby uniknac dalszej ifologii *)
  | (Z(0., 0.), _)          -> Z(0., 0.)  (* jesli mnozymy razy [0, 0] to otrzymujemy [0, 0] - warto wspomniec, ze ocaml domyslnie tego nie robi i w przypadku +/- inf * 0 zwraca nan *)
  | (_, Z(0., 0.))          -> Z(0., 0.)  (* ---------------------------------------------------------------------------------------------------------------------------------------- *)
  | (P(a, b), _)            -> zlacz (razy (Z(neg_infinity, a)) w2, razy (Z(b, infinity)) w2) (* rozbijamy podwojny przedzial na 2 pojedyncze przedzialy *)
  | (_, P(c, d))            -> zlacz (razy w1 (Z(neg_infinity, c)), razy w1 (Z(d, infinity))) (* ------------------------------------------------------- *)
  | (Z(a, b), Z(c, d))      -> mnozenie (a *. c) (a *. d) (b *. c) (b *. d)                   (* mnozymy pojedyncze przedzialy za pomoca osobnej funkcji *)
;;

let podzielic w1 w2 = (* dzielenie to mnozenie razy odwrotnosc *)
  match (w1, w2) with
  | (Pusty, _) | (_, Pusty) | (_, Z(0., 0.)) -> Pusty (* operacje na pustych przedzialach sa bez sensu, ponadto gdy dzielimy przez zero to zwracamy Pusty *)
  | _                                        -> razy w1 (odwrotnosc w2) (* bigbrain pomysl *)
;;
