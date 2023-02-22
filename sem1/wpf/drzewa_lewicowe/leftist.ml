(* Drzewa lewicowe - Stanislaw Bitner *)
(* Sprawdzajacy    -   Jakub Klimek   *)

(* type *)
type 'a queue = Null | Node of 'a * 'a queue * 'a queue * int;;

(* check right height of heap *)
let get_height = function
  | Node(_, _, _, h) -> h
  | _                -> 0

(* heap join as written in task description *)
let rec join d1 d2 = 
  match (d1, d2) with
  | (Null, Null) -> Null
  | (Null, _) -> d2
  | (_, Null) -> d1
  | (Node(x1, _, _, _), Node(x2, _, _, _)) when x1 > x2 -> join d2 d1 (* in order to prevent redundant code we want d1 to have smaller root value than d2 *)
  | (Node(x1, l1, r1, _), Node(x2, l2, r2, _)) ->
      let d3 = join r1 d2 in
      let h3 = get_height d3 in
      match l1 with
      | Null -> Node(x1, d3, Null, 1)
      | Node(_, _, _, hl) -> 
          match (hl <= h3) with
          | true  -> Node(x1, d3, l1, hl + 1)
          | false -> Node(x1, l1, d3, h3 + 1)

(* empty queue constructor *)
let empty = Null;;

(* function for adding new elements "e" to queue q - as written in task description *)
let add e q = join (Node(e, Null, Null, 1)) q

(* exception for when queue is empty and user asks for delete_min *)
exception Empty

(* function for deleting highest priority element and returning it - as written in task descrition *)
let delete_min = function
  | Null             -> raise Empty
  | Node(x, l, r, _) -> (x, join l r)

(* function to chech wheter queue is empty *)
let is_empty = function
  | Null -> true
  | _    -> false

