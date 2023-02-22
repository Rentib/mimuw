(* Modyfikacja drzew  -   Stanisław Bitner  *)
(*   Code reviewer    - Wojciech Rzepliński *)

(* ----------------------------------------------------- *)
(* ------------------- helper type --------------------- *)
(* ----------------------------------------------------- *)

type interval = int * int;;
type height = int;;
type size = int;;

(* ----------------------------------------------------- *)
(* --------------------- type -------------------------- *)
(* ----------------------------------------------------- *)

type t = Empty | Node of t * interval * t * height * size;;
(* left child, interval [x, y] where x <= y, right child, height of tree, number of elements in subtree *)

(* ----------------------------------------------------- *)
(* ----------------- helper functions ------------------ *)
(* ----------------------------------------------------- *)

let height = function
  | Node (_, _, _, h, _) -> h
  | _                    -> 0
;;

let size = function
  | Node (_, _, _, _, s) -> s
  | _                    -> 0
;;

(* Creates new node with son l, value k and right son r.
   All elements in l must be lower than k and all elements in r higher.set l and r must be balanced *)
let make l (x, y) r = 
  Node(
    l, (x, y), r, 
    1 + (max (height l) (height r)), 
    size l + size r + (y - x + 1) (* in interval [x, y] there is exactly (y - x + 1) elements *)
  )
;;

(* returns balanced tree *)
let balance l k r = 
  let handle_left_child child_l child_k child_r =
    if (height child_l) >= (height child_r) then
      make child_l child_k (make child_r k r)
    else
      match child_r with
      | Node(lrl, lrk, lrr, _, _) -> make (make child_l child_k lrl) lrk (make lrr k r)
      | _                         -> assert false
  in
  let handle_right_child child_l child_k child_r = 
    if (height child_r) >= (height child_l) then
      make (make l k child_l) child_k child_r
    else
      match child_l with
      | Node(rll, rlk, rlr, _, _) -> make (make l k rll) rlk (make rlr child_k child_r)
      | _                         -> assert false
  in
  if (height l) > (height r) + 2 then
    match l with
    | Node(ll, lk, lr, _, _) -> handle_left_child ll lk lr
    | _                      -> assert false
  else if (height r) > (height l) + 2 then
    match r with
    | Node(rl, rk, rr, _, _) -> handle_right_child rl rk rr
    | _                      -> assert false
  else
    make l k r
;;

let rec min_element = function
  | Node(Empty, k, _, _, _) -> k
  | Node(l,     _, _, _, _) -> min_element l
  | _                       -> raise Not_found
;;

let rec remove_min_element = function
  | Node(Empty, _, r, _, _) -> r
  | Node(l,     k, r, _, _) -> balance (remove_min_element l) k r
  | _                       -> invalid_arg "ISet.remove_min_elt"
;;

let merge t1 t2 = 
  match (t1, t2) with
  | (Empty, _) -> t2
  | (_, Empty) -> t1
  | _          ->
      let k = min_element t2 in
      balance t1 k (remove_min_element t2)
;;

(*  Adds one interval to the tree - interval must be disjoint with all intervals in tree *)
let rec add_one (x, y) = function
  | Node(l, (a, b), r, _, _) ->
      if x > b then
        let new_r = add_one (x, y) r in 
        balance l (a, b) new_r
      else
        let new_l = add_one (x, y) l in 
        balance new_l (a, b) r
  | _ -> make Empty (x, y) Empty
;;

(* returns tree made of tree l, interval x and tree r
   all elements of l must be smaller than x and all elements of r must be greater than x *)
let rec join l x r = 
  match (l, r) with
  | (Empty, _) -> add_one x r
  | (_, Empty) -> add_one x l
  | (Node(ll, lk, lr, lh, _), Node(rl, rk, rr, rh, _)) ->
      if lh > rh + 2 then 
        balance ll lk (join lr x r) 
      else if rh > lh + 2 then 
        balance (join l x rl) rk rr 
      else
        make l x r
;;

(* first element smaller / greater than x in tree *)
let rec get x f inc = function
  | Node(l, (a, b), r, _, _) ->
      if x < a then
        get x f inc l
      else if x > b then
        get x f inc r
      else
        f (a, b) (* a if looking for smaller, b if looking for greater *)
  | _ -> x + inc (* 1 if looking for smaller, -1 if looking for greater *)
;;

(* ----------------------------------------------------- *)
(* ---------------- main functions --------------------- *)
(* ----------------------------------------------------- *)

let empty = Empty;;

let is_empty t = (t = Empty);;

(* split makes other functions easier *)
let split x t = 
  let rec helper x = function
    | Node(l, (a, b), r, _, _) -> 
        if x < a then
          let (ll, pres, rl) = helper x l in (ll, pres, join rl (a, b) r)
        else if x > b then
          let (lr, pres, rr) = helper x r in (join l (a, b) lr, pres, rr)
        else
          let ll = if x > a then join l (a, x - 1) Empty else l
          and rr = if x < b then join Empty (x + 1, b) r else r
          in (ll, true, rr)
    | _ -> (Empty, false, Empty)
  in 
  helper x t
;;

let remove (x, y) t = 
  let l = (fun (a, b, c) -> a) (split x t)
  and r = (fun (a, b, c) -> c) (split y t) in 
  merge l r
;;

let add (x, y) t = 
  let merged_interval = 
    if x = min_int && y = max_int then 
      (x, y)
    else if x = min_int then 
      (x, get (y + 1) snd (-1) t)
    else if y = max_int then 
      (get (x - 1) fst (+1) t, y)
    else 
      (get (x - 1) fst (+1) t, get (y + 1) snd (-1) t)
  in
  add_one merged_interval (remove merged_interval t)
;;

let rec mem x = function
  | Node(l, (a, b), r, _, _) ->
      if x >= a && x <= b then true
      else if x < a then mem x l
      else mem x r
  | _ -> false
;;

let iter f t = 
  let rec loop = function
    | Node(l, k, r, _, _) -> loop l; f k; loop r
    | _ -> ()
  in loop t
;;

let fold f t acc =
  let rec loop acc = function
    | Node (l, k, r, _, _) -> loop (f k (loop acc l)) r
    | _ -> acc
  in loop acc t
;;

let elements t = 
  let rec helper a t = 
    match t with
    | Node(l, i, r, _, _) -> helper (i :: (helper a r)) l
    | _ -> a
  in helper [] t
;;

let below x t = 
  let rec helper acc = function
    | Node(l, (a, b), r, _, _) ->
      if b = max_int && a = min_int && x > 0 then 
        max_int else
      if x > b then 
        helper (acc + size l + b - a + 1) r
      else if x < a then 
        helper acc l
      else 
        helper (acc + x - a + 1) l
    | _ -> acc
  in 
  let fix = helper 0 t in
  if fix < 0 then 
    max_int 
  else 
    fix
;;

(*
 * PSet - Polymorphic sets
 * Copyright (C) 1996-2003 Xavier Leroy, Nicolas Cannasse, Markus Mottl
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version,
 * with the special exception on linking described in file LICENSE.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *)
