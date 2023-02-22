(* zlozonosc pamieciowa O(n) *)
(* zlozonosc czasowa    O(n) *)

(* 0 -> -1 spojny fragment o sumie 0 *)
(* sumy prefiksowe i spamietywanie *)

let zerojeden arr = 
  let n = Array.length arr in
  (* sumy prefiksowe beda dawac wartosci od -n do n *)
  (* nie mozemy miec ujemnych indeksow tablicy wiec dodajemy do wszystkiego +n *)
  (* to daje nam mozliwosci od 0 do 2n *)

  let first_appearance = Array.make (2 * n + 7) (n + 1) in
  first_appearance.(0 + n) <- (-1);
  let res = ref 0 in
  let pref_sum = ref 0 in

  for i = 0 to (n - 1) do (
    pref_sum := !pref_sum + (if arr.(i) = 1 then 1 else -1);
    first_appearance.(!pref_sum + n) <- min i (first_appearance.(!pref_sum + n));
    res := max !res (i - first_appearance.(!pref_sum + n));
  ) done;

  !res
;;
