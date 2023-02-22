(* --------------------------------------------------------------------------------- *)
(* -----------------------PRZELEWANKA - STANISŁAW BITNER---------------------------- *)
(* -----------------------CODE REVIEW - MAREK ZBYSIŃSKI----------------------------- *)
(* --------------------------------------------------------------------------------- *)

let rec gcd a b = if b = 0 then a else gcd b (a mod b);;
let check arr = 
	let vol = List.exists (fun (l, r) ->  l = r || r = 0) (Array.to_list arr)
	and gcd_l = Array.fold_left (fun acc (l, _) -> gcd acc l) (arr.(0) |> fst) arr
	and gcd_r = Array.fold_left (fun acc (_, r) -> gcd acc r) (arr.(0) |> snd) arr in
	vol && (gcd_r mod gcd_l = 0)
;;

let przelewanka (arr : (int * int) array) = 
  let arr = Array.of_list (List.filter (fun (l, _) -> l > 0) (Array.to_list arr)) in
  if Array.length arr = 0 then 0 else
  if not (check arr) then -1 else

  let n = Array.length arr
  and a = Array.map fst arr (* maximum amount array *)
  and b = Array.map snd arr (* target  amount array *)
  and q = Queue.create ()
  and visited = Hashtbl.create 424242 in

  (* we are cool, so we store hashes instead of arrays (colisions are almost impossible, much better performance) *)
  (* hash mod 1e9 + 7, sum of squares, sum of cubes, xor *)
  let make_hasz = Array.fold_left (fun (h, qu, cu, xs) x -> ((h * 997 + x) mod 1000000007, qu + x * x, cu + x * x * x, (lxor) xs x)) (0, 0, 0, 0) in 
  let target = make_hasz b
  and res = ref (-1) in

  Queue.push (Array.make n 0, 0) q;

  (* bfs over dynamically built graph of states *)
  while (!res < 0 && (not (Queue.is_empty q))) do(
    let (v, dist) = Queue.pop q in
    let vh = make_hasz v in
    if (vh = target) then(
      res := dist;
    )else if (Hashtbl.mem visited vh = false) then(
      Hashtbl.add visited vh true;
      for i = 0 to n - 1 do(
        for j = 0 to n - 1 do(
          (* pour from glass i to glass j *)
          if (i <> j && Array.get v i > 0) then(
            if (Array.get v i <= Array.get a j - Array.get v j) then(
              let u = Array.copy v in
              Array.set u i 0;
              Array.set u j (Array.get v j + Array.get v i);
              Queue.push (u, dist + 1) q;
            )
            else if (Array.get a j > Array.get v j) then(
              let u = Array.copy v in
              Array.set u i (Array.get v i - (Array.get a j - Array.get v j));
              Array.set u j (Array.get a j);
              Queue.push (u, dist + 1) q;
            );
          );
        )done;
        let u1 = Array.copy v in
        let u2 = Array.copy v in
        (* fill glass *)
        Array.set u1 i (Array.get a i);
        Queue.push (u1, dist + 1) q;
        (* empty glass *)
        Array.set u2 i 0;
        Queue.push (u2, dist + 1) q;
      )done;
    )
  )done;
  !res
;;

