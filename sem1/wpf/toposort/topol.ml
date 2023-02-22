(* --------------------------------------------------------------------------------- *)
(* -------------------SORTOWANIE TOPOLOGICZNE - STANISŁAW BITNER-------------------- *)
(* ------------------------CODE REVIEW------- - PIOTR TRZASKOWSKI------------------- *)
(* --------------------------------------------------------------------------------- *)

open PMap

exception Cykliczne;;

let topol l = 
  let visited = ref empty in
  let graph = List.fold_left 
  (fun acc (v, adj) -> if mem v acc = false then add v adj acc else add v (adj @ (find v acc)) acc) empty l in (* adjacency list *)
  let res = ref [] in

  let rec dfs v = (* depth first search *)
    visited := (add v true !visited);
    let adj = if mem v graph = true then find v graph else [] in
    List.iter (fun u -> if not (mem u !visited) then dfs u;) adj;
    res := v :: !res (* reverse postorder gives graph in topological order *)
  in

  List.iter (fun (v, _) -> if not (mem v !visited) then dfs v;) l;
  let prev = ref empty in
  List.iter (fun v -> 
    prev := add v true !prev;
    let adj = if mem v graph = true then find v graph else [] in
    List.iter (fun u -> if mem u !prev then raise Cykliczne) adj (* if edge goes to previous node then there is a cycle in graph *)
  ) !res;
  !res
;;
