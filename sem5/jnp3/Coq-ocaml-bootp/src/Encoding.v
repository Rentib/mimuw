Require Import Definitions.
Require Import String Ascii.
Require Import List.
Require Import Int63.Sint63.
Require Import ZArith.
Import ListNotations.

Definition nat_to_char (n : nat) : ascii :=
  ascii_of_nat (n mod 256).

Definition char_to_nat (c : ascii) : nat :=
  nat_of_ascii c.

Definition char_to_Z (c : ascii) : Z :=
  Z.of_nat (char_to_nat c).

Definition char_to_string (c : ascii) : string :=
  String c EmptyString.

Definition string_to_char (s : string) : ascii :=
  match s with
  | String c _ => c
  | EmptyString => "0"%char
  end.

Fixpoint int_to_bytes_string_helper (bytes : nat) (n : nat) : string :=
  match bytes with
  | 0 => ""
  | S bytes' => int_to_bytes_string_helper bytes' (n / 256) ++ char_to_string (nat_to_char (n mod 256))
  end.

Definition int_to_bytes_string (number_of_bytes : nat) (n : nat) : string :=
  int_to_bytes_string_helper number_of_bytes n.

Fixpoint reverse_string_aux (s acc : string) : string :=
  match s with
  | EmptyString => acc
  | String c s' => reverse_string_aux s' (String c acc)
  end.

Definition reverse_string (s : string) : string :=
  reverse_string_aux s EmptyString.

Fixpoint bytes_string_to_int_helper (s : string) (acc : Z) (pow_of_256 : Z) : nat :=
  match s with
  | EmptyString => Z.to_nat acc
  | String c s' => bytes_string_to_int_helper s' (acc + (char_to_Z c) * pow_of_256) ( pow_of_256 )
  end.

Definition bytes_string_to_int (s : string) : nat :=
  let rev_string := reverse_string s in
  bytes_string_to_int_helper rev_string 0 1.

Fixpoint string_to_int (s : string) : nat :=
  let rev_s := reverse_string s in
    match s with
    | EmptyString => 0
    | String c s' => nat_of_ascii c + 256 * string_to_int s'
    end.

Fixpoint nat_to_string_helper (n : nat) (cnt : nat) : string :=
  match cnt with
  | 0 => "0"
  | S cnt' => 
    if n <? 10 then
      char_to_string (nat_to_char (n + 48))
    else
      let n' := n / 10 in
      let n'' := n mod 10 in
      nat_to_string_helper n' cnt' ++ char_to_string (nat_to_char (n'' + 48))
  end.

Definition nat_to_string (n : nat) : string :=
  nat_to_string_helper n n.

Fixpoint split (sep : ascii) (s : string) : list string :=
  match s with
  | EmptyString => [EmptyString]
  | String c s' =>
    if ascii_dec c sep then
      EmptyString :: split sep s'
    else
      match split sep s' with
      | [] => [] 
      | s'' :: ss => (String c s'') :: ss
      end
  end.

Fixpoint combine (sep : ascii) (ss : list string) : string :=
  match ss with
  | [] => EmptyString
  | s :: ss' => s ++ char_to_string sep ++ combine sep ss'
  end.

Definition ip_to_bytes_string (ip : string) : string :=
  let split_ip := split "." ip in
  match split_ip with
  | [a; b; c; d] =>
    let a' := int_to_bytes_string 1 (string_to_int a) in
    let b' := int_to_bytes_string 1 (string_to_int b) in
    let c' := int_to_bytes_string 1 (string_to_int c) in
    let d' := int_to_bytes_string 1 (string_to_int d) in
    a' ++ b' ++ c' ++ d'
  | _ => ""
  end.

Fixpoint get_substrings_helper (s : string) (lenghts : list nat) (acc : list string) : list string :=
  match lenghts with
  | [] => acc
  | l :: ls =>
    let s' := substring 0 l s in
    get_substrings_helper (substring l (String.length s - l) s) ls (s' :: acc)
  end.

Definition get_substrings (s : string) (lenghts : list nat) : list string :=
  get_substrings_helper s lenghts [].

Definition bytes_string_to_ip (s : string) : string :=
  let split_ip := get_substrings s [1; 1; 1; 1] in
  match split_ip with
  | [a; b; c; d] =>
    let a' := bytes_string_to_int a in
    let b' := bytes_string_to_int b in
    let c' := bytes_string_to_int c in
    let d' := bytes_string_to_int d in
    nat_to_string a' ++ "." ++ nat_to_string b' ++ "." ++ nat_to_string c' ++ "." ++ nat_to_string d'
  | _ => ""
  end.

Fixpoint hw_addr_to_bytes_string_aux (hw_addr : list string) (acc : string) : string :=
  match hw_addr with
  | [] => acc
  | h :: t =>
    let h' := int_to_bytes_string 2 (string_to_int h) in
    hw_addr_to_bytes_string_aux t (h' ++ acc)
  end.

Definition hw_addr_to_bytes_string (hw_addr : string) : string :=
  let split_hw_addr := split ":" hw_addr in
    hw_addr_to_bytes_string_aux split_hw_addr "".

Definition bytes_string_to_hw_addr (s : string) : string :=
  let split_hw_addr := get_substrings s [2;2;2;2;2;2] in
  match split_hw_addr with
  | a::b::c::d::e::f::[] =>
    let a' := bytes_string_to_int a in
    let b' := bytes_string_to_int b in
    let c' := bytes_string_to_int c in
    let d' := bytes_string_to_int d in
    let e' := bytes_string_to_int e in
    let f' := bytes_string_to_int f in
    nat_to_string a' ++ ":" ++ nat_to_string b' ++ ":" ++ nat_to_string c' ++ ":" ++ nat_to_string d' ++ ":" ++ nat_to_string e' ++ ":" ++ nat_to_string f'
  | _ => ""
  end.

Fixpoint fill_bytes_string_aux (to_add : nat) (s : string) : string :=
  match to_add with
  | 0 => s
  | S to_add' => fill_bytes_string_aux to_add' (String (nat_to_char 0) s)
  end.

Definition fill_bytes_string (target_len : nat) (s : string) : string :=
  let len := String.length s in
  if target_len <=? len then
    s
  else
    fill_bytes_string_aux (target_len - len) s.

Definition bootp_packet_to_bytes_string (bootp_packet : BootpPacket) : string :=
  match bootp_packet with
  | Bootp op htype hlen hops xid secs ciaddr yiaddr siaddr giaddr chaddr sname file vend =>
    let op'     := int_to_bytes_string 1 op in
    let htype'  := int_to_bytes_string 1 htype in
    let hlen'   := int_to_bytes_string 1 hlen in
    let hops'   := int_to_bytes_string 1 hops in
    let xid'    := int_to_bytes_string 4 xid in
    let secs'   := int_to_bytes_string 2 secs in
    let filler' := int_to_bytes_string 2 0 in
    let ciaddr' := ip_to_bytes_string ciaddr in
    let yiaddr' := ip_to_bytes_string yiaddr in
    let siaddr' := ip_to_bytes_string siaddr in
    let giaddr' := ip_to_bytes_string giaddr in
    let chaddr' := fill_bytes_string 16 (hw_addr_to_bytes_string chaddr) in
    let sname'  := fill_bytes_string 64 sname in
    let file'   := fill_bytes_string 128 file in
    let vend'   := fill_bytes_string 64 vend in
    op' ++ htype' ++ hlen' ++ hops' ++ xid' ++ secs' ++ filler' ++ ciaddr' ++ yiaddr' ++ siaddr' ++ giaddr' ++ chaddr' ++ sname' ++ file' ++ vend'
  end.

Definition bytes_string_to_bootp_packet (s : string) : BootpPacket :=
  let substrings := get_substrings (reverse_string s) [1; 1; 1; 1; 4; 2; 2; 4; 4; 4; 4; 16; 64; 128; 64] in
  match substrings with
  | [op; htype; hlen; hops; xid; secs; filler; ciaddr; yiaddr; siaddr; giaddr; chaddr; sname; file; vend] =>
    let op'     := bytes_string_to_int op in
    let htype'  := bytes_string_to_int htype in
    let hlen'   := bytes_string_to_int hlen in
    let hops'   := bytes_string_to_int hops in
    let xid'    := bytes_string_to_int xid in
    let secs'   := bytes_string_to_int secs in
    let ciaddr' := bytes_string_to_ip ciaddr in
    let yiaddr' := bytes_string_to_ip yiaddr in
    let siaddr' := bytes_string_to_ip siaddr in
    let giaddr' := bytes_string_to_ip giaddr in
    let chaddr' := bytes_string_to_hw_addr chaddr in
    let sname'  := sname in
    let file'   := file in
    let vend'   := vend in
    Bootp op' htype' hlen' hops' xid' secs' ciaddr' yiaddr' siaddr' giaddr' chaddr' sname' file' vend'
  | _ => Bootp 0 0 0 0 0 0 "" "" "" "" "" "" "" ""
  end.
