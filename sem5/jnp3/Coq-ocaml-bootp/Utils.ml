open Definitions
open Encoding
open Unix
open Nat

let send_socket_port = 67
let receive_socket_port = 68

let broadcast_address = Unix.inet_addr_of_string "255.255.255.255"

let create_rec_socket port = 
  let sockaddr = ADDR_INET (inet_addr_any, port) in
  let socket = socket PF_INET SOCK_DGRAM 0 in
  setsockopt socket SO_BROADCAST true; 
  bind socket sockaddr;
  socket

let create_send_socket port = 
  let sockaddr = ADDR_INET (broadcast_address, port) in
  let socket = socket PF_INET SOCK_DGRAM 0 in
  setsockopt socket SO_BROADCAST true; (* Enable broadcasting on the socket *)
  connect socket sockaddr;
  socket

let uint_to_nat n = 
  let rec uint_to_nat_rec n acc powers_of_two= 
    if n = 0 then acc
    else if n mod 2 = 1 then uint_to_nat_rec (n / 2) (Nat.add acc powers_of_two) (Nat.double powers_of_two)
    else uint_to_nat_rec (n / 2) acc (Nat.double powers_of_two)
  in
  uint_to_nat_rec n Nat.zero Nat.one

let nat_to_uint n = 
  let rec nat_to_uint_rec n acc powers_of_two = 
    if n = Nat.zero then acc
    else if Nat.even n then nat_to_uint_rec (Nat.div2 n) acc (powers_of_two * 2)
    else nat_to_uint_rec (Nat.div2 n) (acc + powers_of_two) (powers_of_two * 2)
  in
  nat_to_uint_rec n 0 1

let bytes_to_nat bytes = 
  let rec helper bytes acc =
    match bytes with
    | b when Bytes.length b = 0 -> acc
    | b -> helper (Bytes.sub b 1 (Bytes.length b - 1)) (acc * 256 + (int_of_char (Bytes.get b 0)))
  in
  let n = helper bytes 0 in
  uint_to_nat n

let bytes_to_bootp bytes = 
  let op = bytes_to_nat (Bytes.sub bytes 0 1) in
  let htype = bytes_to_nat (Bytes.sub bytes 1 1) in
  let hlen = bytes_to_nat (Bytes.sub bytes 2 1) in
  let hops = bytes_to_nat (Bytes.sub bytes 3 1) in
  let xid = bytes_to_nat (Bytes.sub bytes 4 4) in
  let secs = bytes_to_nat (Bytes.sub bytes 8 2) in
  let ciaddr = Bytes.to_string (Bytes.sub bytes 12 4) in
  let yiaddr = Bytes.to_string (Bytes.sub bytes 16 4) in
  let siaddr = Bytes.to_string (Bytes.sub bytes 20 4) in
  let giaddr = Bytes.to_string (Bytes.sub bytes 24 4) in
  let chaddr = Bytes.to_string (Bytes.sub bytes 28 16) in
  let sname  = Bytes.to_string (Bytes.sub bytes 44 64) in
  let file   = Bytes.to_string (Bytes.sub bytes 108 128) in
  let vend   = Bytes.to_string (Bytes.sub bytes 236 64) in
  Definitions.Bootp (op, htype, hlen, hops, xid, secs, ciaddr, yiaddr, siaddr, giaddr, chaddr, sname, file, vend)

let nat_to_bytes n size =
  let rec helper n size bytes =
    if size = 0 then bytes
    else helper (n / 2) (size - 1) (Bytes.concat Bytes.empty [Bytes.make 1 (char_of_int (n mod 256)); bytes])
  in
  helper (nat_to_uint n) size Bytes.empty

let bootp_to_bytes bootp =
  match bootp with
  | Definitions.Bootp (op, htype, hlen, hops, xid, secs, ciaddr, yiaddr, siaddr, giaddr, chaddr, sname, file, vend) ->
      let op_bytes     = nat_to_bytes op 1 in
      let htype_bytes  = nat_to_bytes htype 1 in
      let hlen_bytes   = nat_to_bytes hlen 1 in
      let hops_bytes   = nat_to_bytes hops 1 in
      let xid_bytes    = nat_to_bytes xid 4 in
      let secs_bytes   = nat_to_bytes secs 2 in
      let ciaddr_bytes = Bytes.of_string ciaddr in
      let yiaddr_bytes = Bytes.of_string yiaddr in
      let siaddr_bytes = Bytes.of_string siaddr in
      let giaddr_bytes = Bytes.of_string giaddr in
      let chaddr_bytes = Bytes.of_string chaddr in
      let sname_bytes  = Bytes.of_string sname in
      let file_bytes   = Bytes.of_string file in
      let vend_bytes   = Bytes.of_string vend in
      let bytes = Bytes.concat Bytes.empty [op_bytes; htype_bytes; hlen_bytes; hops_bytes; xid_bytes; secs_bytes; ciaddr_bytes; yiaddr_bytes; siaddr_bytes; giaddr_bytes; chaddr_bytes; sname_bytes; file_bytes; vend_bytes] in
      bytes
  | _ -> Bytes.make 300 '\000'
