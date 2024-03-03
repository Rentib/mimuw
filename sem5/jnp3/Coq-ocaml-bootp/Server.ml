open Protocol
open Definitions
open Utils
open Unix

let packet_size = 300              (* bootp packet size *)
let recv_port   = 6700             (* port to receive BOOTREQUEST *)
let send_port   = 6800             (* port to send BOOTREPLY *)
let hostname    = "server"         (* hostname of the server *)
let serverip    = "192.168.100.83" (* address of the server *)
let gateway     = "192.168.100.83" (* gateway address *)

let configs = [|
  ("192.168.100.83", "boot_loader");
  (* here we can add more configurations *)
|]

let find_config ip =
  let rec helper i = function
  | [] -> None
  | (ip', file) :: _ when ip = ip' -> Some (ip', file)
  | _ :: xs -> helper (i + 1) xs
  in
  helper 0 (Array.to_list configs)
;;

let main () =
  print_endline "Hello from Server";

  let rsock = create_rec_socket recv_port in
  let ssock = create_send_socket send_port in

  let buf = Bytes.create packet_size in
  while true do
    print_endline "Waiting for BOOTREQUEST";
    let (len, addr) = recvfrom rsock buf 0 packet_size [] in
    print_endline "Received BOOTREQUEST";
    (* parse received packet *)
    match bytes_to_bootp buf with
    | Definitions.Bootp (op, htype, hlen, hops, xid, secs, _, _, _, _, chaddr, _, _, options) ->
      print_string "op: ";      print_endline (string_of_int (nat_to_uint op));
      print_string "htype: ";   print_endline (string_of_int (nat_to_uint htype));
      print_string "hlen: ";    print_endline (string_of_int (nat_to_uint hlen));
      print_string "hops: ";    print_endline (string_of_int (nat_to_uint hops));
      print_string "xid: ";     print_endline (string_of_int (nat_to_uint xid));
      print_string "secs: ";    print_endline (string_of_int (nat_to_uint secs));
      print_string "chaddr: ";  print_endline (chaddr);
      print_string "options: "; print_endline (options);
      match addr with
      | ADDR_UNIX _ -> print_endline "Unix address"; ()
      | ADDR_INET (ip, port) ->
          match find_config (string_of_inet_addr ip) with
          | Some (ip', file) ->
              print_endline ("Found config for " ^ ip');
              let packet = Definitions.Bootp (
                Utils.uint_to_nat 2, (* BOOTREPLY[2] *)
                htype,               (* ethernet[1] *)
                hlen,                (* hardware address length: 6 bytes[6] *)
                hops,                (* hops[0] *)
                xid,                 (* transaction id[0] *)
                secs,                (* seconds elapsed[0] *)
                ip',                 (* client ip *)
                ip',                 (* client ip *)
                serverip,            (* server ip *)
                gateway,             (* gateway ip *)
                chaddr,              (* client hardware address *)
                hostname,            (* server name *)
                file,                (* boot file name *)
                options              (* options *)
              ) in
              print_endline ("Sending BOOTREPLY to " ^ ip');
              let packet = bootp_to_bytes packet in
              let _ = sendto ssock packet 0 (Bytes.length packet) [] (ADDR_INET (ip, send_port)) in
              print_endline "Message sent"
      | _ -> ()
  done
;;

let () = main ();;
