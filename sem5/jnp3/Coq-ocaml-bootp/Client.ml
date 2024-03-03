open Definitions
open Protocol
open Utils
open Unix
open Encoding

(* let bootp_packet = Definitions.Bootp (one, double (succ two), double (succ two), zero, xid, passed_seconds, "0.0.0.0", "0.0.0.0", "0.0.0.0", "0.0.0.0", "AB:CD:EF:GH:IJ:KL", "", "", "") in
let packet_string = Encoding.bootp_packet_to_bytes_string bootp_packet in *)

let client_loop send_fd recv_fd = 
  let rec clinet_loop_aux pstate pinput = 
    let (pstate', poutput) = Protocol.client_proto_next pstate pinput in
    match poutput with
    | Definitions.Nothing -> clinet_loop_aux pstate' Definitions.IPacketRecv
    | ReciveTimeout packet -> clinet_loop_aux pstate' (Definitions.IPacketSend packet)
    | OPacket packet -> packet
    | SPacket packet -> clinet_loop_aux pstate' (Definitions.IPacketSend packet)
  in
  clinet_loop_aux NoOp (Definitions.IClient (send_fd, recv_fd))


let main () =
  Random.self_init ();
  let send_sock = create_send_socket 6700 in
  let recv_socket = create_rec_socket 6800 in
  let packet = client_loop send_sock recv_socket in
  match packet with
  | Definitions.Bootp (op, htype, hlen, hops, xid, secs, ciaddr, yiaddr, siaddr, giaddr, chaddr, sname, file, vend) ->
    print_string "op: \n";
    print_string (string_of_int (nat_to_uint op));
    print_string "\nhtype: \n";
    print_string (string_of_int (nat_to_uint htype));
    print_string "\nhlen: \n";
    print_string (string_of_int (nat_to_uint hlen));
    print_string "\nhops: \n";
    print_string (string_of_int (nat_to_uint hops));
    print_string "\nxid: \n";
    print_string (string_of_int (nat_to_uint xid));
    print_string "\nsecs: \n";
    print_string (string_of_int (nat_to_uint secs));
    print_string "\nciaddr: \n";
    print_string ciaddr;
    print_string "\nyiaddr: \n";
    print_string yiaddr;
    print_string "\nsiaddr: \n";
    print_string siaddr;
    print_string "\ngiaddr: \n";
    print_string giaddr;
    print_string "\nchaddr: \n";
    print_string chaddr;
    print_string "\nsname: \n";
    print_string sname;
    print_string "\nfile: \n";
    print_string file;
    print_string "\nvend: \n";
    print_string vend;
  close send_sock;
  close recv_socket;;

let () = main ()
