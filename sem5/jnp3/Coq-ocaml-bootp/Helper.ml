open Definitions
open Utils
open Unix

let init_client send_connection rec_connection = 
  let packet = Definitions.Bootp (Utils.uint_to_nat 1, Utils.uint_to_nat 6, Utils.uint_to_nat 6, Utils.uint_to_nat 0, Utils.uint_to_nat 0, Utils.uint_to_nat 0, "0.0.0.0", "0.0.0.0", "0.0.0.0", "0.0.0.0", "AB:CD:EF:12:34:56", "", "", "") in
  (Definitions.FComm (send_connection, rec_connection), Definitions.SPacket packet)


let send_packet send_connection recv_connection packet = 
  let packet_string = Encoding.bootp_packet_to_bytes_string packet in
  print_string "Sending packet\n";
  let _ = send send_connection (Bytes.of_string packet_string) 0 (String.length packet_string) [] in
  (Definitions.FComm (send_connection, recv_connection), Definitions.Nothing)
  
let recv_packet send_connection recv_connection = 
  let packet_string = Bytes.create 300 in
  print_string "Waiting for packet\n";
  let (_, _) = recvfrom recv_connection packet_string 0 300 [] in
  let packet = Utils.bytes_to_bootp packet_string in
  (Definitions.FComm (send_connection, recv_connection), Definitions.OPacket packet)
