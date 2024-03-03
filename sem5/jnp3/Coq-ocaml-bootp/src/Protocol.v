Require Import Encoding.
Require Import Definitions.
Require Import String.
Require Import Helper.
Include Helper.


Definition client_proto_next (st : PState) (inp : PInput) : (PState * POutput) :=
    match st with
    | NoOp =>
      match inp with
      | IClient send_fd recv_fd => Helper.init_client send_fd recv_fd
      | IPacketSend _ => (SError "Packet is invalid at initialisation", Nothing)
      | IPacketRecv => (SError "Packet is invalid at initialisation", Nothing)
      | IError reason => (SError reason, Nothing)
      end
    | FComm (send_fd, recv_fd) =>
      match inp with
      | IPacketSend packet => Helper.send_packet send_fd recv_fd packet
      | IPacketRecv => Helper.recv_packet send_fd recv_fd
      | IError reason => (SError reason, Nothing)
      | IClient _ _ => (SError "Client is invalid at connection", Nothing)
      end
    | Final => (Final, Nothing)
    | SError reason => (SError reason, Nothing)
    end.
      


