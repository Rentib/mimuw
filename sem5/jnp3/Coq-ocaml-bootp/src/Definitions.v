Require Import String.

Variable file_descr : Set.

Definition send_recv_sockets : Set := file_descr * file_descr.

Inductive BootpPacket : Set :=
| Bootp 
    (op : nat )
    (htype : nat )
    (hlen : nat )
    (hops : nat )
    (xid : nat )
    (secs : nat )
    (ciaddr : string )
    (yiaddr : string )
    (siaddr : string )
    (giaddr : string )
    (chaddr : string )
    (sname : string )
    (file : string )
    (vend : string ).

Inductive PState : Set := 
| NoOp
| FComm (sockets : send_recv_sockets)
| Final
| SError (reason:string).

Inductive PInput : Set :=
| IClient (send_socket : file_descr) (recv_socket : file_descr)
| IPacketSend (packet : BootpPacket)
| IError (reason:string)
| IPacketRecv.

Inductive POutput : Set :=
| SPacket (packet : BootpPacket)
| OPacket (packet : BootpPacket)
| ReciveTimeout (packet : BootpPacket)
| Nothing.
