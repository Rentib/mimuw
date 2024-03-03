open Definitions

val init_client :
    Definitions.file_descr ->
    Definitions.file_descr ->
    Definitions.coq_PState * Definitions.coq_POutput
    
val send_packet :
    Definitions.file_descr ->
    Definitions.file_descr ->
    Definitions.coq_BootpPacket ->
    Definitions.coq_PState * Definitions.coq_POutput

val recv_packet :
    Definitions.file_descr ->
    Definitions.file_descr ->
    Definitions.coq_PState * Definitions.coq_POutput
