Require Import Definitions.

Variable init_client : Definitions.file_descr ->
  Definitions.file_descr ->
  Definitions.PState * Definitions.POutput.

Variable send_packet : Definitions.file_descr ->
  Definitions.file_descr ->
  Definitions.BootpPacket ->
  Definitions.PState * Definitions.POutput.

Variable recv_packet : Definitions.file_descr ->
  Definitions.file_descr ->
  Definitions.PState * Definitions.POutput.