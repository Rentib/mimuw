build_coq:
	- $(MAKE) -f CoqMakefile

COQGENERATEDML :=  Ascii0.ml BinNat.ml BinNums.ml BinPos.ml \
	BinPosDef.ml Bool.ml Byte.ml Datatypes.ml DecidableClass.ml \
	Decimal.ml Definitions.ml Hexadecimal.ml List0.ml Logic.ml \
	Nat.ml Number.ml Protocol.ml Specif.ml String.ml Wf.ml \
	Ascii.ml CarryType.ml List.ml PeanoNat.ml Encoding.ml \
	String0.ml Helper0.ml BinInt.ml

COQGENERATED := $(COQGENERATEDML) $(COQGENERATEDML:.ml=.mli)

build: $(COQGENERATED)
	- ocamlbuild -use-ocamlfind $(COQGENERATED)

clean: 
	- $(MAKE) -f CoqMakefile clean
	- rm -f $(COQGENERATED)
	- find . -name "*.cmi" -delete
	- find . -name "*.cmo" -delete
	- rm Client Server

Client: build_coq Helper.cmo Protocol.cmo Encoding.cmo Utils.cmo Client.ml BinNat.cmo
	ocamlc unix.cma Decimal.cmo Hexadecimal.cmo Datatypes.cmo   Nat.cmo List0.cmo Bool.cmo PeanoNat.cmo BinPos.cmo BinNat.cmo Ascii0.cmo String0.cmo BinInt.cmo Encoding.cmo Utils.cmo Helper.cmo Protocol.cmo  Client.ml -o Client  

Server: build_coq Helper.cmo Protocol.cmo Encoding.cmo Utils.cmo Server.ml BinNat.cmo
	ocamlc unix.cma Decimal.cmo Hexadecimal.cmo Datatypes.cmo   Nat.cmo List0.cmo Bool.cmo PeanoNat.cmo BinPos.cmo BinNat.cmo Ascii0.cmo String0.cmo BinInt.cmo Encoding.cmo Utils.cmo Helper.cmo Protocol.cmo  Server.ml -o Server  

Utils.cmo: build_coq Nat.cmo Protocol.cmo Encoding.cmo Utils.ml
	ocamlc unix.cma -c Utils.ml

Ascii0.cmi: build_coq Ascii0.mli BinNat.cmo Bool.cmo Datatypes.cmo Byte.cmo
	ocamlc -c Ascii0.mli

Ascii0.cmo: build_coq Ascii0.ml Ascii0.cmi
	ocamlc -c Ascii0.ml

Ascii.cmi: build_coq BinNat.cmo Bool.cmo Datatypes.cmo
	ocamlc -c Ascii.mli

Ascii.cmo: build_coq Ascii.ml Ascii.cmi
	ocamlc -c Ascii.ml

CarryType.cmi: build_coq CarryType.mli
	ocamlc -c CarryType.mli

CarryType.cmo: build_coq CarryType.ml CarryType.cmi
	ocamlc -c CarryType.ml

List.cmi: build_coq List.mli Datatypes.cmo Nat.cmo Specif.cmo
	ocamlc -c List.mli

List.cmo: build_coq List.ml List.cmi
	ocamlc -c List.ml

List0.cmi: build_coq List0.mli Datatypes.cmo Nat.cmo Specif.cmo
	ocamlc -c List0.mli

List0.cmo: build_coq List0.ml List0.cmi
	ocamlc -c List0.ml

PeanoNat.cmi: build_coq PeanoNat.mli Bool.cmo Datatypes.cmo DecidableClass.cmo Decimal.cmo Hexadecimal.cmo Nat.cmo Number.cmo
	ocamlc -c PeanoNat.mli

PeanoNat.cmo: build_coq PeanoNat.ml PeanoNat.cmi
	ocamlc -c PeanoNat.ml

BinNat.cmi: build_coq BinNat.mli BinPos.cmo Bool.cmo Datatypes.cmo Decimal.cmo Hexadecimal.cmo Number.cmo
	ocamlc -c BinNat.mli

BinNat.cmo: build_coq BinNat.ml BinNat.cmi
	ocamlc -c BinNat.ml

BinNums.cmi: build_coq BinNums.mli 
	ocamlc -c BinNums.mli

BinNums.cmo: build_coq BinNums.ml BinNums.cmi
	ocamlc -c BinNums.ml

BinPos.cmi: build_coq BinPos.mli BinPosDef.cmo Bool.cmo Datatypes.cmo Decimal.cmo Hexadecimal.cmo Number.cmo Nat.cmo
	ocamlc -c BinPos.mli

BinPos.cmo: build_coq BinPos.ml BinPos.cmi
	ocamlc -c BinPos.ml

BinPosDef.cmi: build_coq BinPosDef.mli Datatypes.cmo Decimal.cmo Hexadecimal.cmo Nat.cmo Number.cmo
	ocamlc -c BinPosDef.mli

BinPosDef.cmo: build_coq BinPosDef.ml BinPosDef.cmi
	ocamlc -c BinPosDef.ml

Bool.cmi: build_coq Bool.mli Datatypes.cmo DecidableClass.cmo
	ocamlc -c Bool.mli

Bool.cmo: build_coq Bool.ml Bool.cmi
	ocamlc -c Bool.ml

Byte.cmi: build_coq Byte.mli 
	ocamlc -c Byte.mli

Byte.cmo: build_coq Byte.ml Byte.cmi
	ocamlc -c Byte.ml

Datatypes.cmi: build_coq Datatypes.mli
	ocamlc -c Datatypes.mli

Datatypes.cmo: build_coq Datatypes.ml Datatypes.cmi	
	ocamlc -c Datatypes.ml

DecidableClass.cmi: build_coq DecidableClass.mli Datatypes.cmo
	ocamlc -c DecidableClass.mli

DecidableClass.cmo: build_coq DecidableClass.ml DecidableClass.cmi
	ocamlc -c DecidableClass.ml

Decimal.cmi: build_coq Decimal.mli Datatypes.cmo
	ocamlc -c Decimal.mli

Decimal.cmo: build_coq Decimal.ml Decimal.cmi
	ocamlc -c Decimal.ml

Definitions.cmi: build_coq Definitions.mli Datatypes.cmo
	ocamlc -c Definitions.mli

Definitions.cmo: build_coq Definitions.ml Definitions.cmi
	ocamlc -c Definitions.ml

Hexadecimal.cmi: build_coq Hexadecimal.mli Datatypes.cmo Decimal.cmo
	ocamlc -c Hexadecimal.mli

Hexadecimal.cmo: build_coq Hexadecimal.ml Hexadecimal.cmi
	ocamlc -c Hexadecimal.ml

List0.cmi: build_coq List0.mli Datatypes.cmo Nat.cmo Specif.cmo
	ocamlc -c List0.mli

List0.cmo: build_coq List0.ml List0.cmi
	ocamlc -c List0.ml

Logic.cmi: build_coq Logic.mli
	ocamlc -c Logic.mli

Logic.cmo: build_coq Logic.ml Logic.cmi	
	ocamlc -c Logic.ml

Nat.cmi: build_coq Nat.mli Datatypes.cmo Decimal.cmo Hexadecimal.cmo Number.cmo
	ocamlc -c Nat.mli

Nat.cmo: build_coq Nat.ml Nat.cmi
	ocamlc -c Nat.ml

Number.cmi: build_coq Number.mli Decimal.cmo Hexadecimal.cmo
	ocamlc -c Number.mli

Number.cmo: build_coq Number.ml Number.cmi
	ocamlc -c Number.ml

Protocol.cmi: build_coq Protocol.mli Definitions.cmo Ascii0.cmo Helper.cmi
	ocamlc -c Protocol.mli

Protocol.cmo: build_coq Protocol.ml Protocol.cmi  Helper.cmo
	ocamlc -c Protocol.ml

Specif.cmi: build_coq Specif.mli Datatypes.cmo
	ocamlc -c Specif.mli

Specif.cmo: build_coq Specif.ml Specif.cmi
	ocamlc -c Specif.ml

String.cmi: build_coq String.mli Ascii0.cmo BinNat.cmo Bool.cmo Datatypes.cmo List0.cmo
	ocamlc -c String.mli

String.cmo: build_coq String.ml String.cmi
	ocamlc -c String.ml

String0.cmi: build_coq String0.mli Ascii0.cmo BinNat.cmo Bool.cmo Datatypes.cmo List0.cmo
	ocamlc -c String0.mli

String0.cmo: build_coq String0.ml String0.cmi
	ocamlc -c String0.ml

Wf.cmi: build_coq Wf.mli
	ocamlc -c Wf.mli

Wf.cmo: build_coq Wf.ml Wf.cmi
	ocamlc -c Wf.ml

Encoding.cmi: build_coq Encoding.mli Ascii0.cmo Datatypes.cmo Definitions.cmo Nat.cmo List0.cmo PeanoNat.cmo String0.cmo BinInt.cmo
	ocamlc -c Encoding.mli

Encoding.cmo: build_coq Encoding.ml Encoding.cmi
	ocamlc -c Encoding.ml

Helper.cmi: build_coq Helper.mli Definitions.cmo Encoding.cmo Utils.cmo
	ocamlc -c Helper.mli

Helper.cmo : build_coq Helper.ml Definitions.cmo Encoding.cmo Utils.cmo
	ocamlc unix.cma -c Helper.ml

BinInt.cmi: build_coq BinInt.mli Datatypes.cmo Bool.cmo
	ocamlc -c BinInt.mli

BinInt.cmo: build_coq BinInt.ml BinInt.cmi
	ocamlc -c BinInt.ml
