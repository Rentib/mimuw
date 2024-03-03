Require Export Definitions.
Require Export Protocol.
Require Export Encoding.

Require Extraction.
Extraction Blacklist Uint63 List Ascii String.

Extract Constant Definitions.file_descr => "Unix.file_descr".

Require Protocol.
Require Encoding.

Require ExtrOcamlBasic.
Require ExtrOcamlNativeString.
Require ExtrOCamlInt63.
Require ExtrOcamlZInt.

Extraction Library Protocol.
Recursive Extraction Library Encoding.