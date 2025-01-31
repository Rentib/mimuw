module Asm.DataTypes where

import Data.Bifunctor
import Data.List (sortOn)
import qualified Data.Map as Map
import Quad.DataTypes

data AsmRegister where
    RegAL :: AsmRegister
    RegBL :: AsmRegister
    RegBPL :: AsmRegister
    RegCL :: AsmRegister
    RegDIL :: AsmRegister
    RegDL :: AsmRegister
    RegEAX :: AsmRegister
    RegEBP :: AsmRegister
    RegECX :: AsmRegister
    RegEDI :: AsmRegister
    RegEDX :: AsmRegister
    RegESI :: AsmRegister
    RegESP :: AsmRegister
    RegR10 :: AsmRegister
    RegR10B :: AsmRegister
    RegR11 :: AsmRegister
    RegR11B :: AsmRegister
    RegR11D :: AsmRegister
    RegR12 :: AsmRegister
    RegR12B :: AsmRegister
    RegR13 :: AsmRegister
    RegR13B :: AsmRegister
    RegR14 :: AsmRegister
    RegR14B :: AsmRegister
    RegR15 :: AsmRegister
    RegR15B :: AsmRegister
    RegR8 :: AsmRegister
    RegR8B :: AsmRegister
    RegR8D :: AsmRegister
    RegR9 :: AsmRegister
    RegR9B :: AsmRegister
    RegR9D :: AsmRegister
    RegRAX :: AsmRegister
    RegRBP :: AsmRegister
    RegRBX :: AsmRegister
    RegRCX :: AsmRegister
    RegRDI :: AsmRegister
    RegRDX :: AsmRegister
    RegRIP :: AsmRegister
    RegRSI :: AsmRegister
    RegRSP :: AsmRegister
    RegSIL :: AsmRegister
    RegSPL :: AsmRegister
    deriving (Eq)

instance Show AsmRegister where
    show RegAL = "al"
    show RegBL = "bl"
    show RegBPL = "bpl"
    show RegCL = "cl"
    show RegDIL = "dil"
    show RegDL = "dl"
    show RegEAX = "eax"
    show RegEBP = "ebp"
    show RegECX = "ecx"
    show RegEDI = "edi"
    show RegEDX = "edx"
    show RegESI = "esi"
    show RegESP = "esp"
    show RegR10 = "r10"
    show RegR10B = "r10b"
    show RegR11 = "r11"
    show RegR11B = "r11b"
    show RegR11D = "r11d"
    show RegR12 = "r12"
    show RegR12B = "r12b"
    show RegR13 = "r13"
    show RegR13B = "r13b"
    show RegR14 = "r14"
    show RegR14B = "r14b"
    show RegR15 = "r15"
    show RegR15B = "r15b"
    show RegR8 = "r8"
    show RegR8B = "r8b"
    show RegR8D = "r8d"
    show RegR9 = "r9"
    show RegR9B = "r9b"
    show RegR9D = "r9d"
    show RegRAX = "rax"
    show RegRBP = "rbp"
    show RegRBX = "rbx"
    show RegRCX = "rcx"
    show RegRDI = "rdi"
    show RegRDX = "rdx"
    show RegRIP = "rip"
    show RegRSI = "rsi"
    show RegRSP = "rsp"
    show RegSIL = "sil"
    show RegSPL = "spl"

data Asm where
    AsmAdd :: String -> String -> Asm
    AsmAnd :: String -> String -> Asm
    AsmCall :: String -> Asm
    AsmCdq :: Asm
    AsmCmp :: String -> String -> Asm
    AsmDec :: String -> Asm
    AsmDq :: String -> Asm
    AsmEpilog :: Asm
    AsmExtern :: Asm
    AsmFuncSpec :: String -> Asm
    AsmGlobl :: Asm
    AsmIdiv :: String -> Asm
    AsmMul :: String -> Asm
    AsmImul :: String -> String -> Asm
    AsmInc :: String -> Asm
    AsmJE :: String -> Asm
    AsmJGE :: String -> Asm
    AsmJG :: String -> Asm
    AsmJLE :: String -> Asm
    AsmJL :: String -> Asm
    AsmJmp :: String -> Asm
    AsmJNE :: String -> Asm
    AsmLabel :: String -> Asm
    AsmLea :: String -> String -> Asm
    AsmMov :: String -> String -> Asm
    AsmMovSXD :: String -> String -> Asm
    AsmMovZX :: String -> String -> Asm
    AsmNeg :: String -> Asm
    AsmNoExecStack :: Asm
    AsmNot :: String -> Asm
    AsmOr :: String -> String -> Asm
    AsmPop :: String -> Asm
    AsmProlog :: Asm
    AsmPush :: String -> Asm
    AsmRet :: Asm
    AsmSal :: String -> String -> Asm
    AsmSar :: String -> String -> Asm
    AsmSETE :: String -> Asm
    AsmSETGE :: String -> Asm
    AsmSETG :: String -> Asm
    AsmSETLE :: String -> Asm
    AsmSETL :: String -> Asm
    AsmSETNE :: String -> Asm
    AsmSub :: String -> String -> Asm
    AsmTest :: String -> String -> Asm
    AsmXor :: String -> String -> Asm
    AsmShr :: String -> String -> Asm
    SecData :: Asm
    SecStr :: String -> Asm
    SectRodata :: Asm
    SectText :: Asm
    StrLabel :: String -> String -> Asm

instance Show Asm where
    show (AsmAdd v1 v2) = "\tadd " ++ v1 ++ ", " ++ v2
    show (AsmAnd v1 v2) = "\tand " ++ v1 ++ ", " ++ v2
    show (AsmCall s) = "\tcall " ++ s
    show (AsmCmp s1 s2) = "\tcmp " ++ s1 ++ ", " ++ s2
    show (AsmDec s) = "\tdec " ++ s
    show (AsmDq s) = "\tdq " ++ s
    show (AsmFuncSpec s) = s
    show (AsmIdiv v1) = "\tidiv " ++ v1
    show (AsmMul s) = "\tmul " ++ s
    show (AsmImul v1 v2) = "\timul " ++ v1 ++ ", " ++ v2
    show (AsmInc s) = "\tinc " ++ s
    show (AsmJG s) = "\tjg " ++ s
    show (AsmJGE s) = "\tjge " ++ s
    show (AsmJL s) = "\tjl " ++ s
    show (AsmJLE s) = "\tjle " ++ s
    show (AsmJNE s) = "\tjne " ++ s
    show (AsmJE s) = "\tje " ++ s
    show (AsmJmp s) = "\tjmp " ++ s
    show (AsmLabel s) = if head s == '.' then s ++ ":" else "\n" ++ s ++ ":"
    show (AsmLea r computation) = "\tlea " ++ r ++ ", " ++ computation
    show (AsmMov s1 s2) = if s2 == "0" && length s1 == 3 then show (AsmXor s1 s1) else "\tmov " ++ s1 ++ ", " ++ s2
    show (AsmMovSXD s1 s2) = "\tmovsxd " ++ s1 ++ ", " ++ s2
    show (AsmMovZX s1 s2) = "\tmovzx " ++ s1 ++ ", " ++ s2
    show (AsmNeg mem) = "\tneg " ++ mem
    show (AsmNot mem) = "\tnot " ++ mem
    show (AsmOr op1 op2) = "\tor " ++ op1 ++ ", " ++ op2
    show (AsmPop r) = "\tpop " ++ r
    show (AsmPush s) = "\tpush " ++ s
    show (AsmSETE s) = "\tsete " ++ s
    show (AsmSETG s) = "\tsetg " ++ s
    show (AsmSETGE s) = "\tsetge " ++ s
    show (AsmSETL s) = "\tsetl " ++ s
    show (AsmSETLE s) = "\tsetle " ++ s
    show (AsmSETNE s) = "\tsetne " ++ s
    show (AsmSal v1 v2) = "\tsal " ++ v1 ++ ", " ++ v2
    show (AsmSar v1 v2) = "\tsar " ++ v1 ++ ", " ++ v2
    show (AsmSub v1 v2) = "\tsub " ++ v1 ++ ", " ++ v2
    show (AsmTest op1 op2) = "\ttest " ++ op1 ++ ", " ++ op2
    show (AsmXor op1 op2) = "\txor " ++ op1 ++ ", " ++ op2
    show (AsmShr v1 v2) = "\tshr " ++ v1 ++ ", " ++ v2
    show (SecStr s) = "\tdb " ++ show s ++ ", 0"
    show (StrLabel lbl valStr) = "\t" ++ lbl ++ ": db " ++ show valStr ++ ", 0"
    show AsmCdq = "\tcdq"
    show AsmEpilog = "\tpop rbp\n\tret"
    show AsmExtern = "\textern "
    show AsmGlobl = "\tglobal main"
    show AsmNoExecStack = "section .note.GNU-stack noalloc noexec nowrite progbits"
    show AsmProlog = "\tpush rbp\n\tmov rbp, rsp"
    show AsmRet = "\tret"
    show SecData = "section .data"
    show SectRodata = "section .rodata"
    show SectText = "\nsection .text"

vTypeWord :: VType -> String
vTypeWord IntQ = "dword"
vTypeWord StringQ = "qword"
vTypeWord BoolQ = "byte"
vTypeWord (ClassQ _) = "qword"
vTypeWord _ = error "unreachable"

data StoragePlace where
    OffsetRBP :: Int -> VType -> StoragePlace
    Register :: AsmRegister -> StoragePlace
    ProgLabel :: String -> StoragePlace
    NullAddr :: StoragePlace
    IntConst :: Int -> StoragePlace
    BoolConst :: Bool -> StoragePlace
    StringConst :: String -> StoragePlace
    deriving (Eq)
instance Show StoragePlace where
    show (OffsetRBP i vt) = vTypeWord vt ++ " " ++ if i > 0 then "[rbp + " ++ show i ++ "]" else "[rbp - " ++ show (-i) ++ "]"
    show (Register reg) = show reg
    show (ProgLabel l) = l
    show NullAddr = "0"
    show (IntConst i) = show i
    show (BoolConst b) = if b then "1" else "0"
    show (StringConst s) = show s

getStoragePlaceType :: StoragePlace -> VType
getStoragePlaceType (OffsetRBP _ vt) = vt
getStoragePlaceType (IntConst _) = IntQ
getStoragePlaceType (BoolConst _) = BoolQ
getStoragePlaceType (Register _) = StringQ
getStoragePlaceType NullAddr = ClassQ ""
getStoragePlaceType _ = error "unreachable"

isConstStorage :: StoragePlace -> Bool
isConstStorage (IntConst _) = True
isConstStorage (BoolConst _) = True
isConstStorage _ = False

data AsmStore = AStore
    { asmenv :: Map.Map String (Address, StoragePlace)
    , curFuncNameAsm :: String
    , lastAddrRBP :: Int
    , builtinFuncExt :: [String]
    , curRSP :: Int
    , curRSPOffset :: Int
    , strLabelsCounter :: Int
    , strLabels :: Map.Map String Bool
    , labelsCounter :: Int
    , asmargs :: [Quad]
    , vtables :: [(String, [String])]
    }
    deriving (Show)

asInit :: (Map.Map String ClassData, [String]) -> AsmStore
asInit (classes, builtins) =
    AStore
        { asmenv = Map.empty
        , curFuncNameAsm = ""
        , lastAddrRBP = 0
        , builtinFuncExt = builtins
        , curRSP = 0
        , curRSPOffset = 0
        , strLabelsCounter = 0
        , strLabels = Map.empty
        , labelsCounter = 0
        , asmargs = []
        , vtables = map createVTable $ Map.elems classes
        }

createVTable :: ClassData -> (String, [String])
createVTable (ClassData name _ methodsMap _ _) = ("_vtable_" ++ name, map snd (sortOn fst (map (second funcName . snd) (Map.toList methodsMap))))
