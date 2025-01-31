{-# OPTIONS_GHC -Wno-missing-signatures #-}

module Quad.DataTypes where

import qualified Data.Map as Map
import Parser.Abs

type Loc = Int
type Env = Map.Map String Loc
data QuadStore = QuadStore
    { env :: Env
    , storeQ :: Map.Map Loc (Label, Address)
    , lastLocQ :: Loc
    , curFuncName :: String
    , curClassName :: String
    , builtinFunc :: [String]
    , defFunc :: Map.Map String FuncData
    , defClass :: Map.Map String ClassData
    , countLabels :: Map.Map String Int
    , exprDepth :: Map.Map Expr Int
    }
    deriving (Show)

qsEmpty :: QuadStore
qsEmpty =
    QuadStore
        { env = Map.empty
        , storeQ = Map.empty
        , lastLocQ = 0
        , curFuncName = ""
        , curClassName = ""
        , builtinFunc = []
        , defFunc = Map.empty
        , defClass = Map.empty
        , countLabels = Map.empty
        , exprDepth = Map.empty
        }

data VType where
    IntQ :: VType
    StringQ :: VType
    BoolQ :: VType
    VoidQ :: VType
    ClassQ :: String -> VType
    deriving (Eq, Show, Ord)

vTypeSize :: VType -> Int
vTypeSize IntQ = 4
vTypeSize StringQ = 8
vTypeSize BoolQ = 1
vTypeSize VoidQ = 0
vTypeSize (ClassQ _) = 8

type Label = String

prettyQuad :: [Quad] -> String
prettyQuad =
    concatMap
        ( \q -> case q of
            (QLabel _) -> show q ++ "\n"
            _ -> "\t" ++ show q ++ "\n"
        )

data FuncData = FuncData
    { funcName :: String
    , returnType :: VType
    , arguments :: [Address]
    , body :: [Quad]
    , stringVars :: [String]
    , stackSize :: Int
    }
instance Show FuncData where
    show (FuncData fname rettype args body _ _) =
        fname ++ "(" ++ concatMap (\arg -> show arg ++ ",") args ++ ") -> " ++ show rettype ++ " {\n" ++ prettyQuad body ++ "}\n"

funcEmpty fname rettype = FuncData fname (getOrigQType rettype) [] [] [] 0
funcUpdStringVars strAddress fbody = fbody{stringVars = strAddress : stringVars fbody}

data ClassData = ClassData
    { className :: String
    , attributes :: Map.Map String (Int, VType) -- offset, type
    , methods :: Map.Map String (Int, FuncData) -- offset, function
    , attrSize :: Int
    , methSize :: Int
    }
    -- TODO: create own show instance
    deriving (Show)

classEmpty cname = ClassData cname Map.empty Map.empty 8 0
classExtend :: String -> ClassData -> [ClassElem] -> ClassData
classExtend cname cd [] = cd{className = cname}
classExtend cname cd ((ClassAttrDef _ _ []) : elems) = classExtend cname cd elems
classExtend cname cd ((ClassAttrDef pos t ((ClassItem _ (Ident attr)) : items)) : elems) =
    classExtend
        cname
        ( ClassData
            { className = className cd
            , attributes = attributes'
            , methods = methods cd
            , attrSize = attrSize cd + size'
            , methSize = methSize cd
            }
        )
        (ClassAttrDef pos t items : elems)
  where
    size = case t of
        Int _ -> 4
        Str _ -> 8
        Bool _ -> 1
        Class _ _ -> 8
        _ -> error "unreachable"
    (size', attributes') = if Map.member attr (attributes cd) then (0, attributes cd) else (size, Map.insert attr (attrSize cd, getOrigQType t) (attributes cd))
classExtend cname cd ((ClassMethodDef _ rettype (Ident method) _ _) : elems) =
    classExtend
        cname
        ( ClassData
            { className = className cd
            , attributes = attributes cd
            , methods = methods'
            , attrSize = attrSize cd
            , methSize = methSize cd + size'
            }
        )
        elems
  where
    (size', methods') = case Map.lookup method (methods cd) of
        Just (offset, _) -> (0, Map.insert method (offset, funcEmpty ("_" ++ cname ++ "_" ++ method) rettype) (methods cd))
        Nothing -> (8, Map.insert method (methSize cd, funcEmpty ("_" ++ cname ++ "_" ++ method) rettype) (methods cd))

data ArithmeticOp where
    AopAdd :: ArithmeticOp
    AopSub :: ArithmeticOp
    AopMul :: ArithmeticOp
    AopDiv :: ArithmeticOp
    AopMod :: ArithmeticOp
    deriving (Eq, Ord, Read)
instance Show ArithmeticOp where
    show AopAdd = "add"
    show AopSub = "sub"
    show AopMul = "mul"
    show AopDiv = "div"
    show AopMod = "mod"

data ConditionalOp where
    CondEQ :: ConditionalOp
    CondNE :: ConditionalOp
    CondGT :: ConditionalOp
    CondGE :: ConditionalOp
    CondLT :: ConditionalOp
    CondLE :: ConditionalOp
    CondAnd :: ConditionalOp
    CondOr :: ConditionalOp
instance Show ConditionalOp where
    show CondEQ = "=="
    show CondNE = "!="
    show CondGT = ">"
    show CondGE = ">="
    show CondLT = "<"
    show CondLE = "<="
    show CondAnd = "&&"
    show CondOr = "||"

rotateCondType :: ConditionalOp -> ConditionalOp
rotateCondType CondEQ = CondEQ
rotateCondType CondNE = CondNE
rotateCondType CondGT = CondLT
rotateCondType CondGE = CondLE
rotateCondType CondLT = CondGT
rotateCondType CondLE = CondGE
rotateCondType CondAnd = CondAnd
rotateCondType CondOr = CondOr

data Address where
    QLoc :: String -> VType -> Address
    QInt :: Int -> Address
    QBool :: Bool -> Address
    QString :: String -> Address
    QNullptr :: Address
    deriving (Eq, Ord)
instance Show Address where
    show (QLoc s _) = s
    show (QInt i) = show i
    show (QBool b) = if b then "1" else "0"
    show (QString s) = s
    show QNullptr = "nullptr"

data Quad where
    QAss :: Address -> Address -> Quad
    QBinOp :: Address -> Address -> ArithmeticOp -> Address -> Quad
    QCall :: Address -> String -> Int -> Quad
    QCallMethod :: Address -> Int -> Int -> Quad
    QCond :: Address -> Address -> Address -> ConditionalOp -> Quad
    QFunc :: FuncData -> Quad
    QJmpCMP :: Address -> String -> String -> Quad
    QJmp :: String -> Quad
    QLabel :: String -> Quad
    QLoad :: Address -> Address -> Int -> Quad
    QNeg :: Address -> Address -> Quad
    QParam :: Address -> Quad
    QRet :: Address -> Quad
    QStore :: Address -> Address -> Int -> Quad
    QVRet :: Quad

instance Show Quad where
    show (QAss var val) = show var ++ " = " ++ show val
    show (QBinOp var val1 op val2) = show var ++ " = " ++ show op ++ " " ++ show val1 ++ " " ++ show val2
    show (QCall var s n) = show var ++ " = call " ++ s ++ " " ++ show n
    show (QCallMethod var i n) = show var ++ " = call " ++ " [" ++ show i ++ "] " ++ show n
    show (QCond var val1 val2 ct) = show var ++ " = " ++ show val1 ++ " " ++ show ct ++ " " ++ show val2
    show (QFunc fd) = show fd
    show (QJmp s) = "jmp " ++ s
    show (QJmpCMP var l1 l2) = "jmpcmp " ++ show var ++ " " ++ l1 ++ " " ++ l2
    show (QLabel s) = s ++ ":"
    show (QLoad var val i) = show var ++ " = *(" ++ show val ++ "+" ++ show i ++ ")"
    show (QNeg var val) = show var ++ " = -" ++ show val
    show (QParam val) = "param " ++ show val
    show (QRet val) = "ret " ++ show val
    show (QStore var val i) = "*(" ++ show var ++ "+" ++ show i ++ ") = " ++ show val
    show QVRet = "vret"

type QuadCode = [Quad]

getOrigQType (Int _) = IntQ
getOrigQType (Str _) = StringQ
getOrigQType (Bool _) = BoolQ
getOrigQType (Void _) = VoidQ
getOrigQType (Class _ (Ident c)) = ClassQ c
getOrigQType _ = error "unreachable"

getRelOperandQuad op qvar val1 val2 =
    case op of
        (EQU _) -> [QCond qvar val1 val2 CondEQ]
        (NE _) -> [QCond qvar val1 val2 CondNE]
        (GE _) -> [QCond qvar val1 val2 CondGE]
        (GTH _) -> [QCond qvar val1 val2 CondGT]
        (LE _) -> [QCond qvar val1 val2 CondLE]
        (LTH _) -> [QCond qvar val1 val2 CondLT]

extractString (QString s) = s
extractString _ = error "unreachable"

isRawString (QString _) = True
isRawString _ = False
isRawStringToInt (QString _) = 1
isRawStringToInt _ = 0

getVType (QInt _) = IntQ
getVType (QLoc _ vtype) = vtype
getVType (QString _) = StringQ
getVType (QBool _) = BoolQ
getVType QNullptr = IntQ

getVarName (QLoc s _) = s
getVarName _ = error "unreachable"

getVarType (QLoc _ vtype) = vtype
getVarType _ = error "unreachable"
