{-# OPTIONS_GHC -Wno-missing-signatures #-}

module Asm.Asm (genAssembly) where

import Asm.DataTypes
import Control.Monad.State
import Control.Monad.Writer
import Data.Bits
import Data.List
import qualified Data.Map as Map
import Data.Maybe
import Quad.DataTypes

arguments32Bit = [RegEDI, RegESI, RegEDX, RegECX, RegR8D, RegR9D]
arguments64Bit = [RegRDI, RegRSI, RegRDX, RegRCX, RegR8, RegR9]
arguments08Bits = [RegDIL, RegSIL, RegDL, RegCL, RegR8B, RegR9B]

argumentsRegisterNum = 6
argumentsStackSize = 8
argumentsStartOffset = 16

stackAlignment = 16
bytesStack = 8

builtins = ["printInt", "printString", "readInt", "readString", "error"]

type AsmMonad a = StateT AsmStore (WriterT [Asm] IO) a

genAssembly qs qc = runWriterT $ evalStateT (runGenAsm qc) (asInit qs)

runGenAsm :: QuadCode -> AsmMonad ()
runGenAsm qc = do
    tell [AsmNoExecStack]
    tell [SecData]
    let saveStringLiteralInDataSec s = do
            curStrLblCnt <- gets strLabelsCounter
            let newStrLbl = "LS" ++ show curStrLblCnt
            l <- gets (Map.lookup s . strLabels)
            case l of
                Just _ -> return ()
                Nothing -> do
                    modify (\cur -> cur{strLabelsCounter = curStrLblCnt + 1, strLabels = Map.insert s True (strLabels cur)})
                    let s' = drop 4 s -- NOTE: this is a dirty hack described in Quad.hs
                    tell [StrLabel newStrLbl s']
                    modify (\cur -> cur{asmenv = Map.insert s (QNullptr, ProgLabel newStrLbl) (asmenv cur)})
    let prepareFunc (QFunc f@(FuncData{})) = do
            let args = "str-" : stringVars f -- NOTE: this is a dirty hack described in Quad.hs
            mapM_ saveStringLiteralInDataSec args
        prepareFunc _ = error "unreachable"
    mapM_ prepareFunc qc
    let addVTables = mapM_ $ \(vt, funcs) -> do
            tell [AsmLabel vt]
            mapM_ (tell . pure . AsmDq) funcs
    gets vtables >>= addVTables
    tell [SectText, AsmGlobl]
    let getBuiltinWrapped s = show AsmExtern ++ intercalate ", " s
        addExternals = tell . map (AsmFuncSpec . getBuiltinWrapped . (: []))
    gets builtinFuncExt >>= addExternals
    mapM_ genAsm qc

genAsm :: Quad -> AsmMonad ()
genAsm (QAss var val) = do
    s1 <- getStorage val
    s <- getStorage var
    case val of
        (QString str) -> movStrToMem s str
        (QLoc _ valType) -> movMemToMem s s1 valType
        (QInt _) -> tell [AsmMov (show s) (show s1)]
        (QBool _) -> tell [AsmMov (show s) (show s1)]
        _ -> error "unreachable"
genAsm q@(QBinOp _ _ AopAdd _) = do
    (s, s1, s2) <- getStorageBinOp q
    tell [AsmMov (show RegEAX) (show s1)]
    tell [AsmAdd (show RegEAX) (show s2)]
    tell [AsmMov (show s) (show RegEAX)]
genAsm q@(QBinOp _ _ AopSub _) = do
    (s, s1, s2) <- getStorageBinOp q
    tell [AsmMov (show RegEAX) (show s1)]
    tell [AsmSub (show RegEAX) (show s2)]
    tell [AsmMov (show s) (show RegEAX)]
genAsm q@(QBinOp _ _ AopMul _) = do
    (s, s1, s2) <- getStorageBinOp q
    case s2 of
        (IntConst n2)
            | isPowerOf2 n2 -> do
                tell [AsmMov (show RegEAX) (show s1)]
                tell [AsmSal (show RegEAX) (show (log2 n2))]
                tell [AsmMov (show s) (show RegEAX)]
        _ -> do
            tell [AsmMov (show RegEAX) (show s1)]
            tell [AsmImul (show RegEAX) (show s2)]
            tell [AsmMov (show s) (show RegEAX)]
genAsm q@(QBinOp _ _ AopDiv _) = do
    (s, s1, s2) <- getStorageBinOp q
    case s2 of
        IntConst d | isPowerOf2 d -> do
            tell [AsmMov (show RegEAX) (show s1)]
            tell [AsmSar (show RegEAX) (show (log2 d))]
            tell [AsmMov (show s) (show RegEAX)]
        IntConst d | d > 0 -> do
            let p = if 2 ^ log2 d < d then log2 d + 1 else log2 d
            let m = ((2 ^ (32 + p) + d - 1) `div` d) .&. (2 ^ ((32 :: Integer) - 1) - 1)
            -- q = m * s1 >> 32
            -- t = ((n - q) >> 1) + q
            -- s = t >> (p - 1)
            tell [AsmMov (show RegEDI) (show s1)]
            tell [AsmMov (show RegEAX) (show RegEDI)]
            tell [AsmMov (show RegECX) (show m)]
            tell [AsmMul (show RegECX)]
            tell [AsmSub (show RegEDI) (show RegEDX)]
            tell [AsmShr (show RegEDI) "1"]
            tell [AsmAdd (show RegEDI) (show RegEDX)]
            tell [AsmShr (show RegEDI) (show (p - 1))]
            tell [AsmMov (show s) (show RegEDI)]
        _ -> do
            tell [AsmMov (show RegEAX) (show s1)]
            tell [AsmMov (show RegECX) (show s2)]
            tell [AsmCdq]
            tell [AsmIdiv (show RegECX)]
            tell [AsmMov (show s) (show RegEAX)]
genAsm q@(QBinOp _ _ AopMod _) = do
    (s, s1, s2) <- getStorageBinOp q
    case s2 of
        IntConst d | isPowerOf2 d -> do
            tell [AsmMov (show RegEAX) (show s1)]
            tell [AsmAnd (show RegEAX) (show (d - 1))]
            tell [AsmMov (show s) (show RegEAX)]
        _ -> do
            -- TODO: optimize to remove division
            tell [AsmMov (show RegEAX) (show s1)]
            tell [AsmMov (show RegECX) (show s2)]
            tell [AsmCdq]
            tell [AsmIdiv (show RegECX)]
            tell [AsmMov (show s) (show RegEDX)]
genAsm param@(QParam _) = modify (\cur -> cur{asmargs = asmargs cur ++ [param]})
genAsm ((QCall var@(QLoc _ varType) ident numArgs)) = do
    args <- gets asmargs
    genArgs args arguments64Bit arguments32Bit
    modify (\cur -> cur{asmargs = []})
    tell [AsmCall ident]
    unless (ident `elem` builtins) $ do
        let pushed = numArgs - argumentsRegisterNum
        when (pushed > 0) $ do
            let size = argumentsStackSize * pushed
            modify (\cur -> cur{curRSP = curRSP cur - size})
            tell [AsmAdd (show RegRSP) (show size)]
    -- TODO: this is ugly
    case varType of
        IntQ -> do
            s <- getStorage var
            tell [AsmMov (show s) (show RegEAX)]
        StringQ -> do
            s <- getStorage var
            tell [AsmMov (show s) (show RegRAX)]
        BoolQ -> do
            s <- getStorage var
            tell [AsmMov (show s) (show RegAL)]
        VoidQ -> pure ()
        ClassQ _ -> do
            s <- getStorage var
            tell [AsmMov (show s) (show RegRAX)]
genAsm (QCond var val1 val2 ct') = do
    s1' <- getStorage val1
    s2' <- getStorage val2
    let (s1, s2, ct) = if isConstStorage s1' then (s2', s1', rotateCondType ct') else (s1', s2', ct')
    s <- getStorage var
    case getStoragePlaceType s1 of
        IntQ -> do
            tell [AsmMov (show RegEAX) (show s1)]
            tell [AsmCmp (show RegEAX) (show s2)]
        BoolQ -> do
            tell [AsmMov (show RegAL) (show s1)]
            tell [AsmCmp (show RegAL) (show s2)]
        _ -> do
            tell [AsmMov (show RegRAX) (show s1)]
            tell [AsmCmp (show RegRAX) (show s2)]
    case ct of
        CondEQ -> tell [AsmSETE (show RegAL)]
        CondNE -> tell [AsmSETNE (show RegAL)]
        CondGT -> tell [AsmSETG (show RegAL)]
        CondGE -> tell [AsmSETGE (show RegAL)]
        CondLT -> tell [AsmSETL (show RegAL)]
        CondLE -> tell [AsmSETLE (show RegAL)]
        CondAnd -> tell [AsmAnd (show RegAL) (show s2)]
        CondOr -> tell [AsmOr (show RegAL) (show s2)]
    case getStoragePlaceType s of
        IntQ -> tell [AsmMov (show s) (show RegEAX)]
        BoolQ -> tell [AsmMov (show s) (show RegAL)]
        _ -> tell [AsmMov (show s) (show RegRAX)]
genAsm (QFunc (FuncData name _ args body _ stackSize)) = do
    initialEnv <- gets asmenv
    tell [AsmLabel name]
    tell [AsmProlog]
    let alignment = stackSize + ((stackAlignment - (stackSize `mod` stackAlignment)) `mod` stackAlignment)
    modify (\cur -> cur{curRSP = curRSP cur + alignment})
    tell [AsmSub (show RegRSP) (show alignment)]
    modify (\cur -> cur{curFuncNameAsm = name})
    movArgs args arguments64Bit arguments32Bit arguments08Bits
    mapM_ genAsm body
    createEndRetLabel >>= tell . pure . AsmLabel
    when (stackSize > 0) $ tell [AsmMov (show RegRSP) (show RegRBP)]
    tell [AsmEpilog]
    modify (\cur -> cur{asmenv = initialEnv, lastAddrRBP = 0, curRSP = 0})
genAsm (QJmp label) = do
    (isNew, cl) <- getLabelOfStringOrLabel label
    tell [AsmJmp (show cl)]
    when isNew $ modify (\cur -> cur{asmenv = Map.insert label (QNullptr, cl) (asmenv cur)})
genAsm (QJmpCMP var lt lf) = do
    (isNewT, clt) <- getLabelOfStringOrLabel lt
    (isNewF, clf) <- getLabelOfStringOrLabel lf
    s <- getStorage var
    tell [AsmTest (show s) "1"]
    tell [AsmJNE (show clt)]
    when isNewT $ modify (\cur -> cur{asmenv = Map.insert lt (QNullptr, clt) (asmenv cur)})
    tell [AsmJmp (show clf)]
    when isNewF $ modify (\cur -> cur{asmenv = Map.insert lf (QNullptr, clf) (asmenv cur)})
genAsm (QLabel labelFalse) = do
    (isNew, codeLabel) <- getLabelOfStringOrLabel labelFalse
    tell [AsmLabel (show codeLabel)]
    when isNew $ modify (\cur -> cur{asmenv = Map.insert labelFalse (QNullptr, codeLabel) (asmenv cur)})
genAsm (QNeg var val) = do
    s1 <- getStorage val
    s <- getStorage var
    tell [AsmMov (show RegEAX) (show s1)]
    tell [AsmNeg (show RegEAX)]
    tell [AsmMov (show s) (show RegEAX)]
genAsm (QRet res) = do
    s <- getStorage res
    case getStoragePlaceType s of
        IntQ -> tell [AsmMov (show RegEAX) (show s)]
        BoolQ -> tell [AsmMov (show RegAL) (show s)]
        _ -> tell [AsmMov (show RegRAX) (show s)]
    genAsm QVRet
genAsm QVRet = createEndRetLabel >>= tell . pure . AsmJmp
genAsm (QLoad var@(QLoc{}) val offset) = do
    s1 <- getStorage val
    s <- getStorage var
    tell [AsmMov (show RegRAX) (show s1)]
    let reg = case getStoragePlaceType s of
            IntQ -> RegEAX
            BoolQ -> RegAL
            _ -> RegRAX
    tell [AsmMov (show reg) ("[" ++ show RegRAX ++ "+" ++ show offset ++ "]")]
    tell [AsmMov (show s) (show reg)]
genAsm (QStore var@(QLoc{}) val offset) = do
    s1 <- getStorage val
    s <- getStorage var
    let rega = case getStoragePlaceType s of
            IntQ -> RegEAX
            BoolQ -> RegAL
            _ -> RegRAX
    let regc = case getStoragePlaceType s1 of
            IntQ -> RegECX
            BoolQ -> RegCL
            _ -> RegRCX
    tell [AsmMov (show rega) (show s)]
    tell [AsmMov (show regc) (show s1)]
    tell [AsmMov ("[" ++ show RegRAX ++ "+" ++ show offset ++ "]") (show regc)]
genAsm (QCallMethod var@(QLoc{}) offset nargs) = do
    args <- gets asmargs
    genArgs args arguments64Bit arguments32Bit
    modify (\cur -> cur{asmargs = []})
    tell [AsmMov (show RegRAX) (show RegRDI)]
    tell [AsmMov (show RegRAX) ("[" ++ show RegRAX ++ "]")]
    tell [AsmAdd (show RegRAX) (show offset)]
    tell [AsmMov (show RegRAX) ("[" ++ show RegRAX ++ "]")]
    tell [AsmCall (show RegRAX)]
    let pushed = nargs - argumentsRegisterNum
    when (pushed > 0) $ do
        let size = argumentsStackSize * pushed
        modify (\cur -> cur{curRSP = curRSP cur - size})
        tell [AsmAdd (show RegRSP) (show size)]
    s <- getStorage var
    case getStoragePlaceType s of
        VoidQ -> pure ()
        BoolQ -> tell [AsmMov (show s) (show RegAL)]
        IntQ -> tell [AsmMov (show s) (show RegEAX)]
        _ -> tell [AsmMov (show s) (show RegRAX)]
genAsm _ = error "unreachable"

-- Function Arguments {{{
genArgs :: [Quad] -> [AsmRegister] -> [AsmRegister] -> AsmMonad ()
genArgs ((QParam val) : rest) (reg : regs) (ereg : eregs) = do
    case val of
        (QInt v) -> tell [AsmMov (show ereg) (show v)]
        (QLoc{}) -> do
            s <- getStorage val
            case getStoragePlaceType s of
                IntQ -> tell [AsmMov (show ereg) (show s)]
                BoolQ -> tell [AsmMovZX (show reg) (show s)]
                _ -> tell [AsmMov (show reg) (show s)]
        (QString s) -> do
            r <- gets (Map.lookup s . asmenv)
            case r of
                Just (_, lbl) -> tell [AsmMov (show reg) (show lbl)]
                Nothing -> tell [AsmMov (show reg) s]
        b@(QBool{}) -> tell [AsmMov (show ereg) (show b)]
        _ -> error "unreachable"
    genArgs rest regs eregs
genArgs qcode [] _ = do
    let paramsToStack (qparam@(QParam _) : rest) accum = qparam : paramsToStack rest accum
        paramsToStack _ accum = accum
    let reverseParams = paramsToStack qcode []
    pushArgs reverseParams
genArgs _ _ _ = return ()

movArgs args [] [] [] = movArgsFromStack args argumentsStartOffset
movArgs [] _ _ _ = return ()
movArgs ((QLoc ident valType) : args) (reg : regs) (ereg : eregs) (areg : aregs) = do
    s <- getStorage (QLoc ident valType)
    -- TODO: some better wey to choose register
    case valType of
        IntQ -> tell [AsmMov (show s) (show ereg)]
        BoolQ -> tell [AsmMov (show s) (show areg)]
        _ -> tell [AsmMov (show s) (show reg)]
    movArgs args regs eregs aregs
movArgs _ _ _ _ = error "unreachable"

movArgsFromStack [] _ = return ()
movArgsFromStack ((QLoc ident valType) : args) stackOffset = case valType of
    IntQ -> do
        -- FIXME: might be very wrong
        let s = OffsetRBP stackOffset valType
        tell [AsmMov (show RegEAX) (show s)]
        let var = QLoc ident valType
        modify (\cur -> cur{asmenv = Map.insert ident (var, s) (asmenv cur)})
        movArgsFromStack args (stackOffset + argumentsStackSize)
    BoolQ -> error "unimplemented"
    StringQ -> error "unimplemented"
    _ -> error "unreachable"
movArgsFromStack _ _ = error "unreachable"

pushArgs [] = return ()
pushArgs ((QParam val) : rest) = do
    case val of
        (QLoc{}) -> do
            s <- getStorage val
            case getStoragePlaceType s of
                IntQ -> tell [AsmMov (show RegEAX) (show s)]
                BoolQ -> tell [AsmMovZX (show RegRAX) (show s)]
                _ -> error "unreachable"
            tell [AsmPush (show RegRAX)]
        (QInt{}) -> do
            tell [AsmPush (show val)]
            modify (\cur -> cur{curRSP = curRSP cur + bytesStack})
        (QBool{}) -> do
            tell [AsmMovZX (show RegRAX) (show val)]
            tell [AsmPush (show RegRAX)]
        (QString s) -> do
            (_, lbl) <- gets (fromJust . Map.lookup s . asmenv)
            tell [AsmPush (show lbl)]
        _ -> error "unreachable"
    modify (\cur -> cur{curRSP = curRSP cur + bytesStack})
    pushArgs rest
pushArgs _ = error "unreachable"

-- }}}
-- Storage {{{
getStorageBinOp :: Quad -> AsmMonad (StoragePlace, StoragePlace, StoragePlace)
getStorageBinOp (QBinOp var val1 op val2) = do
    s <- getStorage var
    s1' <- getStorage val1
    s2' <- getStorage val2
    let (s1, s2) = if isConstStorage s1' && op `elem` [AopAdd, AopMul] then (s2', s1') else (s1', s2')
    return (s, s1, s2)
getStorageBinOp _ = error "unreachable"

getStorage :: Address -> AsmMonad StoragePlace
getStorage addr@(QLoc ident valType) = do
    asmenv <- gets asmenv
    case Map.lookup ident asmenv of
        Just (_address, OffsetRBP off _) -> return $ OffsetRBP off valType
        Just (_address, storage) -> return storage
        Nothing -> do
            rbp <- gets lastAddrRBP
            let newRBPOffset = rbp - vTypeSize valType
            let storage = OffsetRBP newRBPOffset valType
            modify (\cur -> cur{asmenv = Map.insert ident (addr, storage) asmenv})
            modify (\cur -> cur{lastAddrRBP = newRBPOffset})
            return storage
getStorage (QInt n) = return $ IntConst n
getStorage (QBool b) = return $ BoolConst b
getStorage (QString s) = return $ StringConst s
getStorage QNullptr = return NullAddr

-- }}}
-- Helpers {{{
createEndRetLabel = gets (((".L" ++) . (++ "_end")) . curFuncNameAsm)

movMemToMem memToL memFromR valType = do
    let createMemAddr memStorage locType = do
            case memStorage of
                OffsetRBP offset _ -> vTypeWord locType ++ " [rbp" ++ (if offset < 0 then show offset else "+" ++ show offset) ++ "]"
                Register reg -> show reg
                ProgLabel l -> l
                _ -> error "unreachable"
    let isOffset (OffsetRBP{}) = True
        isOffset _ = False
    let moveTempToRAX memStorageAddr = do
            let reg = case valType of
                    IntQ -> RegEAX
                    BoolQ -> RegAL
                    _ -> RegRAX
            tell [AsmMov (show reg) memStorageAddr]
            return reg
    let rightAddr = createMemAddr memFromR valType
    let leftAddr = createMemAddr memToL valType
    if isOffset memFromR && isOffset memToL
        then do
            reg <- moveTempToRAX rightAddr
            tell [AsmMov leftAddr (show reg)]
        else
            tell [AsmMov leftAddr rightAddr]

movStrToMem memToL fullStr = do
    (_, lbl) <- gets (fromJust . Map.lookup fullStr . asmenv)
    movMemToMem memToL lbl StringQ

getLabelOfStringOrLabel origLabel = do
    -- FIXME: this is generated by copilot
    let createNewLabelUpdateCounter = do
            curLabelNr <- gets labelsCounter
            let newLabel = ".L" ++ show curLabelNr
            modify (\cur -> cur{labelsCounter = curLabelNr + 1})
            return newLabel
    lblData <- gets (Map.lookup origLabel . asmenv)
    case lblData of
        Nothing -> do
            newLabel <- createNewLabelUpdateCounter
            return (True, ProgLabel newLabel)
        Just (_, codeLabel) -> return (False, codeLabel)

isPowerOf2 :: Int -> Bool
isPowerOf2 n = n > 0 && (n .&. (n - 1)) == 0
log2 :: Int -> Int
log2 n = if n <= 1 then 0 else 1 + log2 (n `div` 2)

-- }}}
