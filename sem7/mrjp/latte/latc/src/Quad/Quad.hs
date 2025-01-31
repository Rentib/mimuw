{-# OPTIONS_GHC -Wno-missing-signatures #-}

module Quad.Quad (genQuad) where

import Control.Monad.Except
import Control.Monad.State
import Control.Monad.Writer
import qualified Data.Map as Map
import Data.Maybe
import Parser.Abs
import Quad.DataTypes

type QuadMonad a = StateT QuadStore (ExceptT String (WriterT QuadCode IO)) a

builtinFuncsList = ["printInt", "printString", "error", "readInt", "readString"]
strMangle = "str-" -- NOTE: this is a hack to prevent name clashes with user-defined stuff, as they cant have - in them

genQuad :: Program -> IO (Either String (VType, QuadStore), QuadCode)
genQuad program = runWriterT $ runExceptT $ evalStateT (runQuadGen program) qsEmpty

runQuadGen (Program _ topDefs) = do
    mapM_ genEmptyTopDef topDefs -- definitions
    mapM_ genTopDef topDefs -- declarations
    get >>= \qs -> return (IntQ, qs)

genEmptyTopDef :: TopDef -> QuadMonad ()
genEmptyTopDef (FnDef _ rettype (Ident fname) _ _) =
    qsNewFunction fname (funcEmpty fname rettype)
genEmptyTopDef (ClassDef _ (Ident cname) elems) =
    qsNewClass cname (classExtend cname (classEmpty cname) elems)
genEmptyTopDef (ClassExt _ (Ident cname) (Ident super) elems) = do
    -- FIXME: this assumes classes are provided in a correct order
    detClass <- gets defClass
    let s = fromJust $ Map.lookup super detClass
    qsNewClass cname (classExtend cname s elems)

genTopDef (FnDef _ rettype' (Ident ident) args (Block _ stmts)) = do
    let rettype = getOrigQType rettype'
    envWithParams <- saveArgsToEnv args
    let fdata =
            FuncData
                { funcName = ident
                , returnType = rettype
                , arguments = map getArgAddress args
                , body = []
                , stringVars = []
                , stackSize = 0
                }
    qsNewFunction ident fdata
    get >>= \cur -> put cur{curFuncName = ident}
    modify (\cur -> cur{env = envWithParams})
    funcBody <- concat <$> mapM genStmt stmts
    labels <- gets countLabels
    modify (\cur -> cur{countLabels = Map.insert ".t" 0 labels})
    cur <- get
    let fname = curFuncName cur
    f <- gets (fromJust . Map.lookup fname . defFunc)
    let f' = f{body = funcBody}
    put cur{defFunc = Map.insert fname f' (defFunc cur)}
    tell [QFunc f']
genTopDef (ClassDef _ (Ident c) elems) = do
    modify (\cur -> cur{curClassName = c})
    mapM_ genClassElem elems
    modify (\cur -> cur{curClassName = ""})
genTopDef (ClassExt pos (Ident c) _ elems) = do
    genTopDef (ClassDef pos (Ident c) elems)

genClassElem :: ClassElem -> QuadMonad ()
genClassElem (ClassAttrDef{}) = tell [] -- who cares???
genClassElem (ClassMethodDef pos rettype (Ident method') args' (Block _ stmts)) = do
    -- TODO: refactor
    let rettype' = getOrigQType rettype
    c <- gets curClassName
    let method = "_" ++ c ++ "_" ++ method'
    let args = Arg pos (Class pos (Ident c)) (Ident "self") : args'
    envWithParams <- saveArgsToEnv args
    let fdata =
            FuncData
                { funcName = method
                , returnType = rettype'
                , arguments = map getArgAddress args
                , body = []
                , stringVars = []
                , stackSize = 0
                }
    qsNewFunction method fdata
    get >>= \cur -> put cur{curFuncName = method}
    modify (\cur -> cur{env = envWithParams})
    funcBody <- concat <$> mapM genStmt stmts
    labels <- gets countLabels
    modify (\cur -> cur{countLabels = Map.insert ".t" 0 labels})
    cur <- get
    let fname = curFuncName cur
    f <- gets (fromJust . Map.lookup fname . defFunc)
    let f' = f{body = funcBody}
    put cur{defFunc = Map.insert fname f' (defFunc cur)}
    tell [QFunc f']

genStmt :: Stmt -> QuadMonad [Quad]
genStmt (Empty _) = return []
genStmt (BStmt _ (Block _ stmts)) = do
    oldEnv <- gets env
    res <- concat <$> mapM genStmt stmts
    modify (\cur -> cur{env = oldEnv})
    return res
genStmt (Decl _ _ []) = return []
genStmt (Decl pos t ((Init _ (Ident ident) e) : items)) = do
    (val, updcode) <- genExpr e
    cur <- get
    case val of
        (QInt _) -> return ()
        (QLoc _ retType) -> do
            case retType of
                IntQ -> return ()
                StringQ -> return ()
                BoolQ -> return ()
                _ -> return ()
        (QString strVal) -> addConstString strVal
        (QBool _) -> return ()
        _ -> error "unreachable"
    countIdent <- gets (Map.lookup ident . countLabels)
    newName <- case countIdent of
        Nothing -> do
            qsNewLabel ident
            let newName = ident ++ "_0"
            return newName
        Just curNumId -> do
            modify (\cur' -> cur'{countLabels = Map.insert ident (curNumId + 1) (countLabels cur)})
            let newName = ident ++ "_" ++ show curNumId
            return newName
    qsNewVariable ident newName val
    code <- genStmt (Decl pos t items)
    return $ updcode ++ [QAss (QLoc newName (getOrigQType t)) val] ++ code
genStmt (Decl pos t ((NoInit _ item) : items)) = case t of
    (Int _) -> genStmt (Decl pos t (Init pos item (ELitInt pos 0) : items))
    (Str _) -> genStmt (Decl pos t (Init pos item (EString pos "") : items))
    (Bool _) -> genStmt (Decl pos t (Init pos item (ELitFalse pos) : items))
    (Class _ _) -> genStmt (Decl pos t (Init pos item (ELitInt pos 0) : items))
    _ -> error "unreachable"
genStmt (Ass pos1 (EVar pos2 (Ident ident)) e) = do
    tloc <- gets (Map.lookup ident . env)
    case tloc of
        Just loc -> do
            (val, exprCode) <- genExpr e
            (curLabel, _) <- gets (fromJust . Map.lookup loc . storeQ)
            let updCode = exprCode ++ [QAss (QLoc curLabel (getVType val)) val]
            when (isRawString val) $ addConstString (extractString val)
            qsNewVariable ident curLabel val
            return updCode
        Nothing -> genStmt (Ass pos1 (EClassAttr pos2 (EVar pos2 (Ident "self")) (Ident ident)) e)
genStmt (Ass _ (EClassAttr _ cExpr (Ident attr)) e) = do
    (val, exprCode) <- genExpr e
    (var, attrCode) <- genExpr cExpr
    let c = case getVType var of
            ClassQ c' -> c'
            _ -> error "referencing nullptr"
    cd <- gets (fromJust . Map.lookup c . defClass)
    let (offset, _) = fromJust $ Map.lookup attr (attributes cd)
    return $ exprCode ++ attrCode ++ [QStore var val offset]
genStmt (Ass{}) = error "unreachable"
genStmt (Incr pos i) = genStmt $ Ass pos (EVar pos i) (EAdd pos (EVar pos i) (Plus pos) (ELitInt pos 1))
genStmt (Decr pos i) = genStmt $ Ass pos (EVar pos i) (EAdd pos (EVar pos i) (Minus pos) (ELitInt pos 1))
genStmt (Ret _ e) = do
    (retVal, codeExpr) <- genExpr e
    if isRawString retVal
        then do
            addConstString (extractString retVal)
            return $ codeExpr ++ [QRet retVal]
        else
            return $ codeExpr ++ [QRet retVal]
genStmt (VRet _) = return [QVRet]
genStmt (Cond _ e s) = do
    case computeConst e of
        Just (QBool True) -> genStmt s
        Just (QBool False) -> return []
        _ -> do
            lt <- createTmpLabel
            lf <- createTmpLabel
            codeE <- genCond e lt lf
            codeS <- genStmt s
            return $ codeE ++ [QLabel lt] ++ codeS ++ [QJmp lf, QLabel lf]
genStmt (CondElse _ e s1 s2) = do
    case computeConst e of
        Just (QBool True) -> genStmt s1
        Just (QBool False) -> genStmt s2
        _ -> do
            lt <- createTmpLabel
            lf <- createTmpLabel
            le <- createTmpLabel
            codeS1 <- genStmt s1
            codeS2 <- genStmt s2
            codeE <- genCond e lt lf
            return $ codeE ++ [QLabel lt] ++ codeS1 ++ [QJmp le, QLabel lf] ++ codeS2 ++ [QLabel le]
genStmt (While _ e s) = do
    ls <- createTmpLabel
    lc <- createTmpLabel
    le <- createTmpLabel
    codeS <- genStmt s
    codeE <- genCond e ls le
    return $ [QJmp lc, QLabel ls] ++ codeS ++ [QJmp lc, QLabel lc] ++ codeE ++ [QLabel le]
genStmt (SExp _ e) = do
    (_, code) <- genExpr e
    return code

genExpr :: Expr -> QuadMonad (Address, [Quad])
genExpr (EVar pos (Ident ident)) = do
    tloc <- gets (Map.lookup ident . env)
    case tloc of
        Just loc -> do
            (curName, val) <- gets (fromJust . Map.lookup loc . storeQ)
            let val' = QLoc curName (getVType val)
            return (val', [])
        Nothing -> genExpr (EClassAttr pos (EVar pos (Ident "self")) (Ident ident))
genExpr expr@(ELitInt{}) = return (fromJust $ computeConst expr, [])
genExpr expr@(ELitTrue{}) = return (fromJust $ computeConst expr, [])
genExpr expr@(ELitFalse{}) = return (fromJust $ computeConst expr, [])
genExpr expr@(ECastNull{}) = return (fromJust $ computeConst expr, [])
genExpr (EClassAttr _ cExpr (Ident ident)) = do
    -- TODO: refactor
    resTmpName <- createTmpVar
    (val, code) <- genExpr cExpr
    let c = case getVType val of
            ClassQ c' -> c'
            _ -> error "dereferencing nullptr"
    detClass <- gets defClass
    let cd = fromJust $ Map.lookup c detClass
    let (offset, vt) = fromJust $ Map.lookup ident (attributes cd)
    let res = QLoc resTmpName vt
    let code' = code ++ [QLoad res val offset]
    return (res, code')
genExpr (EMethodCall _ cExpr (Ident method) exprList) = do
    -- TODO: refactor
    (val, _) <- genExpr cExpr
    let c = case getVType val of
            ClassQ c' -> c'
            _ -> error "dereferencing nullptr"
    detClass <- gets defClass
    let cd = fromJust $ Map.lookup c detClass
    let (offset, _) = fromJust $ Map.lookup method (methods cd)
    updCode <- do
        valsCodes <- mapM genExpr (cExpr : exprList)
        let paramGenCode = concatMap snd valsCodes
        let addParamsFromList [] qcode = return qcode
            addParamsFromList ((paramVal, _) : rest) qcode = case paramVal of
                (QString s) -> do
                    addConstString s
                    addParamsFromList rest (qcode ++ [QParam paramVal])
                _ -> addParamsFromList rest (qcode ++ [QParam paramVal])
        addParamsFromList valsCodes paramGenCode
    (_, fApp) <- gets (fromJust . Map.lookup method . methods . fromJust . Map.lookup c . defClass)
    let retType = returnType fApp
    newTmpName <- createTmpNamedVar $ c ++ method
    let code' = updCode ++ [QCallMethod (QLoc newTmpName retType) offset (length exprList + 1)]
    return (QLoc newTmpName retType, code')
genExpr (EFunctionCall _ (Ident ident) exprList) = do
    let callFuncParamOrLocal newTmpName retType updCode = do
            let val = QLoc newTmpName retType
            let code' = updCode ++ [QCall val ident (length exprList)]
            return (QLoc newTmpName retType, code')
    updCode <- do
        valsCodes <- mapM genExpr exprList
        let paramGenCode = concatMap snd valsCodes
        let addParamsFromList [] qcode = return qcode
            addParamsFromList ((paramVal, _) : rest) qcode = case paramVal of
                (QString s) -> do
                    addConstString s
                    addParamsFromList rest (qcode ++ [QParam paramVal])
                _ -> addParamsFromList rest (qcode ++ [QParam paramVal])
        addParamsFromList valsCodes paramGenCode
    if ident `elem` builtinFuncsList
        then do
            get >>= \cur -> unless (ident `elem` builtinFunc cur) (put cur{builtinFunc = ident : builtinFunc cur})
            newTmpName <- createTmpNamedVar ident
            let retType = getBuiltinRetType ident
            callFuncParamOrLocal newTmpName retType updCode
        else do
            fApp <- gets (fromJust . Map.lookup ident . defFunc)
            let retType = returnType fApp
            newTmpName <- createTmpNamedVar ident
            callFuncParamOrLocal newTmpName retType updCode
genExpr (EClassNew _ (Ident c)) = do
    newTmpName <- createTmpNamedVar c
    cd <- gets (fromJust . Map.lookup c . defClass)
    get >>= \cur -> unless ("__class_new" `elem` builtinFunc cur) (put cur{builtinFunc = "__class_new" : builtinFunc cur})
    let code = [QParam (QInt (attrSize cd)), QParam (QString ("_vtable_" ++ c)), QCall (QLoc newTmpName (ClassQ c)) "__class_new" 2]
    return (QLoc newTmpName (ClassQ c), code)
genExpr expr@(EString{}) = return (fromJust $ computeConst expr, [])
genExpr expr@(ENeg _ e) = case computeConst expr of
    Just v -> return (v, [])
    _ -> do
        resTmpName <- createTmpVar
        (val, code) <- genExpr e
        let var = QLoc resTmpName IntQ
        let code' = code ++ [QNeg var val]
        return (QLoc resTmpName IntQ, code')
genExpr expr@(EMul _ e1 op e2) = case computeConst expr of
    Just (QInt n) -> return (QInt n, [])
    _ -> do
        resTmpName <- createTmpVar
        depth1 <- getDepth e1
        depth2 <- getDepth e2
        (val1, code1, val2, code2) <- do
            if depth1 >= depth2
                then do
                    (val1', code1') <- genExpr e1
                    (val2', code2') <- genExpr e2
                    return (val1', code1', val2', code2')
                else do
                    (val2', code2') <- genExpr e2
                    (val1', code1') <- genExpr e1
                    return (val1', code1', val2', code2')
        let var = QLoc resTmpName IntQ
        let code' = case op of
                (Times _) -> code1 ++ code2 ++ [QBinOp var val1 AopMul val2]
                (Div _) -> code1 ++ code2 ++ [QBinOp var val1 AopDiv val2]
                (Mod _) -> code1 ++ code2 ++ [QBinOp var val1 AopMod val2]
        return (QLoc resTmpName IntQ, code')
genExpr expr@(EAdd _ e1 op e2) = case computeConst expr of
    Just (QInt n) -> return (QInt n, [])
    Just (QString s) -> return (QString s, [])
    _ -> do
        resTmpName <- createTmpVar
        depth1 <- getDepth e1
        depth2 <- getDepth e2
        (val1, code1, val2, code2) <- do
            if depth1 >= depth2
                then do
                    (val1', code1') <- genExpr e1
                    (val2', code2') <- genExpr e2
                    return (val1', code1', val2', code2')
                else do
                    (val2', code2') <- genExpr e2
                    (val1', code1') <- genExpr e1
                    return (val1', code1', val2', code2')
        case getVType val1 of
            IntQ -> do
                let var = QLoc resTmpName IntQ
                let code' = case op of
                        (Plus _) -> code1 ++ code2 ++ [QBinOp var val1 AopAdd val2]
                        (Minus _) -> code1 ++ code2 ++ [QBinOp var val1 AopSub val2]
                return (QLoc resTmpName IntQ, code')
            StringQ -> do
                let concVar = QLoc resTmpName StringQ
                let code' = code1 ++ code2 ++ [QParam val1, QParam val2, QCall concVar "__strcat" 2]
                get >>= \cur -> unless ("__strcat" `elem` builtinFunc cur) (put cur{builtinFunc = "__strcat" : builtinFunc cur})
                return (QLoc resTmpName StringQ, code')
            _ -> error "unreachable"
genExpr e = case computeConst e of
    Just (QBool b) -> return (QBool b, [])
    _ -> do
        resTmpName <- createTmpVar
        let res = QLoc resTmpName BoolQ
        lt <- createTmpLabel
        lf <- createTmpLabel
        le <- createTmpLabel
        code <- genCond e lt lf
        return (res, code ++ [QLabel lt, QAss res (QBool True), QJmp le, QLabel lf, QAss res (QBool False), QJmp le, QLabel le])

genCond :: Expr -> String -> String -> QuadMonad [Quad]
genCond (EVar _ (Ident ident)) lt lf = do
    loc <- gets (fromJust . Map.lookup ident . env)
    gets (Map.lookup loc . storeQ) >>= \case
        Just (_, val@(QLoc fname retType)) -> do
            let var = QLoc fname retType
            let code = [QCond var val (QBool True) CondAnd, QJmpCMP var lt lf]
            return code
        Just (_, QBool b) -> return [QJmp (if b then lt else lf)]
        _ -> error "unreachable"
genCond (ELitFalse _) _ lf = return [QJmp lf]
genCond (ELitTrue _) lt _ = return [QJmp lt]
genCond (EClassAttr{}) _ _ = error "unimplemented genCond EClassAttr"
genCond expr@(EMethodCall{}) lt lf = do
    (var@(QLoc{}), code) <- genExpr expr
    return $ code ++ [QCond var var (QBool True) CondAnd, QJmpCMP var lt lf]
genCond expr@(EFunctionCall{}) lt lf = do
    (val@(QLoc fname retType), code) <- genExpr expr
    let var = QLoc fname retType
    return $ code ++ [QCond var val (QBool True) CondAnd, QJmpCMP var lt lf]
genCond (ENot _ e) lt lf = genCond e lf lt
genCond expr@(ERel _ e1 op' e2) lt lf = case computeConst expr of
    Just (QBool b) -> return [if b then QJmp lt else QJmp lf]
    _ -> do
        resTmpName <- createTmpVar
        let rotateRelOp op = case op of
                (EQU p) -> EQU p
                (NE p) -> NE p
                (GE p) -> LE p
                (GTH p) -> LTH p
                (LE p) -> GE p
                (LTH p) -> GTH p
        depth1 <- getDepth e1
        depth2 <- getDepth e2
        (val1, code1, val2, code2, op) <- do
            if depth1 >= depth2
                then do
                    (val1', code1') <- genExpr e1
                    (val2', code2') <- genExpr e2
                    return (val1', code1', val2', code2', op')
                else do
                    (val2', code2') <- genExpr e2
                    (val1', code1') <- genExpr e1
                    return (val1', code1', val2', code2', rotateRelOp op')
        let varLoc = QLoc resTmpName BoolQ
        let condType = case op of
                (EQU _) -> CondEQ
                (NE _) -> CondNE
                (GE _) -> CondGE
                (GTH _) -> CondGT
                (LE _) -> CondLE
                (LTH _) -> CondLT
        return $ code1 ++ code2 ++ [QCond varLoc val1 val2 condType, QJmpCMP varLoc lt lf]
genCond expr@(EAnd _ e1 e2) lt lf = case computeConst expr of
    (Just (QBool b)) -> do return [if b then QJmp lt else QJmp lf]
    _ -> do
        lm <- createTmpLabel
        code1 <- genCond e1 lm lf
        code2 <- genCond e2 lt lf
        return $ code1 ++ [QLabel lm] ++ code2
genCond expr@(EOr _ e1 e2) lt lf = case computeConst expr of
    (Just (QBool b)) -> return [if b then QJmp lt else QJmp lf]
    _ -> do
        lm <- createTmpLabel
        code1 <- genCond e1 lt lm
        code2 <- genCond e2 lt lf
        return $ code1 ++ [QLabel lm] ++ code2
genCond _ _ _ = error "unreachable"

-- getDepth {{{
getDepth :: Expr -> QuadMonad Int
getDepth (EVar{}) = pure 1
getDepth (ELitInt{}) = pure 1
getDepth (ELitTrue{}) = pure 1
getDepth (ELitFalse{}) = pure 1
getDepth (EString{}) = pure 1
getDepth (ECastNull{}) = pure 1
getDepth expr@(EClassAttr _ e _) = getDepth1Expr expr e
getDepth (EMethodCall{}) = error "unimplemented EMethodCall"
getDepth (EFunctionCall{}) = pure 1
getDepth (EClassNew{}) = error "unimplemented EClassNew"
getDepth expr@(ENeg _ e) = getDepth1Expr expr e
getDepth (ENot _ e) = getDepth e
getDepth expr@(EMul _ e1 _ e2) = getDepth2Exprs expr e1 e2
getDepth expr@(EAdd _ e1 _ e2) = getDepth2Exprs expr e1 e2
getDepth expr@(ERel _ e1 _ e2) = getDepth2Exprs expr e1 e2
getDepth expr@(EAnd _ e1 e2) = getDepth2Exprs expr e1 e2
getDepth expr@(EOr _ e1 e2) = getDepth2Exprs expr e1 e2

getDepth1Expr expr e = do
    depths <- gets exprDepth
    case Map.lookup expr depths of
        Just d -> pure d
        Nothing -> do
            d <- getDepth e >>= \d -> return (d + 1)
            modify (\cur -> cur{exprDepth = Map.insert expr d (exprDepth cur)})
            return d

getDepth2Exprs expr e1 e2 = do
    depths <- gets exprDepth
    case Map.lookup expr depths of
        Just d -> pure d
        Nothing -> do
            d1 <- getDepth e1
            d2 <- getDepth e2
            let d = min (max d1 d2) (min d1 d2 + 1)
            modify (\cur -> cur{exprDepth = Map.insert expr d (exprDepth cur)})
            return d

-- }}}
-- computeConst {{{
computeConst :: Expr' a -> Maybe Address
computeConst (ELitInt _ n) = Just $ QInt $ fromInteger n
computeConst (ELitTrue _) = Just $ QBool True
computeConst (ELitFalse _) = Just $ QBool False
computeConst (EString _ s) = Just $ QString $ strMangle ++ s
computeConst (ECastNull _ _) = Just QNullptr
computeConst (ENeg _ e) = case computeConst e of
    Just (QInt n) -> Just $ QInt (-n)
    _ -> Nothing
computeConst (ENot _ e) = case computeConst e of
    Just (QBool b) -> Just $ QBool (not b)
    _ -> Nothing
computeConst (EMul _ e1 op e2) =
    let v1 = computeConst e1
        v2 = computeConst e2
     in case (v1, v2) of
            (Just (QInt n1), Just (QInt n2)) -> case op of
                (Times _) -> Just $ QInt (n1 * n2)
                (Div _) -> Just $ QInt (n1 `div` n2)
                (Mod _) -> Just $ if n1 >= 0 then QInt (n1 `mod` n2) else QInt (negate (negate n1 `mod` n2))
            _ -> Nothing
computeConst (EAdd _ e1 op e2) =
    let v1 = computeConst e1
        v2 = computeConst e2
     in case (v1, v2) of
            (Just (QInt n1), Just (QInt n2)) -> case op of
                (Plus _) -> Just $ QInt (n1 + n2)
                (Minus _) -> Just $ QInt (n1 - n2)
            (Just (QString s1), Just (QString s2)) -> case op of
                (Plus _) -> Just $ QString (s1 ++ drop (length strMangle) s2)
                _ -> error "unreachable"
            _ -> Nothing
computeConst (ERel _ e1 op e2) =
    let v1 = computeConst e1
        v2 = computeConst e2
     in case (v1, v2) of
            (Just (QInt n1), Just (QInt n2)) -> case op of
                (EQU _) -> Just $ QBool (n1 == n2)
                (NE _) -> Just $ QBool (n1 /= n2)
                (GE _) -> Just $ QBool (n1 >= n2)
                (GTH _) -> Just $ QBool (n1 > n2)
                (LE _) -> Just $ QBool (n1 <= n2)
                (LTH _) -> Just $ QBool (n1 < n2)
            (Just _, Just _) -> case op of
                (EQU _) -> Just $ QBool (v1 == v2)
                (NE _) -> Just $ QBool (v1 /= v2)
                _ -> Nothing
            _ -> Nothing
computeConst (EAnd _ e1 e2) =
    let v1 = computeConst e1
        v2 = computeConst e2
     in case (v1, v2) of
            (Just (QBool False), _) -> Just $ QBool False
            (Just (QBool b1), Just (QBool b2)) -> Just $ QBool (b1 && b2)
            _ -> Nothing
computeConst (EOr _ e1 e2) =
    let v1 = computeConst e1
        v2 = computeConst e2
     in case (v1, v2) of
            (Just (QBool True), _) -> Just $ QBool True
            (Just (QBool b1), Just (QBool b2)) -> Just $ QBool (b1 || b2)
            _ -> Nothing
computeConst _ = Nothing

-- }}}
-- fuction arguments {{{
saveArgsToEnv [] = gets env
saveArgsToEnv ((Arg _ argType (Ident ident)) : args) = do
    let valType = getOrigQType argType
    let val = QLoc ident valType
    qsNewLabel ident
    qsNewVariable ident ident val
    saveArgsToEnv args
getArgAddress (Arg _ (Int _) (Ident ident)) = QLoc ident IntQ
getArgAddress (Arg _ (Bool _) (Ident ident)) = QLoc ident BoolQ
getArgAddress (Arg _ (Str _) (Ident ident)) = QLoc ident StringQ
getArgAddress (Arg _ (Class _ (Ident c)) (Ident ident)) = QLoc ident (ClassQ c)
getArgAddress _ = error "unreachable"

-- }}}
-- tmp variables {{{
createTmp ident isVar = do
    let candName = (if isVar then ".t" else ".L" ++ ident)
    cntl <- gets (Map.lookup candName . countLabels)
    case cntl of
        Nothing -> do
            qsNewLabel candName
            return $ candName ++ "_0"
        Just numLabels -> do
            let newName = candName ++ "_" ++ show numLabels
            modify (\cur -> cur{countLabels = Map.insert candName (numLabels + 1) (countLabels cur)})
            return newName
createTmpLabel = gets curFuncName >>= \fname -> createTmp fname False
createTmpVar = createTmp "" True
createTmpNamedVar ident = createTmp ident True

-- }}}
-- helper functions {{{
qsNewVariable ident label val = gets lastLocQ >>= \loc -> modify (\cur -> cur{env = Map.insert ident loc (env cur), storeQ = Map.insert loc (label, val) (storeQ cur), lastLocQ = loc + 1})
qsNewFunction name funcInfo = get >>= \cur -> put cur{defFunc = Map.insert name funcInfo (defFunc cur)}
qsNewLabel ident = get >>= \cur -> put cur{countLabels = Map.insert ident 1 (countLabels cur)}
qsNewClass name classInfo = get >>= \cur -> put cur{defClass = Map.insert name classInfo (defClass cur)}

getBuiltinRetType fname =
    case fname of
        "printInt" -> VoidQ
        "readInt" -> IntQ
        "printString" -> VoidQ
        "readString" -> StringQ
        "error" -> VoidQ
        _ -> error "unreachable"

addConstString s = do
    fname <- gets curFuncName
    body <- gets (fromJust . Map.lookup fname . defFunc)
    let body' = body{stringVars = s : stringVars body}
    modify (\cur -> cur{defFunc = Map.insert fname body' (defFunc cur)})

-- }}}
