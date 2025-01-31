module Typechecker.Typechecker (typecheck) where

import Control.Monad.Except
import Control.Monad.State

import Parser.Abs

import Typechecker.DataTypes
import Typechecker.Exceptions

builtins :: [(Ident, TCType)]
builtins =
    [ (Ident "printInt", TCFun [TCInt] TCVoid)
    , (Ident "printString", TCFun [TCString] TCVoid)
    , (Ident "readInt", TCFun [] TCInt)
    , (Ident "readString", TCFun [] TCString)
    , (Ident "error", TCFun [] TCVoid)
    ]

typecheck :: Program -> Either LCException ()
typecheck tu = void $ runExcept $ evalStateT (tcheck tu) tsEmpty

type TypecheckM = TypecheckM' TCType
type TypecheckM' a = StateT TypecheckerState (Except LCException) a
class Typechecker a where
    tcheck :: a -> TypecheckM

opStr :: RelOp' a -> String
opStr (LTH _) = "<"
opStr (LE _) = "<="
opStr (GTH _) = ">"
opStr (GE _) = ">="
opStr (EQU _) = "=="
opStr (NE _) = "!="

instance Typechecker Program where
    tcheck (Program pos topdefs) = do
        mapM_ (\(x, t) -> modify $ tsNew x (t, TSInitialized)) builtins
        let addTopDef :: TopDef -> TypecheckM
            addTopDef (FnDef pos' ret f args _) = do
                checkRedeclared pos' "function" f >> do
                    modify $ tsNew f (fromFunction args ret, TSInitialized)
                    pure TCVoid
            addTopDef (ClassDef pos' c elems) = do
                checkRedeclared pos' "class" c >> do
                    modify $ tsNew c (TCVoid, TSInitialized)
                    modify $ tsNewClass c Nothing elems
                    pure TCVoid
            addTopDef (ClassExt pos' c super elems) = do
                checkRedeclared pos' "class" c >> do
                    modify $ tsNew c (TCVoid, TSInitialized)
                    modify $ tsNewClass c (Just super) elems
                    pure TCVoid
        mapM_ addTopDef topdefs
        mapM_ tcheck topdefs >> get >>= \ts -> case tsGet (Ident "main") ts of
            Just (TCFun [] TCInt, TSInitialized) -> pure TCInt
            Just (TCFun [] t, TSInitialized) -> throwError $ WrongTypeLCException pos (show TCInt) (show t)
            Just (TCFun args _, _) -> throwError $ WrongNumberOfArgumentsLCException pos (Ident "main") (length args) 0
            _ | otherwise -> throwError $ UndefinedReferenceLCException pos (Ident "main")

inherits :: TCType -> TCType -> TypecheckM' Bool
inherits (TCClass c2) (TCClass c1) = do
    ts <- get
    case tsGetClass c1 ts of
        Just (ClassData _ _ (Just c)) -> if c == c2 then pure True else inherits (TCClass c2) (TCClass c)
        _ -> pure False
inherits _ _ = pure False
isSameType :: BNFC'Position -> Expr -> TCType -> TypecheckM
isSameType pos e t = do
    et <- tcheck e
    isInheritance <- t `inherits` et
    unless (et == t || isInheritance) (throwError $ WrongTypeLCException pos (show t) (show et)) >> pure t

checkRedeclared :: BNFC'Position -> String -> Ident -> TypecheckM
checkRedeclared pos s x = do
    ts <- get
    case tsGetLoc x ts of
        Just l -> do
            if x == Ident "" || l < _blockLoc ts
                then pure TCVoid
                else throwError $ RedeclaredLCException pos s x
        Nothing -> pure TCVoid

instance Typechecker TopDef where
    -- NOTE: all topdefs are added to environment beforehand
    tcheck (FnDef pos ret f args block) = do
        ts <- get
        put $ ts{_retType = fromType ret, _hasReturn = False}
        let addArg (Arg _ t x) = tsNew x (fromType t, TSInitialized)
        mapM_ (modify . addArg) args
        void $ tcheck block
        ts' <- get
        unless (_hasReturn ts' || fromType ret == TCVoid) $ throwError $ FunctionWithoutReturnLCException pos f
        put ts
        pure TCVoid
    tcheck (ClassDef _ c elems) = do
        ts <- get
        put $ ts{_class = Just c}
        let addElem (ClassAttrDef _ t xs) = do
                let addItem :: ClassItem -> TypecheckM
                    addItem (ClassItem pos x) = checkRedeclared pos "attribute" x >> modify (tsNew x (fromType t, TSInitialized)) >> pure TCVoid
                mapM_ addItem xs
            addElem (ClassMethodDef{}) = pure ()
        modify $ tsNew (Ident "self") (TCClass c, TSInitialized)
        mapM_ addElem elems
        mapM_ tcheck elems
        put ts
        pure TCVoid
    tcheck (ClassExt pos _c ce _elems) = do
        ts <- get
        void $ case tsGetClass ce ts of
            Just _ -> pure ()
            Nothing -> throwError $ NotAClassLCException pos ce
        -- TODO: implement, this works for tests, but not in general
        return TCVoid
instance Typechecker ClassElem where
    tcheck (ClassAttrDef{}) = pure TCVoid
    tcheck (ClassMethodDef pos t m args b) = tcheck $ FnDef pos t m args b

instance Typechecker Block where
    tcheck (Block _ stmts) = do
        ts <- get
        modify tsSetBlockLoc
        mapM_ tcheck stmts
        ts' <- get
        put ts'{env = env ts, _blockLoc = _blockLoc ts}
        pure TCVoid

instance Typechecker Stmt where
    tcheck (Empty _) = pure TCVoid
    tcheck (BStmt _ block) = tcheck block
    tcheck (Decl _ t xs) = do
        let addItem :: Item -> TypecheckM
            addItem (NoInit pos x) = checkRedeclared pos "variable" x >> modify (tsNew x (fromType t, TSInitialized)) >> pure TCVoid
            addItem (Init pos x e) = checkRedeclared pos "variable" x >> isSameType pos e (fromType t) >> modify (tsNew x (fromType t, TSInitialized)) >> pure TCVoid
        mapM_ addItem xs >> pure TCVoid
    tcheck (Ass pos lv e) = ensureLvalue lv >> tcheck lv >>= isSameType pos e
    tcheck (Incr pos x) = isSameType pos (EVar pos x) TCInt
    tcheck (Decr pos x) = isSameType pos (EVar pos x) TCInt
    tcheck (Ret pos e) = do
        t <- gets _retType
        et <- tcheck e
        isInheritance <- t `inherits` et
        unless (et == t || isInheritance) $ throwError $ WrongReturnTypeLCException pos (show t) (show et)
        modify (\ts -> ts{_hasReturn = True}) >> pure TCVoid
    tcheck (VRet pos) = do
        ts <- get
        unless (TCVoid == _retType ts) $ throwError $ WrongReturnTypeLCException pos (show $ _retType ts) (show TCVoid)
        put (ts{_hasReturn = True}) >> pure TCVoid
    tcheck (Cond pos e s) =
        isSameType pos e TCBool >> do
            hasReturn <- gets _hasReturn
            void $ tcheck (Block pos [s])
            checkCondReturn (eval e) hasReturn
    tcheck (CondElse pos e s1 s2) =
        -- TODO: refactor
        isSameType pos e TCBool >> do
            let e' = eval e
            hasReturn <- gets _hasReturn
            void $ tcheck (Block pos [s1])
            hasReturn1 <- gets _hasReturn
            ts' <- get
            put $ ts'{_hasReturn = hasReturn}
            void $ tcheck (Block pos [s2])
            hasReturn2 <- gets _hasReturn
            ts <- get
            case e' of
                VBool True -> put $ ts{_hasReturn = hasReturn1}
                VBool False -> put $ ts{_hasReturn = hasReturn2}
                _ -> put $ ts{_hasReturn = hasReturn || (hasReturn1 && hasReturn2)}
            pure TCVoid
    tcheck (While pos e s) =
        isSameType pos e TCBool >> do
            hasReturn <- gets _hasReturn
            void $ tcheck (Block pos [s])
            checkCondReturn (eval e) hasReturn
    tcheck (SExp _ e) = tcheck e

instance Typechecker Expr where
    tcheck (EVar pos x) = do
        ts <- get
        case tsGet x ts of
            Just (t, TSInitialized) -> pure $ dropQualifier t
            Just (_, TSUninitialized) -> throwError $ UninitializedVariableLCException pos x
            Nothing -> throwError $ UndeclaredVariableLCException pos x
    tcheck (ELitInt _ _) = pure TCInt
    tcheck (ELitTrue _) = pure TCBool
    tcheck (ELitFalse _) = pure TCBool
    tcheck (EString _ _) = pure TCString
    tcheck (ECastNull _ t) = pure $ fromType t
    tcheck (EClassAttr pos e i) = do
        t <- tcheck e
        case t of
            TCClass c -> do
                ts <- get
                case tsGetAttr c i ts of
                    Just t' -> pure t'
                    Nothing -> throwError $ UndefinedReferenceLCException pos i
            _ -> throwError $ NotAClassLCException pos (Ident (show e)) -- TODO: this is a very ugly message
    tcheck (EMethodCall pos c m args) = do
        t <- tcheck c
        case t of
            TCClass c' -> do
                ts <- get
                case tsGetMethod c' m ts of
                    Just (TCFun params ret) -> do
                        let (nargs, nparams) = (length args, length params)
                        unless (nargs == nparams) $ throwError $ WrongNumberOfArgumentsLCException pos m nparams nargs
                        let okArg :: (TCType, TCType, Expr) -> Bool
                            okArg (t1, t2, _) = dropQualifier t1 == dropQualifier t2
                        argTypes <- mapM tcheck args
                        unless (all okArg $ zip3 params argTypes args) $ throwError (WrongArgumentTypeLCException pos m (show params) (show argTypes))
                        pure ret
                    Just _ -> throwError $ NotAFunctionLCException pos m
                    Nothing -> throwError $ UndefinedReferenceLCException pos m
            _ -> throwError $ NotAClassLCException pos (Ident (show c))
    tcheck (EFunctionCall pos f args) = do
        ts <- get
        case tsGet f ts of
            Just (TCFun params ret, _) -> do
                let (nargs, nparams) = (length args, length params)
                unless (nargs == nparams) $ throwError $ WrongNumberOfArgumentsLCException pos f nparams nargs
                let okArg :: (TCType, TCType, Expr) -> Bool
                    okArg (t1, t2, _) = dropQualifier t1 == dropQualifier t2
                argTypes <- mapM tcheck args
                unless (all okArg $ zip3 params argTypes args) $ throwError (WrongArgumentTypeLCException pos f (show params) (show argTypes))
                pure ret
            Just (_, _) -> throwError $ NotAFunctionLCException pos f
            Nothing -> throwError $ UndefinedReferenceLCException pos f
    tcheck (EClassNew pos c) = do
        ts <- get
        case tsGetClass c ts of
            Just _ -> pure $ TCClass c
            _ -> throwError $ NotAClassLCException pos c
    tcheck (ENeg pos e) = isSameType pos e TCInt
    tcheck (ENot pos e) = isSameType pos e TCBool
    tcheck (EMul pos e1 _ e2) = isSameType pos e1 TCInt >> isSameType pos e2 TCInt
    tcheck (EAdd pos e1 (Plus _) e2) = do
        t <- tcheck e1
        -- if t1 != String and t1 != Int
        unless (t == TCInt || t == TCString) $ throwError $ OperatorUndefinedLCException pos "+" (show t)
        isSameType pos e2 t
    tcheck (EAdd pos e1 _ e2) = isSameType pos e1 TCInt >> isSameType pos e2 TCInt
    tcheck (ERel pos e1 op e2) = do
        t1 <- tcheck e1
        t2 <- tcheck e2
        unless (t1 == t2) $ throwError $ WrongTypeLCException pos (show t1) (show t2)
        _ <- case t1 of
            TCFun{} -> throwError $ OperatorUndefinedLCException pos (opStr op) (show t1)
            TCVoid -> throwError $ OperatorUndefinedLCException pos (opStr op) (show t1)
            _ -> pure ()
        case op of
            (EQU _) -> pure TCBool
            (NE _) -> pure TCBool
            _ -> do
                unless (t1 == TCInt) $ throwError $ OperatorUndefinedLCException pos (opStr op) (show t1)
                pure TCBool
    tcheck (EAnd pos e1 e2) = isSameType pos e1 TCBool >> isSameType pos e2 TCBool
    tcheck (EOr pos e1 e2) = isSameType pos e1 TCBool >> isSameType pos e2 TCBool

checkCondReturn :: Value -> Bool -> TypecheckM
checkCondReturn (VBool True) _ = pure TCVoid
checkCondReturn _ old = do
    ts <- get
    put $ ts{_hasReturn = old}
    pure TCVoid

-- TODO: refactor
data Value where
    VInt :: Integer -> Value
    VBool :: Bool -> Value
    VString :: String -> Value
    VUnknown :: Value

eval :: Expr -> Value
eval (EVar{}) = VUnknown
eval (ELitInt _ n) = VInt n
eval (ELitTrue _) = VBool True
eval (ELitFalse _) = VBool False
eval (EString _ s) = VString s
eval (ECastNull{}) = VUnknown
eval (EClassAttr{}) = VUnknown
eval (EMethodCall{}) = VUnknown
eval (EFunctionCall{}) = VUnknown
eval (EClassNew{}) = VUnknown
eval (ENeg _ e) =
    case eval e of
        VInt n -> VInt (-n)
        _ -> VUnknown
eval (ENot _ e) =
    case eval e of
        VBool b -> VBool (not b)
        _ -> VUnknown
eval (EMul _ e1 op e2) =
    case (eval e1, eval e2) of
        (VInt n1, VInt n2) -> VInt $ case op of
            Times _ -> n1 * n2
            Div _ -> n1 `div` n2
            Mod _ -> n1 `mod` n2
        _ -> VUnknown
eval (EAdd _ e1 op e2) =
    case (eval e1, eval e2) of
        (VInt n1, VInt n2) -> VInt $ case op of
            Plus _ -> n1 + n2
            Minus _ -> n1 - n2
        (VString s1, VString s2) -> VString $ case op of
            Plus _ -> s1 ++ s2
            _ -> error "impossible"
        _ -> VUnknown
eval (ERel _ e1 (EQU _) e2) =
    case (eval e1, eval e2) of
        (VInt n1, VInt n2) -> VBool $ n1 == n2
        (VString s1, VString s2) -> VBool $ s1 == s2
        _ -> VUnknown
eval (ERel _ e1 (NE _) e2) =
    case (eval e1, eval e2) of
        (VInt n1, VInt n2) -> VBool $ n1 /= n2
        (VString s1, VString s2) -> VBool $ s1 /= s2
        _ -> VUnknown
eval (ERel _ e1 op e2) =
    case (eval e1, eval e2) of
        (VInt n1, VInt n2) -> VBool $ case op of
            LTH _ -> n1 < n2
            LE _ -> n1 <= n2
            GTH _ -> n1 > n2
            GE _ -> n1 >= n2
        _ -> VUnknown
eval (EAnd _ e1 e2) =
    case (eval e1, eval e2) of
        (VBool b1, VBool b2) -> VBool $ b1 && b2
        _ -> VUnknown
eval (EOr _ e1 e2) =
    case (eval e1, eval e2) of
        (VBool b1, VBool b2) -> VBool $ b1 || b2
        _ -> VUnknown

ensureLvalue :: Expr -> TypecheckM' ()
ensureLvalue (EVar pos x) = do
    ts <- get
    case tsGet x ts of
        Just (_, TSInitialized) -> pure ()
        Just (_, TSUninitialized) -> throwError $ UninitializedVariableLCException pos x
        Nothing -> throwError $ UndeclaredVariableLCException pos x
ensureLvalue (ELitInt pos n) = throwError $ LvalueRequiredLCException pos ("integer literal" ++ show n)
ensureLvalue (ELitTrue pos) = throwError $ LvalueRequiredLCException pos "true"
ensureLvalue (ELitFalse pos) = throwError $ LvalueRequiredLCException pos "false"
ensureLvalue (EString pos s) = throwError $ LvalueRequiredLCException pos ("string literal" ++ s)
ensureLvalue (ECastNull pos _) = throwError $ LvalueRequiredLCException pos "null"
ensureLvalue (EClassAttr _ e _) = ensureLvalue e
ensureLvalue (EMethodCall pos _ m _) = throwError $ LvalueRequiredLCException pos ("method call" ++ show m)
ensureLvalue (EFunctionCall pos f _) = throwError $ LvalueRequiredLCException pos ("function call" ++ show f)
ensureLvalue (EClassNew pos x) = throwError $ LvalueRequiredLCException pos ("new class" ++ show x)
ensureLvalue (ENeg pos e) = throwError $ LvalueRequiredLCException pos ("negation of" ++ show e)
ensureLvalue (ENot pos e) = throwError $ LvalueRequiredLCException pos ("negation of" ++ show e)
ensureLvalue (EMul pos e1 op e2) = throwError $ LvalueRequiredLCException pos ("multiplication of" ++ show e1 ++ show op ++ show e2)
ensureLvalue (EAdd pos e1 op e2) = throwError $ LvalueRequiredLCException pos ("addition of" ++ show e1 ++ show op ++ show e2)
ensureLvalue (ERel pos e1 op e2) = throwError $ LvalueRequiredLCException pos ("comparison of" ++ show e1 ++ show op ++ show e2)
ensureLvalue (EAnd pos e1 e2) = throwError $ LvalueRequiredLCException pos ("and of" ++ show e1 ++ show e2)
ensureLvalue (EOr pos e1 e2) = throwError $ LvalueRequiredLCException pos ("or of" ++ show e1 ++ show e2)
