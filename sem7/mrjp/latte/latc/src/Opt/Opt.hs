{-# OPTIONS_GHC -Wno-unused-matches #-}

module Opt.Opt (optimize) where

import Control.Monad.Except
import Control.Monad.State
import Control.Monad.Writer
import qualified Data.Map as Map
import Opt.DataTypes
import Quad.DataTypes

type OptimizeM a = StateT OptimizeStore (ExceptT String (WriterT QuadCode IO)) a

optimize :: QuadCode -> IO (Either String OptimizeStore, QuadCode)
optimize qc = runWriterT $ runExceptT $ evalStateT (runOpt qc) osEmpty

runOpt :: QuadCode -> OptimizeM OptimizeStore
runOpt qc = do
    mapM_ optimize' qc
    get >>= \s -> return s

-- stack size {{{
allocate :: Address -> OptimizeM ()
allocate (QLoc var vt) = do
    allocated <- gets _allocated
    case Map.lookup var allocated of
        Just _ -> pure ()
        Nothing -> do
            allocatedBytes <- gets _allocatedBytes
            modify (\s -> s{_allocated = Map.insert var True allocated, _allocatedBytes = allocatedBytes + vTypeSize vt})
allocate _ = pure ()

calculateStackSize :: Quad -> OptimizeM ()
calculateStackSize (QAss res _) = allocate res
calculateStackSize (QBinOp res _ _ _) = allocate res
calculateStackSize (QCall res _ _) = allocate res
calculateStackSize (QCond res _ _ _) = allocate res
calculateStackSize (QFunc fd@(FuncData f _ args b _ _)) = do
    modify (\s -> s{_allocated = Map.empty, _allocatedBytes = 0})
    mapM_ calculateStackSize b
    allocatedBytes <- gets _allocatedBytes
    mapM_ allocate args
    tell [QFunc fd{stackSize = allocatedBytes + 8}] -- TODO: this +8 is for self in methods, but it should be done nicely and not for every function
calculateStackSize (QJmp{}) = pure ()
calculateStackSize (QJmpCMP{}) = pure ()
calculateStackSize (QLabel l) = pure ()
calculateStackSize (QNeg res _) = allocate res
calculateStackSize (QParam{}) = pure ()
calculateStackSize (QRet res) = allocate res
calculateStackSize QVRet = pure ()
calculateStackSize (QLoad res _ _) = allocate res
calculateStackSize (QStore{}) = pure ()
calculateStackSize (QCallMethod res _ _) = allocate res

-- }}}
-- Count Variable References {{{
countVarRefs :: Quad -> OptimizeM ()
countVarRefs (QAss var val) = modify $ osAddRef val
countVarRefs (QBinOp var lhs op rhs) = do
    modify $ osAddRef lhs
    modify $ osAddRef rhs
countVarRefs (QCond var lhs rhs op) = do
    modify $ osAddRef lhs
    modify $ osAddRef rhs
countVarRefs (QJmpCMP val lt lf) = modify $ osAddRef val
countVarRefs (QLoad var addr offset) = modify $ osAddRef addr
countVarRefs (QNeg var val) = modify $ osAddRef val
countVarRefs (QParam val) = modify $ osAddRef val
countVarRefs (QRet val) = modify $ osAddRef val
countVarRefs (QStore addr val offset) = do
    modify $ osAddRef addr
    modify $ osAddRef val
countVarRefs _ = pure ()

-- }}}

optimize' :: Quad -> OptimizeM ()
optimize' (QFunc (FuncData funcName returnType arguments body stringVars stackSize)) = do
    put osEmpty
    modify $ osNewBlock funcName
    mapM_ (modify . osNewAddr) arguments
    arguments' <- mapM (gets . osGetRValue) arguments
    mapM_ optimize' body
    labels <- gets _curLabels
    blocks <- gets _blockEnv
    blocks' <- forM (map (blocks Map.!) labels) addTerminator
    let body' = concatMap _blockQuad blocks'
    mapM_ countVarRefs body'

    mapM_ removeDeadCode body'
    let removeDeadCode' = do
            quad <- gets _quad
            modify (\s -> s{_quad = []})
            mapM_ removeDeadCode quad
            quad' <- gets _quad
            unless (length quad == length quad') removeDeadCode'
    removeDeadCode'
    body'' <- gets _quad
    calculateStackSize $ QFunc (FuncData funcName returnType arguments' body'' stringVars stackSize)
optimize' (QAss var val) = do
    val' <- gets $ osGetRValue val
    modify $ osNewAddr var
    var' <- gets $ osGetLValue var
    modify $ osAddBlockQuad $ QAss var' val'
    modify $ osSetAddr var val'
optimize' (QBinOp var lhs op rhs) = do
    lhs' <- gets $ osGetRValue lhs
    rhs' <- gets $ osGetRValue rhs
    lcse <- gets $ osGetLCSE (lhs', op, rhs')
    case (lhs', rhs') of
        (QInt n, QInt m) -> optimize' $ QAss var $ QInt $ case op of
            AopAdd -> n + m
            AopSub -> n - m
            AopMul -> n * m
            AopDiv -> n `div` m
            AopMod -> n `mod` m
        _ -> do
            case lcse of
                Just addr -> optimize' $ QAss var addr
                Nothing -> do
                    modify $ osNewAddr var
                    var' <- gets $ osGetLValue var
                    modify $ osAddBlockQuad $ QBinOp var' lhs' op rhs'
                    modify $ osNewLCSE (lhs', op, rhs') var'
optimize' (QCall var fname nargs) = do
    modify $ osNewAddr var
    var' <- gets $ osGetLValue var
    modify $ osAddBlockQuad $ QCall var' fname nargs
optimize' (QCallMethod var offset nargs) = do
    modify $ osNewAddr var
    var' <- gets $ osGetLValue var
    modify $ osAddBlockQuad $ QCallMethod var' offset nargs
optimize' (QCond var lhs rhs op) = do
    lhs' <- gets $ osGetRValue lhs
    rhs' <- gets $ osGetRValue rhs
    modify $ osNewAddr var
    var' <- gets $ osGetLValue var
    modify $ osAddBlockQuad $ QCond var' lhs' rhs' op
optimize' (QJmpCMP val lt lf) = do
    val' <- gets $ osGetRValue val
    modify $ osSetBlockTerminator $ QJmpCMP val' lt lf
optimize' (QJmp l) = modify $ osSetBlockTerminator $ QJmp l
optimize' (QLabel l) = do
    modify $ osNewBlock l
    modify $ osAddBlockQuad $ QLabel l
optimize' (QLoad var addr offset) = do
    addr' <- gets $ osGetRValue addr
    modify $ osNewAddr var
    var' <- gets $ osGetLValue var
    modify $ osAddBlockQuad $ QLoad var' addr' offset
optimize' (QNeg var val) = do
    val' <- gets $ osGetRValue val
    modify $ osNewAddr var
    var' <- gets $ osGetLValue var
    modify $ osAddBlockQuad $ QNeg var' val'
optimize' (QParam val) = do
    val' <- gets $ osGetRValue val
    modify $ osAddBlockQuad $ QParam val'
optimize' (QRet val) = do
    val' <- gets $ osGetRValue val
    modify $ osAddBlockQuad $ QRet val'
optimize' (QStore addr val offset) = do
    addr' <- gets $ osGetRValue addr
    val' <- gets $ osGetRValue val
    modify $ osAddBlockQuad $ QStore addr' val' offset
optimize' QVRet = modify $ osAddBlockQuad QVRet

addTerminator :: BasicBlock -> OptimizeM BasicBlock
addTerminator block = addTerminator' block (_terminator block)

addTerminator' :: BasicBlock -> Maybe Quad -> OptimizeM BasicBlock
addTerminator' block Nothing = pure block
addTerminator' thisBlock (Just (QJmp l)) = do
    blockEnv <- gets _blockEnv
    let thatBlock = blockEnv Map.! l
    let thisFinalVars = _blockVarEnv thisBlock
    let thatInitialVars = _blockVars thatBlock
    let intersection = Map.intersectionWith (,) thisFinalVars thatInitialVars
    let relevant = Map.filter (uncurry (/=)) intersection
    let phiQuads = map (\(var, (thisAddr, thatAddr)) -> QAss thatAddr thisAddr) $ Map.toList relevant
    let l' = _blockLabel thisBlock ++ "_" ++ drop 2 l
    let quads = if null phiQuads then [QJmp l] else [QLabel l'] ++ phiQuads ++ [QJmp l]
    return thisBlock{_blockQuad = _blockQuad thisBlock ++ quads}
addTerminator' thisBlock (Just (QJmpCMP val lt lf)) = do
    blockEnv <- gets _blockEnv
    let thatBlockT = blockEnv Map.! lt
    let thatBlockF = blockEnv Map.! lf
    let thisFinalVars = _blockVarEnv thisBlock
    let thatInitialVarsT = _blockVars thatBlockT
    let thatInitialVarsF = _blockVars thatBlockF
    let intersectionT = Map.intersectionWith (,) thisFinalVars thatInitialVarsT
    let intersectionF = Map.intersectionWith (,) thisFinalVars thatInitialVarsF
    let relevantT = Map.filter (uncurry (/=)) intersectionT
    let relevantF = Map.filter (uncurry (/=)) intersectionF
    let phiQuadsT = map (\(var, (thisAddr, thatAddr)) -> QAss thatAddr thisAddr) $ Map.toList relevantT
    let phiQuadsF = map (\(var, (thisAddr, thatAddr)) -> QAss thatAddr thisAddr) $ Map.toList relevantF
    let lt' = _blockLabel thisBlock ++ "_" ++ drop 2 lt
    let lf' = _blockLabel thisBlock ++ "_" ++ drop 2 lf
    let quads = case (null phiQuadsT, null phiQuadsF) of
            (True, True) -> [QJmpCMP val lt lf]
            (False, True) -> [QJmpCMP val lt' lf] ++ [QLabel lt'] ++ phiQuadsT ++ [QJmp lt]
            (True, False) -> [QJmpCMP val lt lf'] ++ [QLabel lf'] ++ phiQuadsF ++ [QJmp lf]
            (False, False) -> [QJmpCMP val lt' lf'] ++ [QLabel lt'] ++ phiQuadsT ++ [QJmp lt] ++ [QLabel lf'] ++ phiQuadsF ++ [QJmp lf]
    return thisBlock{_blockQuad = _blockQuad thisBlock ++ quads}
addTerminator' _ _ = error "unreachable"

removeDeadCode :: Quad -> OptimizeM ()
removeDeadCode q@(QAss var val) = do
    refs <- gets _varRefs
    if Map.member var refs
        then modify $ osNewQuad q
        else do modify $ osDelRef val
removeDeadCode q@(QBinOp var lhs op rhs) = do
    refs <- gets _varRefs
    if Map.member var refs
        then modify $ osNewQuad q
        else do
            modify $ osDelRef lhs
            modify $ osDelRef rhs
removeDeadCode q@(QCond var lhs rhs op) = do
    refs <- gets _varRefs
    if Map.member var refs
        then modify $ osNewQuad q
        else do
            modify $ osDelRef lhs
            modify $ osDelRef rhs
removeDeadCode q@(QLoad var addr offset) = do
    refs <- gets _varRefs
    if Map.member var refs
        then modify $ osNewQuad q
        else modify $ osDelRef addr
removeDeadCode q@(QNeg var val) = do
    refs <- gets _varRefs
    if Map.member var refs
        then modify $ osNewQuad q
        else modify $ osDelRef val
removeDeadCode q = modify $ osNewQuad q
