module Opt.DataTypes where

import qualified Data.Map as Map
import Quad.DataTypes

data OptimizeStore where
    OptimizeStore ::
        { _curLabel :: Label
        , _curLabels :: [Label]
        , _blockEnv :: Map.Map Label BasicBlock
        , _varEnv :: Map.Map String Address
        , _varCnt :: Map.Map String Int
        , _allocated :: Map.Map String Bool
        , _allocatedBytes :: Int
        , _lcse :: Map.Map (Address, ArithmeticOp, Address) Address -- TODO: add LCSE for ConditionalOp
        , _localVarEnv :: Map.Map String Address
        , _quad :: [Quad]
        , _varRefs :: Map.Map Address Int
        } ->
        OptimizeStore

osAddRef :: Address -> OptimizeStore -> OptimizeStore
osAddRef addr os = os{_varRefs = Map.insertWith (+) addr 1 (_varRefs os)}

osDelRef :: Address -> OptimizeStore -> OptimizeStore
osDelRef addr os = os{_varRefs = Map.update (\n -> if n == 1 then Nothing else Just (n - 1)) addr (_varRefs os)}

osNewQuad :: Quad -> OptimizeStore -> OptimizeStore
osNewQuad q os = os{_quad = _quad os ++ [q]}

osEmpty :: OptimizeStore
osEmpty =
    OptimizeStore
        { _curLabel = ""
        , _curLabels = []
        , _blockEnv = Map.empty
        , _varEnv = Map.empty
        , _varCnt = Map.empty
        , _allocated = Map.empty
        , _allocatedBytes = 0
        , _lcse = Map.empty
        , _localVarEnv = Map.empty
        , _quad = []
        , _varRefs = Map.empty
        }

osNewBlock :: Label -> OptimizeStore -> OptimizeStore
osNewBlock l os = os{_curLabel = l, _curLabels = _curLabels os ++ [l], _blockEnv = Map.insert l block (_blockEnv os), _lcse = Map.empty, _localVarEnv = Map.empty}
  where
    block' = blockEmpty l
    block = block'{_blockVars = _varEnv os}

osNewAddr :: Address -> OptimizeStore -> OptimizeStore
osNewAddr (QLoc x t) os = os{_varEnv = env, _varCnt = cnt}
  where
    env = Map.insert x (QLoc (x ++ show n) t) (_varEnv os)
    cnt = Map.insert x (n + 1) (_varCnt os)
    n = Map.findWithDefault 0 x (_varCnt os)
osNewAddr _ _ = error "unreachable"

osSetAddr :: Address -> Address -> OptimizeStore -> OptimizeStore
osSetAddr (QLoc x _) addr os = os{_localVarEnv = Map.insert x addr (_localVarEnv os)}
osSetAddr _ _ _ = error "unreachable"

osGetLValue :: Address -> OptimizeStore -> Address
osGetLValue (QLoc x t) os = Map.findWithDefault (QLoc x t) x (_varEnv os)
osGetLValue addr _ = addr

osGetRValue :: Address -> OptimizeStore -> Address
osGetRValue (QLoc x t) os = case Map.lookup x (_localVarEnv os) of
    Just addr -> addr
    Nothing -> Map.findWithDefault (QLoc x t) x (_varEnv os)
osGetRValue addr _ = addr

osAddBlockQuad :: Quad -> OptimizeStore -> OptimizeStore
osAddBlockQuad q os = os{_blockEnv = Map.adjust (\b -> b{_blockQuad = _blockQuad b ++ [q]}) (_curLabel os) (_blockEnv os)}

osSetBlockTerminator :: Quad -> OptimizeStore -> OptimizeStore
osSetBlockTerminator q os = os{_blockEnv = Map.adjust (\b -> b{_blockVarEnv = _varEnv os, _terminator = Just q}) (_curLabel os) (_blockEnv os)}

osGetLCSE :: (Address, ArithmeticOp, Address) -> OptimizeStore -> Maybe Address
osGetLCSE (a1, op, a2) os = Map.lookup (a1, op, a2) (_lcse os)

osNewLCSE :: (Address, ArithmeticOp, Address) -> Address -> OptimizeStore -> OptimizeStore
osNewLCSE (a1, op, a2) a os = os{_lcse = Map.insert (a1, op, a2) a (_lcse os)}

data BasicBlock where
    BasicBlock ::
        { _blockLabel :: Label
        , _blockQuad :: [Quad]
        , _blockPred :: [Label]
        , _blockVars :: Map.Map String Address -- initial variables
        , _blockVarEnv :: Map.Map String Address -- final variables
        , _terminator :: Maybe Quad
        } ->
        BasicBlock

blockEmpty :: Label -> BasicBlock
blockEmpty l = BasicBlock l [] [] Map.empty Map.empty Nothing

data Phi = Phi
    { _phiAddr :: Address
    , _phiId :: Int
    , _phiType :: VType
    , _phiOperands :: [(Label, Address)]
    }
