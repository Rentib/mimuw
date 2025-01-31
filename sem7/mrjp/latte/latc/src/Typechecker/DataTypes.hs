module Typechecker.DataTypes where

import qualified Data.Map as Map
import Data.Maybe
import Parser.Abs

import Typechecker.EnvStore

data TCType where
    TCInt :: TCType
    TCBool :: TCType
    TCString :: TCType
    TCVoid :: TCType
    TCClass :: Ident -> TCType
    TCFun :: [TCType] -> TCType -> TCType
    deriving (Eq)

instance Show TCType where
    show TCInt = "int"
    show TCBool = "bool"
    show TCString = "string"
    show TCVoid = "void"
    show (TCClass (Ident s)) = "class " ++ s
    show (TCFun args ret) = "(" ++ unwords (map show args) ++ ") -> " ++ show ret

fromType :: Type -> TCType
fromType (Int _) = TCInt
fromType (Bool _) = TCBool
fromType (Str _) = TCString
fromType (Void _) = TCVoid
fromType (Class _ c) = TCClass c
fromType (Fun _ ret args) = TCFun (map fromType args) (fromType ret)

fromFunction :: [Arg] -> Type -> TCType
fromFunction args ret = TCFun (map fromArg args) (fromType ret)

fromArg :: Arg -> TCType
fromArg (Arg _ t _) = fromType t

dropQualifier :: TCType -> TCType
dropQualifier t = t

data TCState where
    TSInitialized :: TCState
    TSUninitialized :: TCState
    deriving (Show, Eq)

type TCValue = (TCType, TCState)

data ClassData where
    ClassData ::
        { attrs :: [(Ident, TCType)]
        , methods :: [(Ident, TCType)]
        , inheritance :: Maybe Ident
        } ->
        ClassData
    deriving (Show)

data TypecheckerState where
    TypecheckerState ::
        { env :: Env
        , classes :: Map.Map Ident ClassData
        , store :: Store TCValue
        , _retType :: TCType
        , _hasReturn :: Bool
        , _blockLoc :: Loc
        , _class :: Maybe Ident
        } ->
        TypecheckerState

tsEmpty :: TypecheckerState
tsEmpty = TypecheckerState envEmpty Map.empty storeEmpty TCVoid False 0 Nothing

tsNew :: Ident -> TCValue -> TypecheckerState -> TypecheckerState
tsNew x v TypecheckerState{..} = TypecheckerState{env = env', store = store', ..}
  where
    (store'', l) = storeNewLoc store
    env' = envPut x l env
    store' = storePut l v store''

tsNewClass :: Ident -> Maybe Ident -> [ClassElem] -> TypecheckerState -> TypecheckerState
tsNewClass c super elems TypecheckerState{..} = TypecheckerState{env = env, classes = classes', store = store, ..}
  where
    classes' = Map.insert c (ClassData attrs methods super) classes
    attrs = [(x, fromType t) | ClassAttrDef _ t items <- elems, ClassItem _ x <- items]
    methods = [(x, fromFunction args t) | ClassMethodDef _ t x args _ <- elems]

tsUpdate :: Ident -> TCValue -> TypecheckerState -> TypecheckerState
tsUpdate x v TypecheckerState{..} = TypecheckerState{env = env, store = store', ..}
  where
    l = fromJust $ envGet x env
    store' = storePut l v store

tsGet :: Ident -> TypecheckerState -> Maybe TCValue
tsGet x TypecheckerState{..} = case envGet x env of
    Just l -> storeGet l store
    Nothing -> Nothing

tsSetBlockLoc :: TypecheckerState -> TypecheckerState
tsSetBlockLoc ts@TypecheckerState{..} = ts{_blockLoc = _newloc store}

tsGetBlockLoc :: TypecheckerState -> Loc
tsGetBlockLoc TypecheckerState{..} = _blockLoc

tsGetLoc :: Ident -> TypecheckerState -> Maybe Loc
tsGetLoc x TypecheckerState{..} = envGet x env

tsGetClass :: Ident -> TypecheckerState -> Maybe ClassData
tsGetClass c TypecheckerState{..} = Map.lookup c classes

tsGetAttr :: Ident -> Ident -> TypecheckerState -> Maybe TCType
tsGetAttr c a TypecheckerState{..} = case Map.lookup c classes of
    Just cd@(ClassData attrs _ _) ->
        case lookup a attrs of
            Just t -> Just t
            Nothing -> case inheritance cd of
                Just i -> tsGetAttr i a TypecheckerState{..}
                Nothing -> Nothing
    Nothing -> Nothing

tsGetMethod :: Ident -> Ident -> TypecheckerState -> Maybe TCType
tsGetMethod c m TypecheckerState{..} = case Map.lookup c classes of
    Just cd@(ClassData _ methods _) ->
        case lookup m methods of
            Just t -> Just t
            Nothing -> case inheritance cd of
                Just i -> tsGetMethod i m TypecheckerState{..}
                Nothing -> Nothing
    Nothing -> Nothing

tsGetInheritance :: Ident -> TypecheckerState -> Maybe Ident
tsGetInheritance c TypecheckerState{..} = case Map.lookup c classes of
    Just (ClassData _ _ i) -> i
    Nothing -> Nothing
