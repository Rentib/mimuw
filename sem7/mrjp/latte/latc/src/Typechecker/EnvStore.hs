module Typechecker.EnvStore where

import Data.Map as Map
import Parser.Abs

type Loc = Integer
data Env where Env :: {_env :: Map Ident Loc} -> Env
data Store a where Store :: {_store :: Map Loc a, _newloc :: Loc} -> Store a

envEmpty :: Env
envEmpty = Env Map.empty
envPut :: Ident -> Loc -> Env -> Env
envPut x l (Env e) = Env $ Map.insert x l e
envGet :: Ident -> Env -> Maybe Loc
envGet x (Env e) = Map.lookup x e

storeEmpty :: Store a
storeEmpty = Store Map.empty 0
storePut :: Loc -> a -> Store a -> Store a
storePut l v (Store s n) = Store (Map.insert l v s) n
storeGet :: Loc -> Store a -> Maybe a
storeGet l (Store s _) = Map.lookup l s
storeNewLoc :: Store a -> (Store a, Loc)
storeNewLoc (Store s l) = (Store s (l + 1), l)
