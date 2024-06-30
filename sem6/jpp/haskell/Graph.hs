module Graph where
import Set(Set)
import qualified Set as Set
import Control.Monad (ap)
import Data.List (sort)
class Graph g where
  empty   :: g a
  vertex  :: a -> g a
  union   :: g a -> g a -> g a
  connect :: g a -> g a -> g a

data Relation a = Relation { domain :: Set a, relation :: Set (a, a) }
    deriving (Eq)

instance Show a => Show (Relation a) where
    show (Relation v e) = "Relation {domain = " ++ show (Set.toList v)
                        ++ ", relation = " ++ show (Set.toList e) ++ "}"

data Basic a = Empty
             | Vertex a
             | Union (Basic a) (Basic a)
             | Connect (Basic a) (Basic a)

instance Graph Relation where
    empty =
        Relation Set.empty Set.empty
    vertex x =
        Relation (Set.singleton x) Set.empty
    union (Relation v1 e1) (Relation v2 e2) =
        Relation (Set.union v1 v2) (Set.union e1 e2)
    connect (Relation v1 e1) (Relation v2 e2) =
        Relation (Set.union v1 v2) (Set.union (Set.union e1 e2) e3)
      where
        e3 = Set.fromList [(v, u) | v <- Set.toList v1, u <- Set.toList v2]

sortRmDup l =
    removeDuplicatesFromSorted (sort l)
  where
    removeDuplicatesFromSorted [] = []
    removeDuplicatesFromSorted (x : xs) = x : removeDuplicatesFromSorted (dropWhile (== x) xs)
                
instance (Ord a, Num a) => Num (Relation a) where
    fromInteger = vertex . fromInteger
    (+) (Relation v1 e1) (Relation v2 e2) =
        Relation
            (Set.fromList (sortRmDup (Set.toList v1 ++ Set.toList v2)))
            (Set.fromList (sortRmDup (Set.toList e1 ++ Set.toList e2)))
    (*) (Relation v1 e1) (Relation v2 e2) =
        Relation
            (Set.fromList (sortRmDup (Set.toList v1 ++ Set.toList v2)))
            (Set.fromList (sortRmDup (Set.toList e1 ++ Set.toList e2 ++ e3)))
          where
            e3 = [(v, u) | v <- Set.toList v1, u <- Set.toList v2]
    signum = const empty
    abs = id
    negate = id

instance Graph Basic where
    empty = Empty
    vertex = Vertex
    union = Union
    connect = Connect

instance Ord a => Eq (Basic a) where
    (==) a b =
        let Relation v1 e1 = fromBasic a
            Relation v2 e2 = fromBasic b
         in v1 == v2 && e1 == e2

instance (Ord a, Num a) => Num (Basic a) where
    fromInteger = vertex . fromInteger
    (+)         = union
    (*)         = connect
    signum      = const empty
    abs         = id
    negate      = id

instance Semigroup (Basic a) where
  (<>) = union

instance Monoid (Basic a) where
  mempty = Empty

fromBasic :: Graph g => Basic a -> g a
fromBasic Empty = Graph.empty
fromBasic (Vertex v) = Graph.vertex v
fromBasic (Union a b) = Graph.union (fromBasic a) (fromBasic b)
fromBasic (Connect a b) = Graph.connect (fromBasic a) (fromBasic b)

notElemSorted :: (Ord a) => [a] -> [a] -> [a] -> [a]
notElemSorted acc [] _ = acc
notElemSorted acc x [] = x ++ acc
notElemSorted acc (x : xs) (y : ys)
    | x == y = notElemSorted acc xs ys
    | x < y = notElemSorted (x : acc) xs (y : ys)
    | x > y = notElemSorted acc (x : xs) ys

instance (Ord a, Show a) => Show (Basic a) where
    show b =
        let Relation v e = fromBasic b in
        let sv = Set.toAscList v in
        let se = Set.toAscList e in
        let ev = Set.toAscList (Set.fromList (foldr (\x acc -> fst x : snd x : acc) [] se)) in
        let sv' = notElemSorted [] sv ev in
        "edges " ++ show se ++ " + vertices " ++ show sv'

-- | Example graph
-- >>> example34
-- edges [(1,2),(2,3),(2,4),(3,5),(4,5)] + vertices [17]

example34 :: Basic Int
example34 = 1*2 + 2*(3+4) + (3+4)*5 + 17

todot :: (Ord a, Show a) => Basic a -> String
todot b =
    let Relation v e = fromBasic b in
    let (sv, se) = (Set.toAscList v, Set.toAscList e) in
    let ev = Set.toAscList (Set.fromList (foldr (\x acc -> fst x : snd x : acc) [] se)) in
    let sv' = notElemSorted [] sv ev in
    "digraph {\n"
    ++ foldr (\x acc -> show (fst x) ++ " -> " ++ show (snd x) ++ ";\n" ++ acc) "" se
    ++ foldr (\x acc -> show x ++ ";\n" ++ acc) "" sv'
    ++ "}"

instance Functor Basic where
    fmap _ Empty = Empty
    fmap f (Vertex a) = Vertex (f a)
    fmap f (Union l r) = Union (fmap f l) (fmap f r)
    fmap f (Connect l r) = Connect (fmap f l) (fmap f r)

-- | Merge vertices
-- >>> mergeV 3 4 34 example34
-- edges [(1,2),(2,34),(34,5)] + vertices [17]

mergeV :: Eq a => a -> a -> a -> Basic a -> Basic a
mergeV u v w = fmap (\x -> if x == u || x == v then w else x)

instance Applicative Basic where
    pure = Vertex
    (<*>) = Control.Monad.ap

instance Monad Basic where
    (>>=) Empty _ = Empty
    (>>=) (Vertex x) f = f x
    (>>=) (Union l r) f = Union (l >>= f) (r >>= f)
    (>>=) (Connect l r) f = Connect (l >>= f) (r >>= f)

-- | Split Vertex
-- >>> splitV 34 3 4 (mergeV 3 4 34 example34)
-- edges [(1,2),(2,3),(2,4),(3,5),(4,5)] + vertices [17]

splitV :: Eq a => a -> a -> a -> Basic a -> Basic a
splitV w v u = (>>= f)
  where
    f x = if x == w then Union (Vertex v) (Vertex u) else Vertex x
