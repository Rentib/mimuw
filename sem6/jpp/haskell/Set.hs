module Set(Set(..), empty, null, singleton, union, fromList
              , member, toList, toAscList, elems
              ) where
import Prelude hiding(null)
import Data.List (sort)

data Set a = Empty
           | Singleton a
           | Union (Set a) (Set a)

empty :: Set a
empty = Empty

null :: Set a -> Bool
null Empty = True
null _ = False

member :: Eq a => a -> Set a -> Bool
member _ Empty = False
member x (Singleton y) = x == y
member x (Union l r) = member x l || member x r

singleton :: a -> Set a
singleton = Singleton

fromList :: [a] -> Set a
fromList = foldr insert empty

toList :: Set a -> [a]
toList Empty = []
toList (Singleton x) = [x]
toList (Union l r) = toList l ++ toList r

toAscList :: Ord a => Set a -> [a]
toAscList s =
    removeDuplicatesFromSorted (sort (toList s))
  where
    removeDuplicatesFromSorted [] = []
    removeDuplicatesFromSorted (x : xs) = x : removeDuplicatesFromSorted (dropWhile (== x) xs)

elems :: Set a -> [a]
elems = toList

union :: Set a -> Set a -> Set a
union = Union

insert :: a -> Set a -> Set a
insert x = union (singleton x)

instance Ord a => Eq (Set a) where
    s == t = toAscList s == toAscList t

instance Semigroup (Set a) where
    (<>) = union

instance Monoid (Set a) where
    mempty = empty
    mappend = union

instance Show a => Show (Set a) where
    show s = "fromList " ++ show (toList s)

instance Functor Set where
    fmap _ Empty = Empty
    fmap f (Singleton x) = Singleton (f x)
    fmap f (Union l r) = Union (fmap f l) (fmap f r)
