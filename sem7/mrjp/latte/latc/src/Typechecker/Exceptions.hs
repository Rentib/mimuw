module Typechecker.Exceptions where

import Parser.Abs

type LCException = LCException' BNFC'Position
data LCException' a where
    UnimplementedLCException :: a -> LCException' a
    UndefinedReferenceLCException :: a -> Ident -> LCException' a
    WrongTypeLCException :: a -> String -> String -> LCException' a
    RedeclaredLCException :: a -> String -> Ident -> LCException' a
    FunctionWithoutReturnLCException :: a -> Ident -> LCException' a
    WrongReturnTypeLCException :: a -> String -> String -> LCException' a
    UninitializedVariableLCException :: a -> Ident -> LCException' a
    UndeclaredVariableLCException :: a -> Ident -> LCException' a
    NotAFunctionLCException :: a -> Ident -> LCException' a
    WrongNumberOfArgumentsLCException :: a -> Ident -> Int -> Int -> LCException' a
    WrongArgumentTypeLCException :: a -> Ident -> String -> String -> LCException' a
    OperatorUndefinedLCException :: a -> String -> String -> LCException' a
    LvalueRequiredLCException :: a -> String -> LCException' a
    NotAClassLCException :: a -> Ident -> LCException' a

instance Show LCException where
    show (UnimplementedLCException pos) = "Unimplemented at " ++ showpos pos
    show (UndefinedReferenceLCException pos (Ident x)) = "Undefined reference to " ++ x ++ " at " ++ showpos pos
    show (WrongTypeLCException pos expected got) = "Wrong type at " ++ showpos pos ++ ", expected " ++ expected ++ ", got " ++ got
    show (RedeclaredLCException pos s (Ident x)) = "Redeclared " ++ s ++ " " ++ x ++ " at " ++ showpos pos
    show (FunctionWithoutReturnLCException pos (Ident f)) = "Non-void function " ++ f ++ " does not return at " ++ showpos pos
    show (WrongReturnTypeLCException pos expected got) = "Wrong return type at " ++ showpos pos ++ ", expected " ++ expected ++ ", got " ++ got
    show (UninitializedVariableLCException pos (Ident x)) = "Uninitialized variable " ++ x ++ " at " ++ showpos pos
    show (UndeclaredVariableLCException pos (Ident x)) = "Undeclared variable " ++ x ++ " at " ++ showpos pos
    show (NotAFunctionLCException pos (Ident x)) = x ++ " is not a function at " ++ showpos pos
    show (WrongNumberOfArgumentsLCException pos (Ident f) expected got) = "Wrong number of arguments for function " ++ f ++ " at " ++ showpos pos ++ ", expected " ++ show expected ++ ", got " ++ show got
    show (WrongArgumentTypeLCException pos (Ident f) expected got) = "Wrong argument type for function " ++ f ++ " at " ++ showpos pos ++ ", expected " ++ expected ++ ", got " ++ got
    show (OperatorUndefinedLCException pos op t) = "Operator " ++ op ++ " undefined for type " ++ t ++ ", at " ++ showpos pos
    show (LvalueRequiredLCException pos s) = "Lvalue required as " ++ s ++ " at " ++ showpos pos
    show (NotAClassLCException pos (Ident c)) = c ++ " is not a class at " ++ showpos pos

showpos :: BNFC'Position -> String
showpos (Just (l, c)) = "line " ++ show l ++ ", column " ++ show c
showpos Nothing = "unknown position"
