{-# OPTIONS_GHC -Wno-missing-signatures #-}

module Compiler (compile) where

import Asm.Asm
import Opt.Opt
import Parser.Par
import Quad.DataTypes
import Quad.Quad
import System.Exit
import System.IO
import Typechecker.Typechecker

compile :: FilePath -> IO ()
compile f = do
    s <- readFile f
    case pProgram (myLexer s) of
        (Left err) -> hPutStrLn stderr "ERROR" >> die err
        (Right tree) -> case typecheck tree of
            (Left err) -> hPutStrLn stderr "ERROR" >> die (show err)
            (Right _) -> do
                hPutStrLn stderr "SUCCESS"
                (eitherQuad, quadCode) <- genQuad tree
                case eitherQuad of
                    (Right (_, qstore)) -> do
                        (_, optimizedQuadCode) <- optimize quadCode
                        (_, asmCode) <- genAssembly (defClass qstore, builtinFunc qstore) optimizedQuadCode
                        mapM_ print asmCode
                        exitSuccess
                    (Left err) -> hPutStrLn stderr "ERROR" >> die (show err)
