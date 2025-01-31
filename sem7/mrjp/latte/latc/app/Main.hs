module Main where

import System.Environment
import System.Exit

import qualified Compiler

main :: IO ()
main =
    getArgs >>= \case
        ["-h"] -> usage exitSuccess
        [f] -> Compiler.compile f
        _ -> usage exitFailure

usage :: IO () -> IO ()
usage exitCode = do
    getProgName >>= \p -> putStrLn $ "Usage: " ++ p ++ " <file>"
    exitCode
