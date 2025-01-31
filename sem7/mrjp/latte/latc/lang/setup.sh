#!/bin/sh

set -e

rm -f Makefile
rm -rf Latte
bnfc -m --functor --haskell -d Latte.cf
sed -i '10s/^$/GHC_OPTS = -package array/' Makefile
make -j
mkdir -p Parser/
cp -f Latte/*.hs Parser/
rm Parser/Test.hs
sed -i 's/Latte/Parser/g' Parser/*.hs
rm -rf ../src/Parser
mv Parser ../src
rm -f Makefile
rm -rf Latte
