#!/bin/bash

echo "hosts,n,time"

hs=( 1 2 3 5 8 10 )
ns=( 40 50 75 100 150 200 300 )

for hosts in "${hs[@]}"; do
    for n in "${ns[@]}"; do
       time=$(mpiexec -n $hosts ./laplace-par.exe $n 2>&1 | tail -n1 | cut -d' ' -f2 | cut -d= -f2)
        echo "$hosts,$n,$time"
    done
done
