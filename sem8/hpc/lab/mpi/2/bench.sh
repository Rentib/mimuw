#!/bin/bash

echo "hosts,n,time"

hs=( 1 2 3 4 5 )
ns=( 10 50 100 250 500 750 1000 2000 3000 4000 5000 )

for hosts in "${hs[@]}"; do
    for n in "${ns[@]}"; do
        time=$(mpiexec -n $hosts -host orange02,orange04,orange05,orange06,orange10 ./floyd-warshall-par.exe $n 2>&1 | grep -o -E "[0-9]+\.[0-9]+" | tr '\n' ';' | python3 -c "x = input().strip().split(';')[:-1]; xd = sum(map(float, x)) / len(x); print(xd)")
        echo "$hosts,$n,$time"
    done
done
