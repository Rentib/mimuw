#!/bin/sh

good="./examples/good/"
bad="./examples/bad/"

make

n_good_ok=0
n_good_wa=0
n_bad_ok=0
n_bad_wa=0

for file in "$good"*.lat; do
    echo "Testing $file"
    bin="$(basename "$file")"
    bin="${bin%.lat}"
    if ! ./latc_x86_64 "$file" >/dev/null; then
        n_good_wa=$((n_good_wa + 1))
        printf "\033[0;31mFAIL\033[0m\n"
    else
        input="${file%.lat}.input"
        if [ -f "$input" ]; then
            ./"$bin" <"$input" >output
        else
            ./"$bin" >output
        fi
        if diff "${file%.lat}.output" output >/dev/null 2>&1; then
            n_good_ok=$((n_good_ok + 1))
            printf "\033[0;32mOK\033[0m\n"
        else
            n_good_wa=$((n_good_wa + 1))
            printf "\033[0;31mFAIL\033[0m\n"
        fi
    fi
    rm -f "$bin.s" "$bin.o" "$bin"
done
rm -f output

for file in "$bad"*.lat; do
    echo "Testing $file"
    bin="$(basename "$file")"
    bin="${bin%.lat}"
    if ! ./latc_x86_64 "$file" >/dev/null; then
        printf "\033[0;32mOK\033[0m\n"
        n_bad_ok=$((n_bad_ok + 1))
    else
        printf "\033[0;31mFAIL\033[0m\n"
        n_bad_ok=$((n_bad_wa + 1))
    fi
    rm -f "$bin.s" "$bin.o" "$bin"
done

echo "Good tests: $n_good_ok OK, $n_good_wa WA"
echo "Bad tests: $n_bad_ok OK, $n_bad_wa WA"
