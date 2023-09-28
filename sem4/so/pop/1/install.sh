#!/bin/sh

die() {
    echo "$@"
    exit 1
}

[[ -f $1 ]] || die "Usage: $0 <file>"
[[ $(wc -l < $1) -eq 7 ]] || die "$1 does not have 7 lines"

content=$(cat $1)
content="string:	db \`${content//$'\n'/$'\\\\r\\\\n'}\\\\r\\\\n\`, 0\n"
awk -v content="$content" 'NR==209 {printf "%s", content; next} 1' bl.s | cat -v > bl.s.tmp

mv bl.s.tmp bl.s
yes | pkgin install nasm
nasm -f bin -o bl.bin bl.s
dd bs=2048 count=1 if=bl.bin of=/dev/c0d0
