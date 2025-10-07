#!/bin/bash

if [ $# -eq 0 ]; then
    >&2 echo -e "Usage: run_batch.sh <DIR> <BIN> <TYPE>\n\n<BIN> will run inside <DIR> \nIf <TYPE> provided and <TYPE> == nsys, <BIN> will be profiled using nsight-systems"
    exit 1
fi


DIR=$(pwd)/$1
BIN=$2
TYPE=$3

#FILE=$DIR/$BIN.cu
#if [ ! -f "$FILE" ]; then
#    echo "$FILE does not exist"
#    exit 1
#fi


BIN_CMD=$DIR/$BIN
NSYS=/home/adinar/nvidia/nsys
NCU=/home/adinar/nvidia/ncu
RES=$DIR/results

case "${TYPE}" in
  nsys)
  PROF_OUT="${DIR}/report.nsys-rep"
  echo "Will write report to ${PROF_OUT}"
  CMD="${NSYS} profile --trace=cuda -o ${PROF_OUT} --force-overwrite true ${BIN_CMD}"
  ;;
  ncu)
  CMD="${NCU} --export profiler_output --force-overwrite --set full ${BIN_CMD}"
  ;;
  *)
  CMD="${BIN_CMD}"
  ;;
esac

rm -f $RES

EOF_PHRASE="DONE_EOF"

echo "Will run ${BIN} in dir ${DIR} using command '${CMD}'."

function run_sbatch {
sbatch <<BATCHEOF
#!/bin/bash
#
#SBATCH --job-name=$BIN
#SBATCH --partition=hpc
#SBATCH --gres=gpu:1
#SBATCH --output=$RES
#SBATCH --time=00:05:00

cd $DIR
make

$CMD

echo '----------'
echo DONE_EOF

BATCHEOF

  until [ -f $RES ]; do
      echo "Waiting for job to run..."
      sleep 1
  done

  echo -e 'Results:'
  echo '----------'

  # Will wait for the whole output of the script
  sh -c "tail -n +0 --pid=$$ -f ${RES} | { sed '/DONE_EOF/ q' && kill $$ ;}"
}

function run_directly {
  cd $DIR
  make

  $CMD
}

if command -v sbatch &> /dev/null
then
  run_sbatch
else
  echo "sbatch not found, running directly"
  run_directly
fi
