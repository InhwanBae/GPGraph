#!/bin/bash
echo "Start training task queues"

# Hyperparameters
dataset_array=("eth" "hotel" "univ" "zara1" "zara2")
device_id_array=(0 1 2 3 4)
tag="GPGraph-SGCN"

# Arguments
while getopts t:d:i: flag
do
  case "${flag}" in
    t) tag=${OPTARG};;
    d) dataset_array=(${OPTARG});;
    i) device_id_array=(${OPTARG});;
    *) echo "usage: $0 [-t TAG] [-d \"eth hotel univ zara1 zara2\"] [-i \"0 1 2 3 4\"]" >&2
      exit 1 ;;
  esac
done

if [ ${#dataset_array[@]} -ne ${#device_id_array[@]} ]
then
    printf "Arrays must all be same length. "
    printf "len(dataset_array)=${#dataset_array[@]} and len(device_id_array)=${#device_id_array[@]}\n"
    exit 1
fi

# Signal handler
pid_array=()

sighdl ()
{
  echo "Kill training processes"
  for (( i=0; i<${#dataset_array[@]}; i++ ))
  do
    kill ${pid_array[$i]}
  done
  echo "Done."
  exit 0
}

trap sighdl SIGINT SIGTERM

# Start training tasks
for (( i=0; i<${#dataset_array[@]}; i++ ))
do
  printf "Training ${dataset_array[$i]}"
  CUDA_VISIBLE_DEVICES=${device_id_array[$i]} python3 train.py --gpu_num ${device_id_array[$i]} --lr 0.01 \
  --dataset "${dataset_array[$i]}" --tag "${tag}" --use_lrschd --num_epochs 300 &
  pid_array[$i]=$!
  printf " job ${#pid_array[@]} pid ${pid_array[$i]}\n"
done

for (( i=0; i<${#dataset_array[@]}; i++ ))
do
  wait ${pid_array[$i]}
done
echo "Training end."

# Start test
python3 test.py

echo "Done."