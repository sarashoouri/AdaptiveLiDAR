#!/usr/bin/env bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=fusion
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-gpu=4
#SBATCH --nodes=1
#SBATCH --time=10-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48000m

#SBATCH --output=./Codes/logs/dist_train_CMT.log

echo "Hello, world!"


PARTITION=$1
JOB_NAME=$fusion
CONFIG="./Codes/CMT/projects/configs/fusion/cmt_voxel0100_r50_800x320_cbgs.py"
WORK_DIR=$4
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --launcher="slurm"
