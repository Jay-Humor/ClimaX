#!/bin/bash
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -J test
#SBATCH -t 72:00:00
#SBATCH -p kshdnormal
##SBATCH -p kshdtest
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --mem=110G
#SBATCH -o ./log/output_%j.log
#SBATCH -e ./log/error_%j.log
time=`date +%Y.%m.%d-%H:%M:%S`
mkdir ./log/${time}

module use /public/software/modules
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/dtk-22.04.2
module load apps/PyTorch/1.10.0a0-build/torch1.10.0a0-dtk21.10-build
# export PYTHONPATH=$PYTHONPATH:/public/home/qindaotest/rendq/pythonlib
rootdir=$(pwd)
cd $rootdir

# Multiple GPUs
NUM_GPUS_PER_NODE=4
NUM_NODES=1
NODE_RANK=0
SCRIPT_PATH="/public/home/qindaotest/huangshijie/ClimaX-Pytorch/train.py"

pretrained_path="/public/home/qindaotest/huangshijie/ClimaX-Pytorch/ClimaX-5.625deg.ckpt"
root_dir="/public/home/qindaotest/huangshijie/data-minimal"
batch_size=4

# Set OMP_NUM_THREADS environment variable
export OMP_NUM_THREADS=1

# Run the script with torchrun
python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  $SCRIPT_PATH \
  --seed_everything 42 \
  --lr 5e-7 \
  --beta_1 0.9 \
  --beta_2 0.99 \
  --weight_decay 1e-5 \
  --warmup_epochs 10000 \
  --max_epochs 100000 \
  --warmup_start_lr 1e-8 \
  --eta_min 1e-8 \
  --pretrained_path $pretrained_path \
  --root_dir $root_dir \
  --predict_range 72 \
  --hrs_each_step 1 \
  --buffer_size 10000 \
  --batch_size $batch_size \
  --num_workers 1 \
  --pin_memory False

mv ./log/output_$SLURM_JOB_ID.log ./log/${time}/output_${time}.log
mv ./log/error_$SLURM_JOB_ID.log ./log/${time}/error_${time}.log