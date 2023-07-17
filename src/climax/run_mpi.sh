#!/bin/bash -l
##SBATCH -n 32
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH -J huangshijie
#SBATCH -t 240:00:90
#SBATCH -p kshdexclu01
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --mem=110G
#SBATCH -o ./log_shared/output-%j.log
#SBATCH -e ./log_shared/error-%j.log

# Create a log directory for the job
time=`date +%Y.%m.%d-%H.%M.%S-$SLURM_JOB_ID`
mkdir ./log_shared/${time}

# Load modules
module use /public/software/modules
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/dtk-22.10.1
module load apps/PyTorch/1.10.0a0-build/torch1.10.0a0-dtk21.10-build
# source /public/home/linse/.bashrc

# Multiple GPUs
NUM_GPUS_PER_NODE=4

#CUDA_VISIBLE_DEVICES=0,1,2,3
# Set the script path and the pretrained model path
SCRIPT_PATH="./src/train.py"
pretrained_path="$(pwd)/ckpts/ClimaX-5.625deg.ckpt"

# Set the root directory of the data
root_dir="/public/home/dqren/raindata/AIR/data/climax-data"

# Set the batch size
batch_size=4

# Set OMP_NUM_THREADS environment variable
export OMP_NUM_THREADS=1

export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(hostname)

# Run the script
#APP="python -m torch.distributed.launch  \
#  --nproc_per_node=$NUM_GPUS_PER_NODE \
APP="python $SCRIPT_PATH \
  --seed_everything 42 \
  --lr 5e-7 \
  --beta_1 0.9 \
  --beta_2 0.99 \
  --weight_decay 1e-5 \
  --warmup_epochs 20 \
  --max_epochs 200 \
  --warmup_start_lr 1e-8 \
  --eta_min 1e-8 \
  --pretrained_path $pretrained_path \
  --root_dir $root_dir \
  --log_dir $(pwd)/log/$time \
  --predict_range 120 \
  --hrs_each_step 1 \
  --buffer_size 10000 \
  --batch_size $batch_size \
  --num_workers 1 \
  --pin_memory False"

set -x
srun -u --mpi=pmix_v3 \
    bash -c "
    source export_DDP_vars.sh
    ${APP} 
    "

# Move the log files to the log directory
mv $(pwd)/log/output-$SLURM_JOB_ID.log $(pwd)/log/${time}/output-${time}.log
mv $(pwd)/log/error-$SLURM_JOB_ID.log $(pwd)/log/${time}/error-${time}.log

# Remove the __pycache__ directory
rm -rf $(pwd)/src/__pycache__
rm -rf $(pwd)/src/utils/__pycache__
