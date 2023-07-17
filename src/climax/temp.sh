#!/bin/bash
##SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH -J huangshijie
#SBATCH -t 72:00:00
#SBATCH -p kshdexclu01
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH -o log/test.o
#SBATCH -e log/test.e

# module use /public/software/modules
# module purge
# module load compiler/devtoolset/7.3.1
# module load mpi/hpcx/2.7.4/gcc-7.3.1
# module load compiler/rocm/dtk-22.04.2
# module load apps/PyTorch/1.10.0a0-build/torch1.10.0a0-dtk21.10-build
source /public/home/linse/.bashrc

src='/public/home/qindaotest/wym/Fourcastnet_change/FoueCastNet_160_160_history/FourCastNet_160_160/exp/afno_backbone/0/autoregressive_predictions_2m_temperature.h5'

v=2

python src/utils/space_plot.py $src $v

# python train_loss_rmse.py

