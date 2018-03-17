#!/bin/bash
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks 6
#SBATCH --nodes 3
#SBATCH --partition gpu
#SBATCH --gres gpu:p100:2
#SBATCH --output /home/users/sturgeod/distributed_cnn/distributed.out
#SBATCH --error /home/users/sturgeod/distributed_cnn/distributed.err



DIR=/home/users/sturgeod/distributed_cnn

source /home/exacloud/lustre1/fnl_lab/code/external/venvs/miniconda3/bin/activate
$DIR/distributed.py --maxiter 1000

