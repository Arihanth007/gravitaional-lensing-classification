#!/bin/bash
#SBATCH -G 1
#SBATCH -p long
#SBATCH -J bppred_train
#SBATCH -A research
#SBATCH --constraint=2080ti
#SBATCH -n 10
#SBATCH -N 1
#SBATCH --time=4-00:00:00

source /home2/arihanth.srikar/anaconda3/bin/activate ml
echo $CONDA_PREFIX
export PYTHONUNBUFFERED=1

mkdir -p /scratch/arihanth.srikar
mkdir -p /scratch/arihanth.srikar/models

cp dataset.zip /scratch/arihanth.srikar
unzip /scratch/arihanth.srikar/dataset.zip -d /scratch/arihanth.srikar/
mkdir -p /scratch/arihanth.srikar/dataset/models

python3 CustomDS.py