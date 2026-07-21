#!/bin/bash
#SBATCH --job-name=patient-heterogeneity-ranges
#SBATCH --output=/home/dsanortiz/logs/%x_%j.out
#SBATCH --error=/home/dsanortiz/logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

set -e

mkdir -p /home/dsanortiz/logs

source /opt/miniconda/etc/profile.d/conda.sh
conda activate patient-heterogeneity

cd /home/dsanortiz

echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

echo "Checking GPU with nvidia-smi..."
nvidia-smi || true

echo "Checking TensorFlow GPU visibility..."
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

python /home/dsanortiz/patient-heterogeneity-balancing/training_pipeline/3-train.py

echo "Job finished at: $(date)"

conda deactivate