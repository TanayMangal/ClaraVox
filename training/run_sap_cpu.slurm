#!/bin/bash
# SLURM job for the SAP Whisper fine-tune on a CPU-only node.
# Fill in the [BRACKETED] parts for your cluster (ask whoever runs it for the
# partition name and the longest time limit you're allowed), then:
#     sbatch training/run_sap_cpu.slurm
# If the job hits the time limit, just submit it again: --resume continues from the
# last checkpoint (one is saved every 200 steps).

#SBATCH --job-name=whisper-sap
#SBATCH --partition=[PARTITION]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=[NUMBER_OF_CORES]   # as many as the node has
#SBATCH --mem=128G                          # plenty; the script streams audio from disk
#SBATCH --time=[MAX_TIME e.g. 2-00:00:00]
#SBATCH --output=sap_work/logs/%x-%j.out

set -euo pipefail
mkdir -p sap_work/logs

# conda/venv with: pip install -r training/requirements-sap.txt
source [PATH_TO_ENV]/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$PWD/hf_cache                # keep model downloads inside the project

# Leave a few cores for the data loader processes.
THREADS=$(( SLURM_CPUS_PER_TASK > 8 ? SLURM_CPUS_PER_TASK - 4 : SLURM_CPUS_PER_TASK ))

python training/13_sap_finetune_whisper.py \
    --base-model ./whisper-torgo \
    --method lora --train-encoder-from 6 \
    --epochs 1 --batch 8 --grad-accum 4 \
    --threads $THREADS --workers 4 \
    --bf16 off \
    --resume
