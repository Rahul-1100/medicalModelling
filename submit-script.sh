#!/bin/bash
#SBATCH --job-name="ICD10_custom"
#SBATCH --partition=ampere
#SBATCH --time=32:00:00
#SBATCH --signal=B:USR1@600            # SIGUSR1 to the batch step 10 min before the time limit -> graceful checkpoint & exit
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=rahul.gupta@stud.uni-hannover.de
#SBATCH --output=slurm/%j-%x.out
#SBATCH --error=slurm/%j-%x.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# 1. Properly activate the virtual environment
source .venv/bin/activate

# 2. Prevent tokenizer deadlocks with DataLoader workers
export TOKENIZERS_PARALLELISM=false

# Optional: Set W&B to offline if compute nodes lack internet access, 
# or ensure WANDB_API_KEY is exported in your environment.
# export WANDB_MODE=offline

# 3. Build the unified tokenizer if it doesn't exist yet
if [ ! -f ./unified_icd10_gemma_tokenizer/tokenizer.json ]; then
    echo "Tokenizer not found. Building unified ICD-10 tokenizer..."
    python build_hierarchical_icd10_vocab.py --codes_source synth
fi

# 4. Run training
python train_unified_tokenizer.py \
    --data_source synth \
    --unified_tokenizer_dir ./unified_icd10_gemma_tokenizer \
    --output_dir ./slurm_out \
    --response_mode icd10_custom \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --learning_rate 2e-4 \
    --max_text_length 2048 \
    --save_steps 5000 \
    --max_runtime_minutes 1860
