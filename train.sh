#!/bin/bash

#SBATCH --job-name="ICD10_custom"              # the name you set for your job
#SBATCH --partition=ampere              # the partition your job runs on. You might not have access to all partitions, just test it out. Not all GPUs are accessible on all paritions, check `si`
#SBATCH --time=96:00:00                 # the maximum runtime of your job
#SBATCH --nodes=1                       # the number of nodes you want to use, 1 node is enough for your use-case
#SBATCH --mem=100G                      # maximum amount of RAM you use. If you exceed this number, the job gets cancelled!
#SBATCH --cpus-per-task=8               # specifies the number of CPU cores you allocate
#SBATCH --gres=gpu:a100:2               # this specifies what type of GPUs and how many you want, run the command `si` to find the names of the gpus in the column GRES
#SBATCH --mail-type=FAIL,END            # you can add BEGIN in this list, to get an email once a job starts
#SBATCH --mail-user=rahul.gupta@stud.uni-hannover.de     # email address, the updates of your jobs will be sent to
#SBATCH --output=slurm/%j-%x.out        # this outputs the logs into a folder `slurm` and file <JOB-ID>-<JOB-NAME>.out. You can also specify the --error option, to split the errors into a separate file (e.g. slurm/%j-%x.err)
#SBATCH --signal=B:USR1@300             # send SIGUSR1 to the batch step 5 min before the time limit so the trainer can checkpoint & exit gracefully

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"                  # run everything relative to where you submitted from
mkdir -p slurm                          # log dir must exist before slurm writes into it

# --- first build the tokenizer if it does not exist yet (CPU-only, a few minutes) ---
if [ ! -f ./unified_icd10_gemma_tokenizer/tokenizer.json ]; then
    python build_hierarchical_icd10_vocab.py --codes_source union
fi

# --- train ---
export TOKENIZERS_PARALLELISM=false
python train_unified_tokenizer.py \
    --unified_tokenizer_dir ./unified_icd10_gemma_tokenizer \
    --output_dir ./slurm_out \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation 8
