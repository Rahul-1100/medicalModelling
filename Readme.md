# ICD-10 Medical Coding with a Unified Hierarchical Tokenizer

Fine-tune a MedGemma (Gemma-3-4B) language model so it can read clinical text and
generate ICD-10 codes as **hierarchical tokens** — e.g. the code `A02.0` becomes the
token sequence:

```
CHAP_A   BLOCK_A02   CAT_A02   SUBCAT_0
```

These hierarchy tokens are added to the pretrained Gemma vocabulary, so all text and
all codes flow through **one tokenizer and one embedding layer**.

## How it works (big picture)

1. **Build the unified tokenizer** (`build_hierarchical_icd10_vocab.py`)
   - Loads ICD-10 codes (default: the `FiscaAI/synth-ehr-icd10cm-prompt` dataset from Hugging Face).
   - Splits each code into hierarchy components (chapter / block / category / subcategory).
   - Trains a small BPE on those components, then merges the learned tokens into the
     Gemma tokenizer (Gemma's original vocab + token IDs are never changed; new tokens
     are appended at the end).
   - Output directory: `./unified_icd10_gemma_tokenizer/`

2. **Train with the unified tokenizer** (`train_unified_tokenizer.py`)
   - Loads MedGemma-3-4B (4-bit, Unsloth) + LoRA.
   - Resizes the embedding matrix to the unified vocab size (new rows are trainable via
     `modules_to_save=["embed_tokens"]`).
   - Rewrites ICD-10 codes in each training example into hierarchy token strings
     (`CHAP_A BLOCK_A02 CAT_A02 SUBCAT_0`) and encodes everything with the unified tokenizer.
   - Data defaults to the same SynthAI dataset, so training and tokenizer share identical codes.

## Requirements

- GPU with ~16GB+ VRAM (4-bit 4B model + LoRA)
- 32GB system RAM is plenty
- Python 3.10+ with: `pip install unsloth transformers datasets tokenizers pandas torch wandb peft`

## Step-by-step: build tokenizer, then train

### Step 1 — Build the unified tokenizer

```bash
python build_hierarchical_icd10_vocab.py --codes_source union
```

- `--codes_source` chooses which codes train the BPE:
  - `union` (recommended): SynthAI codes + the full ICD-10 registry in `./2/`
  - `synth`: only the SynthAI dataset codes (4,407 unique)
  - `files`: only the CSV/TXT registry in `./2/`
- The script downloads the SynthAI dataset to local cache `FiscaAI___synth-ehr-icd10cm-prompt/`
  on first run (about 1 GB).
- It ends with a self-test. Verify it prints `CHAP_A` etc. under
  "Special ICD tokens in vocab" and a final vocab of `262144 + <new tokens>`.
- **Runtime:** a few minutes on CPU; no GPU or training involved.

### Step 2 — Train

```bash
python train_unified_tokenizer.py \
  --unified_tokenizer_dir ./unified_icd10_gemma_tokenizer \
  --output_dir ./hf_unified_out
```

What you can (and usually don't need to) change:

| Flag | Default | Notes |
|------|---------|-------|
| `--data_source` | `synth` | `synth` = Hugging Face SynthAI dataset; `jsonl` = custom file via `--data_path` (JSONL with `system/user/assistant/codes`) |
| `--unified_tokenizer_dir` | — | required, output of Step 1 |
| `--model_name` | `unsloth/gemma-3-4b-it-bnb-4bit` | any Gemma-3 checkpoint with a matching base tokenizer |
| `--response_mode` | `icd10_custom` | `icd10` = plain codes, `explanation` = explanation text only |
| `--num_epochs`, `--batch_size`, `--learning_rate` | 3 / 2 / 2e-4 | LoRA-friendly defaults; scale with data size |

The tokenizer-match sanity check runs at startup: it prints the number of `CHAP_* /
BLOCK_* / CAT_* / SUBCAT_*` tokens found in the unified vocab. Because the vocabulary
is derived from the same SynthAI code set as the training data, every code in training
data maps to single, stable token IDs.

### Step 3 — Inspect checkpoints

Each epoch saves a checkpoint under `--output_dir`:

```
checkpoint-epoch-N/
├── lora_adapters/    # LoRA weights + trainable embed_tokens
└── tokenizer/        # the unified tokenizer (needed for inference later)
```

Note: checkpoints include the embedding matrix because `embed_tokens` is trained —
each one is a few hundred MB larger than a plain LoRA adapter.

## Troubleshooting

- **`[hierarchy] 'BLOCK_A00' not in unified vocab`** — a code in the training data isn't
  covered by the tokenizer build. Rebuild with `--codes_source union` (Step 1).
- **`Could not load synth dataset`** — no internet and no local cache. Run Step 1 once
  with internet, or pass `--synth_cache_dir` pointing to an existing HF cache.
- **HF rate-limits / auth** — set `export HF_TOKEN=<your token>` for faster downloads.
- **Wrong/old tokenizer in use** — if `build_hierarchical_icd10_vocab.py` output changed
  (e.g. new `--codes_source`), delete `unified_icd10_gemma_tokenizer/` and rebuild so
  `merge_info.json` matches.

## Repo map

| File | Purpose |
|------|---------|
| `build_hierarchical_icd10_vocab.py` | Build pipeline: codes → hierarchical BPE → merged with Gemma → `unified_icd10_gemma_tokenizer/` |
| `train_unified_tokenizer.py` | Training: MedGemma + LoRA + unified tokenizer + SynthAI data |
| `inference_option2.py`, `train_option2.py` | Legacy Option-2 pipeline (separate code-embedding head); superseded |
| `tokenizer.py`, `annotator.py`, `test_tokenizer.py` | Utilities from the earlier code-as-index approach |
| `icd10Download.py` | Helper to fetch the ICD-10 reference files in `./2/` |

The bundled `./2/` folder holds the full ICD-10-CM registry (CSV + ICD9→ICD10 map),
used by `--codes_source union` and by the legacy utilities.
