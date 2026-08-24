
# ICD-10 Medical Text Processing & Training Pipeline

## Aim
This pipeline converts unstructured medical text into structured ICD-10 coded data using a BERT-based entity recognition model, then trains a Gemma-2-9B language model with custom ICD-10 tokens for medical coding tasks. The goal is to enable the model to understand and generate medical text with standardized ICD-10 classifications.

## Pipeline Steps

### 1. Custom Tokenizer Setup
- Load base tokenizer (BERT/Gemma) and append 100 reserved special tokens `<ICD_00>` through `<ICD_99>` [1]
- Expand model embedding matrix to accommodate new tokens

### 2. BERT ICD-10 Conversion Pipeline
- Initialize BERT NER pipeline for medical entity recognition and ICD-10 mapping [1]
- **Step 1**: Convert natural language medical text → Standard ICD-10 codes using BERT entity extraction [1]
- **Step 2**: Convert raw ICD-10 codes (e.g., A15, B20, C18.9) → Custom `<ICD_XX>` tokens via regex [1]

### 3. Coverage Auditing
- Audit dataset ICD-10 columns to verify 100% coverage by custom vocabulary without subword splitting [1]
- Generate audit report identifying unmapped codes and coverage percentage [1]

### 4. Model Configuration
- Load Gemma-2-9B with Unsloth 4-bit quantization [2]
- Configure QLoRA adapters (r=16, target modules: q/k/v/o/gate/up/down proj) [2]

### 5. Dataset Preprocessing
- Load medical dataset (Wikidoc or custom Kaggle CSV) [2]
- Apply full preprocessing pipeline to both instruction and output fields:
  - Text → ICD-10 codes (BERT) → `<ICD_XX>` tokens [2]
- Format in Gemma chat template (`<start_of_turn>user/end_of_turn`) [2]

### 6. Training
- Train with SFTTrainer using modified tokenizer containing ICD tokens [2]
- Save both LoRA adapter and custom tokenizer for inference [2]

## Conclusion
This end-to-end pipeline bridges clinical natural language processing with structured medical coding by:
1. **Standardizing** medical terminology through BERT-powered ICD-10 entity recognition
2. **Tokenizing** ICD codes as atomic vocabulary units for efficient model learning
3. **Fine-tuning** a compact LLM (Gemma-2-9B) with domain-specific tokenization
4. **Ensuring coverage** via automated auditing of ICD-10 representation

The resulting model can process clinical narratives and generate ICD-10 coded outputs directly, enabling applications in automated medical coding, clinical documentation improvement, and healthcare analytics.

## Requirements

### Python Dependencies
```bash
pip install unsloth transformers trl datasets accelerate torch pandas
```

### External Dependencies

| Component | Source | Purpose |
|-----------|--------|---------|
| **ICD-BERT** | [github.com/suamin/ICD-BERT](https://github.com/suamin/ICD-BERT) | Pre-trained BERT model for ICD-10 coding from clinical text. Use this model (or fine-tune it) as the `ICD_BERT_MODEL` in `train.py` for Step 2 of the pipeline. |
| **ICD-10 Verifier Dataset** | [kaggle.com/datasets/arashnic/icd10-codes-and-descriptions](https://www.kaggle.com/datasets/arashnic/icd10-codes-and-descriptions) | Reference dataset containing ICD-10 codes and descriptions. Use this to validate coverage via `ICD10CoverageAuditor` and/or as training data for the BERT ICD-10 converter. |

### Hardware Requirements
- GPU with ≥24GB VRAM (for 4-bit Gemma-2-9B + BERT pipeline)
- ≥32GB system RAM
- CUDA 11.8+ / PyTorch 2.0+

## Configuration

Update these variables in `train.py` before running:

```python
# BERT ICD-10 model (from ICD-BERT repo or your fine-tuned version)
ICD_BERT_MODEL = "suamin/ICD-BERT"  # or your local path

# Dataset selection
# dataset = load_dataset("csv", data_files="icd10_codes_and_descriptions.csv", split="train")
dataset = load_dataset("medical_meadow_wikidoc", split="train")
```

## Usage

```bash
# 1. Clone ICD-BERT (if using local)
git clone https://github.com/suamin/ICD-BERT

# 2. Download Kaggle dataset
kaggle datasets download -d arashnic/icd10-codes-and-descriptions
unzip icd10-codes-and-descriptions.zip

# 3. Install requirements
pip install -r requirements.txt

# 4. Run training
python train.py
```

## Output
- `gemma_med_icd_adapter/` — LoRA adapter + custom tokenizer with 100 `<ICD_XX>` tokens
- `icd10_unmapped_audit_report.csv` — Coverage audit results (if enabled)


## TODO
apply bert model and then see how can ICD10 code be fitted into the pipeline 