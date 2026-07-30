import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, pipeline
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
import re

# ==========================================
# 1. SETUP MODEL & CUSTOM ICD TOKENIZER
# ==========================================
max_seq_length = 2048
model_name = "unsloth/gemma-2-9b-it"

# Load Base Gemma Model
model, _ = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# Load and Modify Tokenizer with your 100 Reserved ICD Tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)
icd_tokens = [f"<ICD_{i:02d}>" for i in range(100)]
tokenizer.add_special_tokens({"additional_special_tokens": icd_tokens})

# CRITICAL STEP: Expand model embedding matrix for the new tokens
model.resize_token_embeddings(len(tokenizer))

# Set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 2. CONFIGURE QLORA ADAPTERS
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ==========================================
# 3. BERT ICD-10 PREPROCESSING PIPELINE (from tokenizer.py)
# ==========================================
# Initialize the BERT-based ICD-10 converter
# You can specify a medical BERT model fine-tuned for ICD-10 coding
# Example: "bert-base-uncased" or a medical-specific model like "emilyalsentzer/Bio_ClinicalBERT"
ICD_BERT_MODEL = "bert-base-uncased"  # Change to your fine-tuned ICD-10 BERT model

print(f"Loading BERT ICD-10 pipeline: {ICD_BERT_MODEL}")
icd_pipeline = pipeline(
    "ner",
    model=ICD_BERT_MODEL,
    tokenizer=ICD_BERT_MODEL,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

def replace_text_with_icd_codes(text: str) -> str:
    """
    Uses BERT to detect medical terms and replace them with predicted ICD-10 codes.
    From tokenizer.py ICDPreprocessingPipeline.replace_text_with_icd_codes
    """
    entities = icd_pipeline(text)
    
    # Reverse sort prevents string character offset shifting [1]
    entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    text_list = list(text)
    for entity in entities:
        start = entity['start']
        end = entity['end']
        # The entity_group should contain the ICD-10 code prediction
        icd_code = entity.get('entity_group', entity.get('word'))
        
        text_list[start:end] = f" {icd_code} "
        
    return "".join(text_list)

def convert_icd_to_tokens(text: str) -> str:
    """
    Replaces raw ICD-10 occurrences (e.g., A15, B20, C18.9) in text with custom <ICD_XX> tokens.
    Enhanced from tokenizer.py to match ANY letter A-Z (not just A/B)
    """
    def replace_match(match):
        digits = match.group(2)
        return f"<ICD_{digits}>"

    # Matches any letter A-Z, optional dot/space, 2 digits, optional decimal
    return re.sub(r"\b([A-Z])\.?\s*(\d{2})(?:\.\d+)?\b", replace_match, text)

def preprocess_medical_text(text: str) -> str:
    """
    Complete preprocessing pipeline from tokenizer.py ICDPreprocessingPipeline.preprocess:
    Step 1: Normal Text -> Standard ICD-10 codes (via BERT)
    Step 2: ICD-10 codes -> <ICD_XX> Tokens (via Regex)
    """
    if not text or not isinstance(text, str):
        return text
    
    # Step 1: Convert medical entities to ICD-10 codes using BERT [1]
    text_with_icd = replace_text_with_icd_codes(text)
    
    # Step 2: Convert ICD-10 codes to custom tokens
    final_text = convert_icd_to_tokens(text_with_icd)
    
    return final_text

# ==========================================
# 4. LOAD DATASET & APPLY FULL PREPROCESSING
# ==========================================
# Load your dataset
# dataset = load_dataset("csv", data_files="your_kaggle_file.csv", split="train")
dataset = load_dataset("medical_meadow_wikidoc", split="train")

print(f"Dataset loaded: {len(dataset)} samples")

def format_prompts(examples):
    instructions = examples["input"]
    outputs = examples["output"]

    texts = []
    for inst, out in zip(instructions, outputs):
        # Apply full preprocessing pipeline to both instruction and output [2]
        inst_clean = preprocess_medical_text(inst)
        out_clean = preprocess_medical_text(out)

        formatted_text = (
            f"<start_of_turn>user\n{inst_clean}<end_of_turn>\n"
            f"<start_of_turn>model\n{out_clean}<end_of_turn>"
        )
        texts.append(formatted_text)

    return {"text": texts}

print("Applying BERT ICD-10 preprocessing to dataset...")
dataset = dataset.map(format_prompts, batched=True, num_proc=2)

# ==========================================
# 5. TRAIN WITH CUSTOM TOKENIZER
# ==========================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,  # Pass your modified tokenizer with ICD tokens
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer.train()

# Save both adapter and custom tokenizer
model.save_pretrained("gemma_med_icd_adapter")
tokenizer.save_pretrained("gemma_med_icd_adapter")

print("Training complete! Model and tokenizer saved to 'gemma_med_icd_adapter'")