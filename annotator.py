#!/usr/bin/env python3
"""
annotate_dataset.py

Offline script to annotate medical_meadow_wikidoc with ICD-10 codes using BioGPT.
Outputs: annotated_medical_meadow.jsonl
"""

import json
import re
from pathlib import Path
from typing import List, Set
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Standard ICD-10 regex pattern (e.g., I10, E11.9, J44.1, S72.001A)
ICD10_PATTERN = re.compile(r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b", re.IGNORECASE)

def extract_valid_codes(generated_text: str, valid_vocab: Set[str], max_codes: int = 5) -> List[str]:
    """Finds ICD-10-like tokens in text and keeps only those present in your vocabulary."""
    found = ICD10_PATTERN.findall(generated_text)
    
    # Normalize: strip dots or match standard formatting depending on your code_to_idx keys
    clean_codes = []
    for code in found:
        formatted = code.upper().replace(".", "") # or keep dot if your vocab uses dots
        if formatted in valid_vocab and formatted not in clean_codes:
            clean_codes.append(formatted)
        elif code.upper() in valid_vocab and code.upper() not in clean_codes:
            clean_codes.append(code.upper())
            
    return clean_codes[:max_codes]

def main():
    artifacts_dir = Path("./icd10_tokenizer")
    output_file = Path("./data/annotated_medical_meadow.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load your ICD-10 vocabulary
    with open(artifacts_dir / "code_to_idx.json", "r") as f:
        code_to_idx = json.load(f)
    valid_vocab = set(code_to_idx.keys())
    print(f"Loaded {len(valid_vocab)} valid ICD-10 codes.")

    # 2. Load dataset
    print("Loading medical_meadow_wikidoc...")
    dataset = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")

    # 3. Load BioGPT
    model_name = "microsoft/biogpt"
    print(f"Loading {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float321
    ).to(device)
    model.eval()

    # 4. Process and annotate in batches
    batch_size = 16
    annotated_records = []

    print("Generating ICD-10 codes with BioGPT...")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]
            
            prompts = []
            
            for q, a in zip(batch["input"], batch["output"]):
                # Truncate text to fit context window
                context = (q + " " + a)[:350]
                prompt = (
                    f"Medical passage: {context}\n"
                    f"Question: What are the primary ICD-10 diagnosis codes for this condition?\n"
                    f"ICD-10 codes:"
                )
                prompts.append(prompt)

            # Tokenize batch
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=25,
                    temperature=0.2,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode and parse
            for j, prompt in enumerate(prompts):
                # Get only the newly generated tokens
                input_len = inputs["input_ids"][j].shape[0]
                gen_text = tokenizer.decode(generated_ids[j][input_len:], skip_special_tokens=True)
                
                # Extract and filter valid codes
                codes = extract_valid_codes(gen_text, valid_vocab)
                
                # Fallback if BioGPT produced no valid code from vocabulary
                if not codes:
                    codes = ["Z0000"] if "Z0000" in valid_vocab else [list(valid_vocab)[0]]

                record = {
                    "question": batch["input"][j],
                    "answer": batch["output"][j],
                    "icd10_codes": codes
                }
                
                out_f.write(json.dumps(record) + "\n")

    print(f"Annotation complete! Saved to {output_file}")

if __name__ == "__main__":
    main()