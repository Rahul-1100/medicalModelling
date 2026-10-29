#!/usr/bin/env python3
"""
inference_option2.py

Inference script for MedGemma trained with Option 2 (Special Token Replacement).
Loads the model, tokenizer, and special token manager to generate ICD-10 codes.
"""

import os
os.environ["HF_HOME"] = "/mnt/data/medgemma_project/hf_cache"

import json
import pickle
import argparse
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

def load_model_and_tokenizer(checkpoint_dir: str):
    """Load the trained model, tokenizer, and special token manager."""
    print(f"Loading from {checkpoint_dir}...")
    
    # Load tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load special token manager
    with open(checkpoint_dir / "special_token_manager.pkl", "rb") as f:
        special_token_manager = pickle.load(f)
    
    # Load base model
    model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3-4b-it-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Resize embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA adapters
    model.load_adapter(checkpoint_dir / "lora_adapters")
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, tokenizer, special_token_manager

def preprocess_input(text: str, special_token_manager) -> str:
    """Replace ICD-10 codes in input text with special tokens."""
    return special_token_manager.replace_codes_in_text(text)

def postprocess_output(text: str, special_token_manager) -> str:
    """Convert special tokens back to ICD-10 codes in generated text."""
    return special_token_manager.decode_special_tokens(text)

def generate_icd10_codes(model, tokenizer, special_token_manager, 
                         user_prompt: str, system_prompt: str = "",
                         max_new_tokens: int = 512, temperature: float = 0.7):
    """
    Generate ICD-10 codes for a medical question.
    """
    # Build conversation
    conversation = f"### Question:\n{user_prompt}\n\n### Answer:\n"
    
    # Preprocess: replace any ICD-10 codes in the prompt with special tokens
    processed_conversation = preprocess_input(conversation, special_token_manager)
    
    # Tokenize
    inputs = tokenizer(
        processed_conversation,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # Postprocess: convert special tokens back to ICD-10 codes
    final_text = postprocess_output(generated_text, special_token_manager)
    
    return final_text

def main():
    parser = argparse.ArgumentParser(description="Inference with Option 2 trained model")
    parser.add_argument('--checkpoint_dir', type=str, 
                        default="/mnt/data/medgemma_project/medgemma-icd10_option2_checkpoints/final",
                        help="Path to final checkpoint directory")
    parser.add_argument('--user_prompt', type=str, required=True,
                        help="Medical question to generate ICD-10 codes for")
    parser.add_argument('--system_prompt', type=str, default="",
                        help="Optional system prompt")
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()
    
    # Load model and components
    model, tokenizer, special_token_manager = load_model_and_tokenizer(
        Path(args.checkpoint_dir)
    )
    
    print(f"Special tokens in vocabulary: {len(special_token_manager.get_special_tokens())}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print("\n" + "="*60)
    
    # Generate
    result = generate_icd10_codes(
        model, tokenizer, special_token_manager,
        user_prompt=args.user_prompt,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    print(f"Input: {args.user_prompt}")
    print(f"\nGenerated ICD-10 codes:\n{result}")

if __name__ == "__main__":
    main()