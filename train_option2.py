#!/usr/bin/env python3
"""
train_option2.py

Train MedGemma-3-4B (Unsloth) on medical_meadow_wikidoc with ICD-10 integration.
Approach: Option 2 - Special Token Replacement (Unified Text Modality)

ICD-10 codes in text are replaced with special tokens like <ICD10:A00.0>
which are added to the text tokenizer's vocabulary. Single modality, simpler!

Requirements:
pip install unsloth transformers datasets torch accelerate peft trl wandb
"""

import os
# MUST BE SET FIRST: Routes Unsloth/Gemma downloads to the NVMe drive
os.environ["HF_HOME"] = "/mnt/data/medgemma_project/hf_cache"

import json
import re
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType
from unsloth import FastLanguageModel
import wandb

# ============================================================
# CONFIGURATION
# ============================================================

# Special token format for ICD-10 codes in text
ICD10_SPECIAL_TOKEN_PREFIX = "<ICD10:"
ICD10_SPECIAL_TOKEN_SUFFIX = ">"
ICD10_PATTERN = re.compile(r'\b[A-TV-Z][0-9][0-9AB]\.?[0-9A-TV-Z]{0,4}\b')

# Maximum number of unique ICD-10 codes we expect to add as special tokens
# This controls vocabulary expansion size
MAX_ICD10_SPECIAL_TOKENS = 50000

# ============================================================
# ICD-10 TOKENIZER (for preprocessing only)
# ============================================================

class ICD10Tokenizer:
    """Loads the saved ICD-10 tokenizer for code validation/preprocessing."""
    def __init__(self, artifacts_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(artifacts_dir)
            
    def __len__(self):
        return len(self.tokenizer)
        
    def get_indices(self, code: str) -> List[int]:
        return self.tokenizer.encode(code, add_special_tokens=False)
    
    def is_valid_code(self, code: str) -> bool:
        """Check if a string is a valid ICD-10 code in our vocabulary."""
        try:
            indices = self.tokenizer.encode(code, add_special_tokens=False)
            # Valid if it tokenizes to something non-empty and not just UNK
            return len(indices) > 0 and (len(indices) > 1 or indices[0] != self.tokenizer.unk_token_id)
        except:
            return False

# ============================================================
# SPECIAL TOKEN MANAGEMENT
# ============================================================

class ICD10SpecialTokenManager:
    """
    Manages the mapping between ICD-10 codes and special tokens.
    Builds a vocabulary of special tokens from the training data.
    """
    def __init__(self, icd10_tokenizer: ICD10Tokenizer, max_tokens: int = MAX_ICD10_SPECIAL_TOKENS):
        self.icd10_tokenizer = icd10_tokenizer
        self.max_tokens = max_tokens
        self.code_to_special_token: Dict[str, str] = {}
        self.special_token_to_code: Dict[str, str] = {}
        self.special_tokens_list: List[str] = []
        
    def build_vocabulary_from_data(self, data_list: List[Dict]) -> List[str]:
        """
        Scan all data to find unique ICD-10 codes and create special tokens for them.
        Returns list of special tokens to add to tokenizer.
        """
        print("Scanning dataset for ICD-10 codes...")
        unique_codes: Set[str] = set()
        
        for item in data_list:
            # Codes from the structured field
            codes = item.get("codes", [])
            for code in codes:
                if self.icd10_tokenizer.is_valid_code(code):
                    unique_codes.add(code)
            
            # Also scan text fields for ICD-10 patterns
            for field in ["system", "user", "assistant"]:
                text = item.get(field, "")
                found_codes = ICD10_PATTERN.findall(text)
                for code in found_codes:
                    if self.icd10_tokenizer.is_valid_code(code):
                        unique_codes.add(code)
        
        print(f"Found {len(unique_codes)} unique valid ICD-10 codes in dataset")
        
        # Limit to max_tokens most frequent (or just first N if no frequency info)
        sorted_codes = sorted(list(unique_codes))[:self.max_tokens]
        
        # Create special tokens
        self.special_tokens_list = []
        for code in sorted_codes:
            special_token = f"{ICD10_SPECIAL_TOKEN_PREFIX}{code}{ICD10_SPECIAL_TOKEN_SUFFIX}"
            self.code_to_special_token[code] = special_token
            self.special_token_to_code[special_token] = code
            self.special_tokens_list.append(special_token)
        
        print(f"Created {len(self.special_tokens_list)} special tokens")
        return self.special_tokens_list
    
    def replace_codes_in_text(self, text: str) -> str:
        """Replace all ICD-10 codes in text with their special tokens."""
        def replace_match(match):
            code = match.group(0)
            if code in self.code_to_special_token:
                return self.code_to_special_token[code]
            # If code not in our vocabulary, keep as-is (will be tokenized normally)
            return code
        
        return ICD10_PATTERN.sub(replace_match, text)
    
    def get_special_tokens(self) -> List[str]:
        return self.special_tokens_list
    
    def decode_special_tokens(self, text: str) -> str:
        """Convert special tokens back to ICD-10 codes (for generation/output)."""
        for special, code in self.special_token_to_code.items():
            text = text.replace(special, code)
        return text

# ============================================================
# DATASET (Unified Text Modality)
# ============================================================

class SynthAIICD10Dataset(Dataset):
    """
    Dataset that preprocesses text to replace ICD-10 codes with special tokens.
    Single modality - only text tokenization needed!
    """
    def __init__(self, data_list, special_token_manager: ICD10SpecialTokenManager, 
                 text_tokenizer, max_text_length=2048, choose_response: str = "icd10"):
        self.dataset = data_list
        self.special_token_manager = special_token_manager
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.response = choose_response

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        system_prompt = item.get("system", "")
        user_prompt = item.get("user", "")
        
        assistant_response_icd10 = ""
        assistant_response_explanation = ""
        assistant_full = item.get("assistant", "")
        
        # Split assistant response if it contains codes + explanation
        codes_str = ", ".join(item.get("codes", []))
        if codes_str and codes_str + "\n\n" in assistant_full:
            parts = assistant_full.split(codes_str + "\n\n", 1)
            assistant_response_icd10 = parts[0] + codes_str
            assistant_response_explanation = parts[1] if len(parts) > 1 else ""
        else:
            assistant_response_icd10 = assistant_full
            assistant_response_explanation = assistant_full
        
        assistant_response = assistant_response_icd10 if self.response == "icd10" else assistant_response_explanation
        
        # Build conversation
        conversation = f"### Question:\n{user_prompt}\n\n### Answer:\n{assistant_response}"
        
        # KEY STEP: Replace ICD-10 codes in text with special tokens
        processed_conversation = self.special_token_manager.replace_codes_in_text(conversation)
        
        # Tokenize as regular text (single modality!)
        text_encoding = self.text_tokenizer(
            processed_conversation,
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "labels": text_encoding["input_ids"].squeeze(0).clone()
        }

# ============================================================
# MODEL WRAPPER (Simplified - No ICD-10 Embedding/Fusion!)
# ============================================================

class MedGemmaWithICD10(nn.Module):
    """
    Simplified model - just MedGemma with LoRA, no separate ICD-10 modules!
    The ICD-10 knowledge is embedded in the text tokenizer vocabulary.
    """
    def __init__(self, model_name, lora_config=None):
        super().__init__()
        
        print(f"Loading {model_name} with Unsloth...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None, 
            load_in_4bit=True,
        )
        
        if lora_config is None:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            "loss": outputs.loss,
            "lm_logits": outputs.logits,
        }

    def generate(self, input_ids, attention_mask, **kwargs):
        """Generate with the model."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

# ============================================================
# CUSTOM TRAINER (Simplified)
# ============================================================

class ICD10Trainer:
    def __init__(self, model, train_dataset, eval_dataset, args, special_token_manager=None):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args
        self.special_token_manager = special_token_manager

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        num_training_steps = len(train_dataset) * args.num_epochs // (args.batch_size * args.gradient_accumulation)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_training_steps
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        self.eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler('cuda')

    def train(self):
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            
            for step, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs["loss"] / self.args.gradient_accumulation
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.args.gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                total_loss += loss.item() * self.args.gradient_accumulation
                
                if step % 10 == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss.item() * self.args.gradient_accumulation:.4f}")
                    if wandb.run:
                        wandb.log({"train/loss": loss.item() * self.args.gradient_accumulation,
                                   "train/lr": self.scheduler.get_last_lr()[0]})
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
            
            self.evaluate(epoch)
            self.save_checkpoint(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    total_loss += outputs["loss"].item()
        
        avg_loss = total_loss / len(self.eval_loader)
        print(f"Epoch {epoch} eval loss: {avg_loss:.4f}")
        
        if wandb.run:
            wandb.log({"eval/loss": avg_loss})
        
        self.model.train()
        return avg_loss

    def save_checkpoint(self, epoch):
        output_dir = Path(self.args.output_dir) / f"checkpoint-epoch-{epoch}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.model.save_pretrained(output_dir / "lora_adapters")
        self.model.tokenizer.save_pretrained(output_dir / "tokenizer")
        
        # Save special token manager for inference
        if self.special_token_manager:
            import pickle
            with open(output_dir / "special_token_manager.pkl", "wb") as f:
                pickle.dump(self.special_token_manager, f)
        
        print(f"Saved checkpoint to {output_dir}")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train MedGemma with ICD-10 special tokens (Option 2)")
    parser.add_argument('--data_path', type=str, 
                        default="/mnt/data/medgemma_project/dataset/annotated_medical_meadow.jsonl",
                        help="Path to annotated_medical_meadow.jsonl")
    parser.add_argument('--icd10_artifacts_dir', type=str, 
                        default="/mnt/data/medgemma_project/icd10_tokenizer")
    parser.add_argument('--model_name', type=str, default="unsloth/gemma-3-4b-it-bnb-4bit")
    parser.add_argument('--output_dir', type=str, 
                        default="/mnt/data/medgemma_project/medgemma-icd10_option2_checkpoints")
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_text_length', type=int, default=2048)
    parser.add_argument('--max_icd10_special_tokens', type=int, default=MAX_ICD10_SPECIAL_TOKENS,
                        help="Max unique ICD-10 codes to add as special tokens")
    parser.add_argument('--response_mode', type=str, default="icd10", choices=["icd10", "explanation"],
                        help="Which part of assistant response to train on")
    args = parser.parse_args()

    wandb.init(project="medgemma-icd10-option2", config=vars(args))

    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    print(f"Loaded {len(train_data)} training samples and {len(eval_data)} evaluation samples.")

    # Load ICD-10 tokenizer (for preprocessing/validation only)
    print("Loading ICD-10 tokenizer...")
    icd10_tokenizer = ICD10Tokenizer(args.icd10_artifacts_dir)
    print(f"  Loaded tokenizer with vocab size {len(icd10_tokenizer)}")

    # Build special token vocabulary from training data
    print("Building ICD-10 special token vocabulary...")
    special_token_manager = ICD10SpecialTokenManager(icd10_tokenizer, max_tokens=args.max_icd10_special_tokens)
    special_tokens = special_token_manager.build_vocabulary_from_data(train_data + eval_data)
    
    # Load text tokenizer and ADD special tokens
    print("Loading text tokenizer and adding special tokens...")
    text_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    
    # Add special tokens to tokenizer
    num_added = text_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Added {num_added} special tokens to text tokenizer")
    print(f"New tokenizer vocab size: {len(text_tokenizer)}")
    
    # IMPORTANT: Resize model embeddings after adding tokens
    # This will be done after model loading in MedGemmaWithICD10

    # Datasets (with preprocessing)
    print("Preparing datasets...")
    train_dataset = SynthAIICD10Dataset(
        train_data, special_token_manager, text_tokenizer, 
        max_text_length=args.max_text_length, choose_response=args.response_mode
    )

    eval_dataset = SynthAIICD10Dataset(
        eval_data, special_token_manager, text_tokenizer, 
        max_text_length=args.max_text_length, choose_response=args.response_mode
    )

    # Model (simplified!)
    print("Initializing model...")
    model = MedGemmaWithICD10(
        model_name=args.model_name,
    )
    
    # Resize embeddings to accommodate new special tokens
    print(f"Resizing model embeddings from {model.model.config.vocab_size} to {len(text_tokenizer)}")
    model.model.resize_token_embeddings(len(text_tokenizer))
    model.tokenizer = text_tokenizer  # Update tokenizer reference

    # Trainer
    trainer = ICD10Trainer(model, train_dataset, eval_dataset, args, special_token_manager)

    # Train
    print("Starting training...")
    trainer.train()

    # Final save
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(final_dir / "lora_adapters")
    model.tokenizer.save_pretrained(final_dir / "tokenizer")
    
    import pickle
    with open(final_dir / "special_token_manager.pkl", "wb") as f:
        pickle.dump(special_token_manager, f)

    print(f"Training complete! Model saved to {final_dir}")
    wandb.finish()

if __name__ == '__main__':
    main()