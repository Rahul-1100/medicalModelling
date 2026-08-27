#!/usr/bin/env python3
"""
train.py

Train MedGemma-3-4B (Unsloth) on medical_meadow_wikidoc with ICD-10 integration.
Approach: Hybrid - Text encoder + ICD-10 embedding layer + fusion

Requirements:
pip install unsloth transformers datasets torch accelerate peft trl wandb
"""

import os
# MUST BE SET FIRST: Routes Unsloth/Gemma downloads to the NVMe drive
os.environ["HF_HOME"] = "/mnt/data/medgemma_project/hf_cache"

import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

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
# TOKENIZER (Updated for Hugging Face artifact output)
# ============================================================

class ICD10Tokenizer:
    def __init__(self, artifacts_dir):
        # Loads the saved tokenizer from your tokenizer.py script
        self.tokenizer = AutoTokenizer.from_pretrained(artifacts_dir)
            
    def __len__(self):
        return len(self.tokenizer)
        
    def get_indices(self, code: str) -> List[int]:
        # encode() converts "A00.0" -> [10001, 10002, 45, ...] (approx 15-16 tokens)
        return self.tokenizer.encode(code, add_special_tokens=False)

# ============================================================
# ICD-10 EMBEDDING LAYER (Updated for 2D Sequence Padding)
# ============================================================

class FlattenedICD10Embedding(nn.Module):
    """
    Looks up tokens from the tokenizer (vocab ~28k) and flattens them 
    so cross-attention can read the hierarchy sequentially.
    """
    def __init__(self, vocab_size, hidden_size=2560):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, code_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            code_indices: [Batch, N_codes (10), Seq_len (18)]
        Returns:
            [Batch, N_codes * Seq_len (180), Hidden (2560)]
        """
        B, N_codes, seq_len = code_indices.shape
        
        # Look up embeddings: [B, 10, 18, 2560]
        embeds = self.embedding(code_indices)
        
        # Flatten N_codes and seq_len into a single sequence for cross-attention
        flat_embeds = embeds.view(B, N_codes * seq_len, -1) 
        
        return flat_embeds

# ============================================================
# FUSION MODULE
# ============================================================

class ICD10FusionModule(nn.Module):
    """
    Fuses ICD-10 embeddings with MedGemma hidden states.
    Uses cross-attention: ICD-10 codes attend to text representations.
    """
    def __init__(self, hidden_size=2560, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.icd10_proj = nn.Linear(hidden_size, hidden_size)
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_hidden: torch.Tensor, icd10_embeds: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        icd10_proj = self.icd10_proj(icd10_embeds)
        
        attn_out, _ = self.cross_attn(
            query=icd10_proj,
            key=text_hidden,
            value=text_hidden,
            key_padding_mask=~text_mask.bool() if text_mask is not None else None
        )
        
        pooled_mean = attn_out.mean(dim=1)
        pooled_max = attn_out.max(dim=1)[0]
        pooled = pooled_mean + pooled_max
        
        out = self.output_proj(self.dropout(pooled))
        return self.norm(out)

# ============================================================
# DATASET (Updated for 2D Padding [10 codes x 18 tokens])
# ============================================================

class MedicalMeadowICD10Dataset(Dataset):
    def __init__(self, data_list, icd10_tokenizer, text_tokenizer, max_text_length=2048, max_codes=10, max_code_tokens=18):
        self.dataset = data_list
        self.icd10_tokenizer = icd10_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.max_codes = max_codes
        self.max_code_tokens = max_code_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        conversation = f"### Question:\n{item['question']}\n\n### Answer:\n{item['answer']}"
        
        text_encoding = self.text_tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        raw_codes = item.get("icd10_codes", [])
        
        # Padded matrix of shape [10, 18] filled with 0s
        padded_matrix = torch.zeros((self.max_codes, self.max_code_tokens), dtype=torch.long)
        
        for i, code in enumerate(raw_codes[:self.max_codes]):
            code_tokens = self.icd10_tokenizer.get_indices(code)
            
            # Truncate to max_code_tokens (18) just in case
            code_tokens = code_tokens[:self.max_code_tokens]
            
            # Insert into our padded matrix
            padded_matrix[i, :len(code_tokens)] = torch.tensor(code_tokens, dtype=torch.long)
            
        return {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "icd10_indices": padded_matrix, # [10, 18]
            "labels": text_encoding["input_ids"].squeeze(0).clone()
        }

# ============================================================
# MODEL WRAPPER (Updated)
# ============================================================

class MedGemmaWithICD10(nn.Module):
    def __init__(self, model_name, icd10_tokenizer_path, lora_config=None):
        super().__init__()
        
        self.icd10_tokenizer = ICD10Tokenizer(icd10_tokenizer_path)
        
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
        
        hidden_size = self.model.config.hidden_size  
        
        # Swap in the new Flattened Embedding
        self.icd10_embedding = FlattenedICD10Embedding(
            vocab_size=len(self.icd10_tokenizer),
            hidden_size=hidden_size
        )
        
        self.fusion = ICD10FusionModule(hidden_size=hidden_size)
        
        # Auxiliary prediction head
        self.icd10_head = nn.Linear(hidden_size, len(self.icd10_tokenizer))

    def forward(self, input_ids, attention_mask, icd10_indices, labels=None):
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        text_hidden = outputs.hidden_states[-1] 
        icd10_embeds = self.icd10_embedding(icd10_indices) 
        fused = self.fusion(text_hidden, icd10_embeds, attention_mask)
        lm_logits = self.model.lm_head(text_hidden) 
        
        loss = None
        if labels is not None:
            # 1. Causal LM loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 2. Auxiliary ICD-10 Token Prediction Loss
            icd10_logits = self.icd10_head(fused) 
            icd10_labels = torch.zeros_like(icd10_logits)
            
            for b in range(icd10_indices.size(0)):
                # Target unique sub-tokens present in this batch item
                unique_indices = torch.unique(icd10_indices[b])
                for idx in unique_indices:
                    if idx > 0: # Ignore padding index 0
                        icd10_labels[b, idx] = 1.0
            
            icd10_loss = F.binary_cross_entropy_with_logits(icd10_logits, icd10_labels)
            loss = loss + 0.1 * icd10_loss 
        
        return {
            "loss": loss,
            "lm_logits": lm_logits,
        }

# ============================================================
# CUSTOM TRAINER
# ============================================================

class ICD10Trainer:
    def __init__(self, model, train_dataset, eval_dataset, args):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args

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
                icd10_indices = batch["icd10_indices"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        icd10_indices=icd10_indices,
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
                icd10_indices = batch["icd10_indices"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        icd10_indices=icd10_indices,
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
        torch.save(self.model.icd10_embedding.state_dict(), output_dir / "icd10_embedding.pt")
        torch.save(self.model.fusion.state_dict(), output_dir / "fusion.pt")
        torch.save(self.model.icd10_head.state_dict(), output_dir / "icd10_head.pt")
        
        print(f"Saved checkpoint to {output_dir}")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train MedGemma with ICD-10 integration")
    parser.add_argument('--data_path', type=str, 
                        default="/mnt/data/medgemma_project/dataset/annotated_medical_meadow.jsonl",
                        help="Path to annotated_medical_meadow.jsonl")
    parser.add_argument('--icd10_artifacts_dir', type=str, 
                        default="/mnt/data/medgemma_project/icd10_tokenizer")
    parser.add_argument('--model_name', type=str, default="unsloth/gemma-3-4b-it-bnb-4bit")
    parser.add_argument('--output_dir', type=str, 
                        default="/mnt/data/medgemma_project/medgemma-icd10_checkpoints")
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_text_length', type=int, default=2048)
    parser.add_argument('--max_codes', type=int, default=10)
    parser.add_argument('--max_code_tokens', type=int, default=18)
    args = parser.parse_args()

    wandb.init(project="medgemma-icd10", config=vars(args))

    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    print(f"Loaded {len(train_data)} training samples and {len(eval_data)} evaluation samples.")

    # Load ICD-10 tokenizer
    print("Loading ICD-10 tokenizer...")
    icd10_tokenizer = ICD10Tokenizer(args.icd10_artifacts_dir)
    print(f"  Loaded tokenizer with vocab size {len(icd10_tokenizer)}")

    # Load text tokenizer (MedGemma tokenizer)
    text_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    # Datasets
    print("Preparing datasets...")
    train_dataset = MedicalMeadowICD10Dataset(
        train_data, icd10_tokenizer, text_tokenizer, 
        max_text_length=args.max_text_length, max_codes=args.max_codes, max_code_tokens=args.max_code_tokens
    )

    eval_dataset = MedicalMeadowICD10Dataset(
        eval_data, icd10_tokenizer, text_tokenizer, 
        max_text_length=args.max_text_length, max_codes=args.max_codes, max_code_tokens=args.max_code_tokens
    )

    # Model
    print("Initializing model...")
    model = MedGemmaWithICD10(
        model_name=args.model_name,
        icd10_tokenizer_path=args.icd10_artifacts_dir
    )

    # Trainer
    trainer = ICD10Trainer(model, train_dataset, eval_dataset, args)

    # Train
    print("Starting training...")
    trainer.train()

    # Final save
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(final_dir / "lora_adapters")
    torch.save(model.icd10_embedding.state_dict(), final_dir / "icd10_embedding.pt")
    torch.save(model.fusion.state_dict(), final_dir / "fusion.pt")
    torch.save(model.icd10_head.state_dict(), final_dir / "icd10_head.pt")

    print(f"Training complete! Model saved to {final_dir}")
    wandb.finish()

if __name__ == '__main__':
    main()