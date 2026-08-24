#!/usr/bin/env python3
"""
train.py

Train MedGemma-3-4B (Unsloth) on medical_meadow_wikidoc with ICD-10 integration.
Approach: Hybrid (Approach A) - Text encoder + ICD-10 embedding layer + fusion

- Keeps MedGemma's language understanding intact
- Adds structured ICD-10 knowledge via separate embedding layer
- Uses hierarchical initialization for ICD-10 embeddings

Requirements:
pip install unsloth transformers datasets torch accelerate peft trl wandb

Usage:
python train.py \
    --icd10_artifacts_dir ./icd10_tokenizer \
    --output_dir ./medgemma-icd10 \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation 8
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from unsloth import FastLanguageModel
import wandb

# ============================================================
# ICD-10 ANNOTATOR RECOMMENDATIONS
# ============================================================

ICD10_ANNOTATOR_OPTIONS = {
    "clinical_bert_icd10": {
        "model": "emilyalsentzer/Bio_ClinicalBERT",
        "head": "Multi-label classification (19,927 labels)",
        "source": "Fine-tune on MIMIC-III/IV discharge summaries",
        "pros": "Clinical domain, free, well-known",
        "cons": "Needs fine-tuning, 19K labels = large head"
    },
    "bert_icd10_cm": {
        "model": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "head": "Sentence embedding + k-NN on code descriptions",
        "source": "Embed descriptions, retrieve nearest",
        "pros": "No training needed, handles new codes",
        "cons": "Slower inference, less accurate"
    },
    "icd10_gpt": {
        "model": "microsoft/biogpt-large",
        "head": "Generative (predict codes as text)",
        "source": "Fine-tune on clinical notes -> ICD-10 sequences",
        "pros": "Natural language generation, flexible",
        "cons": "Needs GPU, slower"
    },
    "quick_annotator": {
        "model": "distilbert-base-uncased",
        "head": "Multi-label on MIMIC-IV (top 1000 codes)",
        "source": "Fast, covers 80% of cases",
        "pros": "Fast, lightweight",
        "cons": "Misses rare codes"
    }
}

# ============================================================
# MISSING TOKENIZER STUB (Added to make script runnable)
# ============================================================

class ICD10Tokenizer:
    def __init__(self, artifacts_dir):
        with open(Path(artifacts_dir) / "code_to_idx.json") as f:
            self.code_to_idx = json.load(f)
            
    def __len__(self):
        return len(self.code_to_idx)
        
    def get_indices(self, codes: List[str]) -> List[int]:
        return [self.code_to_idx.get(c, 0) for c in codes]

# ============================================================
# ICD-10 EMBEDDING LAYER WITH HIERARCHICAL INITIALIZATION
# ============================================================

class HierarchicalICD10Embedding(nn.Module):
    """
    ICD-10 embedding layer with hierarchical initialization.
    Codes sharing chapter/block start with similar embeddings.
    """

    def __init__(self, code_to_idx_path, hierarchy_path, hidden_size=2560, init_from_hierarchy=True):
        super().__init__()
        
        with open(code_to_idx_path) as f:
            self.code_to_idx = json.load(f)
        with open(hierarchy_path) as f:
            self.hierarchy = json.load(f)
        
        self.num_codes = len(self.code_to_idx)
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(self.num_codes, hidden_size)
        
        if init_from_hierarchy:
            self._initialize_hierarchical()

    def _initialize_hierarchical(self):
        """Initialize embeddings so hierarchical neighbors are close"""
        print("Initializing ICD-10 embeddings hierarchically...")
        
        # Group by chapter
        chapter_groups = {}
        block_groups = {}
        
        for code, idx in self.code_to_idx.items():
            h = self.hierarchy.get(code, {})
            chapter = h.get('chapter', code[0] if code else 'X')
            block = h.get('block', code[:3] if len(code) >= 3 else code)
            
            chapter_groups.setdefault(chapter, []).append(idx)
            block_groups.setdefault(block, []).append(idx)
        
        with torch.no_grad():
            # 1. Chapter-level initialization
            for chapter, indices in chapter_groups.items():
                chapter_vec = torch.randn(self.hidden_size) * 0.02
                for idx in indices:
                    self.embedding.weight[idx] = chapter_vec.clone()
            
            # 2. Block-level perturbation
            for block, indices in block_groups.items():
                block_vec = torch.randn(self.hidden_size) * 0.01
                for idx in indices:
                    self.embedding.weight[idx] += block_vec
            
            # 3. Individual noise
            self.embedding.weight.data += torch.randn_like(self.embedding.weight) * 0.005
        
        print(f"  Initialized {self.num_codes} codes across {len(chapter_groups)} chapters")

    def forward(self, code_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            code_indices: [B, N_codes] or [N_codes]
        Returns:
            [B, N_codes, H] or [N_codes, H]
        """
        return self.embedding(code_indices)

    def get_indices(self, codes: List[str]) -> List[int]:
        """Convert code strings to indices"""
        return [self.code_to_idx.get(c, 0) for c in codes]

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
        
        # Project ICD-10 embeddings to MedGemma hidden size (if different)
        self.icd10_proj = nn.Linear(hidden_size, hidden_size)
        
        # Cross-attention: ICD-10 queries attend to text keys/values
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_hidden: torch.Tensor, icd10_embeds: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            text_hidden: [B, L_text, H] - MedGemma hidden states
            icd10_embeds: [B, N_codes, H] - ICD-10 embeddings
            text_mask: [B, L_text] - attention mask for text
        Returns:
            [B, H] - fused representation (pooled)
        """
        B, N_codes, H = icd10_embeds.shape
        
        # Project ICD-10 embeddings
        icd10_proj = self.icd10_proj(icd10_embeds)  # [B, N_codes, H]
        
        # Cross-attention: ICD-10 codes query the text
        # Query: ICD-10, Key/Value: Text
        attn_out, _ = self.cross_attn(
            query=icd10_proj,
            key=text_hidden,
            value=text_hidden,
            key_padding_mask=~text_mask.bool() if text_mask is not None else None
        )  # [B, N_codes, H]
        
        # Pool across codes (mean + max)
        pooled_mean = attn_out.mean(dim=1)  # [B, H]
        pooled_max = attn_out.max(dim=1)[0]  # [B, H]
        pooled = pooled_mean + pooled_max
        
        # Output projection
        out = self.output_proj(self.dropout(pooled))
        out = self.norm(out)
        
        return out  # [B, H]

# ============================================================
# DATASET
# ============================================================

class MedicalMeadowICD10Dataset(Dataset):
    """
    Dataset for medical_meadow_wikidoc with ICD-10 codes.
    """

    def __init__(self, icd10_tokenizer, text_tokenizer, split="train", 
                 max_text_length=2048, max_codes=10, num_samples=None):
        self.icd10_tokenizer = icd10_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.max_codes = max_codes
        
        print(f"Loading medical_meadow_wikidoc ({split})...")
        self.dataset = load_dataset("medical-meadow/medical_meadow_wikidoc", split=split)
        
        if num_samples:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Format conversation
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        conversation = f"### Question:\n{question}\n\n### Answer:\n{answer}"
        
        # Tokenize text
        text_encoding = self.text_tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # TODO: In production, run your ICD-10 annotator here
        # For now, simulate with placeholder - REPLACE WITH YOUR ANNOTATOR
        icd10_codes = self._get_icd10_codes(question, answer)
        
        # Convert to indices
        icd10_indices = self.icd10_tokenizer.get_indices(icd10_codes)
        
        # Pad/truncate
        if len(icd10_indices) < self.max_codes:
            icd10_indices += [0] * (self.max_codes - len(icd10_indices))
        else:
            icd10_indices = icd10_indices[:self.max_codes]
        
        return {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "icd10_indices": torch.tensor(icd10_indices, dtype=torch.long),
            "icd10_codes": icd10_codes,
            "labels": text_encoding["input_ids"].squeeze(0).clone()  # For causal LM
        }

    def _get_icd10_codes(self, question: str, answer: str) -> List[str]:
        """
        PLACEHOLDER - Replace with your actual ICD-10 annotator!
        
        Options:
        1. Call your fine-tuned ClinicalBERT annotator
        2. Use rule-based keyword matching (quick baseline)
        3. Use embedding similarity to code descriptions
        """
        text = (question + " " + answer).lower()
        codes = []
        
        # Simple keyword mapping (expand this!)
        keyword_map = {
            "myocardial infarction": "I219",
            "mi": "I219",
            "heart attack": "I219",
            "chest pain": "R079",
            "diabetes": "E119",
            "hypertension": "I10",
            "pneumonia": "J189",
            "copd": "J441",
            "stroke": "I639",
            "sepsis": "A419",
            "cancer": "C801",
            "depression": "F329",
            "anxiety": "F419",
        }
        
        for keyword, code in keyword_map.items():
            if keyword in text:
                codes.append(code)
        
        return codes[:self.max_codes] if codes else ["Z0000"]  # Default: general exam

# ============================================================
# MODEL WRAPPER
# ============================================================

class MedGemmaWithICD10(nn.Module):
    """
    MedGemma + ICD-10 embedding + fusion.
    """

    def __init__(self, model_name, icd10_tokenizer_path, lora_config=None):
        super().__init__()
        
        # Load ICD-10 artifacts
        self.icd10_tokenizer = ICD10Tokenizer(icd10_tokenizer_path)
        
        # Load MedGemma with Unsloth
        print(f"Loading {model_name} with Unsloth...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,  # Auto
            load_in_4bit=True,
        )
        
        # Apply LoRA
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
        
        # ICD-10 components
        hidden_size = self.model.config.hidden_size  # 2560 for Gemma-3-4B
        
        self.icd10_embedding = HierarchicalICD10Embedding(
            code_to_idx_path=Path(icd10_tokenizer_path) / "code_to_idx.json",
            hierarchy_path=Path(icd10_tokenizer_path) / "hierarchy.json",
            hidden_size=hidden_size
        )
        
        self.fusion = ICD10FusionModule(hidden_size=hidden_size)
        
        # Output head for ICD-10 prediction (optional auxiliary task)
        self.icd10_head = nn.Linear(hidden_size, len(self.icd10_tokenizer))

    def forward(self, input_ids, attention_mask, icd10_indices, labels=None):
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            icd10_indices: [B, N_codes]
            labels: [B, L] (for causal LM loss)
        """
        # Get MedGemma hidden states
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Use last layer hidden states
        text_hidden = outputs.hidden_states[-1]  # [B, L, H]
        
        # ICD-10 embeddings
        icd10_embeds = self.icd10_embedding(icd10_indices)  # [B, N_codes, H]
        
        # Fusion
        fused = self.fusion(text_hidden, icd10_embeds, attention_mask)  # [B, H]
        
        # Standard LM loss (on original logits)
        lm_logits = self.model.lm_head(text_hidden)  # [B, L, V]
        
        loss = None
        if labels is not None:
            # Causal LM loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Optional: Auxiliary ICD-10 prediction loss
            icd10_logits = self.icd10_head(fused)  # [B, num_codes]
            icd10_labels = torch.zeros_like(icd10_logits)
            for b in range(icd10_indices.size(0)):
                for idx in icd10_indices[b]:
                    if idx > 0:
                        icd10_labels[b, idx] = 1.0
            
            icd10_loss = F.binary_cross_entropy_with_logits(icd10_logits, icd10_labels)
            loss = loss + 0.1 * icd10_loss  # Weighted auxiliary loss
        
        return {
            "loss": loss,
            "lm_logits": lm_logits,
            "icd10_logits": self.icd10_head(fused) if labels is not None else None,
            "fused_representation": fused
        }

    def generate(self, input_ids, attention_mask, icd10_indices, **kwargs):
        """Generate with ICD-10 conditioning"""
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        text_hidden = outputs.hidden_states[-1]
        icd10_embeds = self.icd10_embedding(icd10_indices)
        fused = self.fusion(text_hidden, icd10_embeds, attention_mask)
        
        return self.model.generate(input_ids, attention_mask=attention_mask, **kwargs)

# ============================================================
# CUSTOM TRAINER
# ============================================================

class ICD10Trainer:
    def __init__(self, model, train_dataset, eval_dataset, args):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        num_training_steps = len(train_dataset) * args.num_epochs // (args.batch_size * args.gradient_accumulation)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_training_steps
        )
        
        # DataLoaders
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
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda')

    def train(self):
        self.model.train()
        
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            
            for step, batch in enumerate(self.train_loader):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                icd10_indices = batch["icd10_indices"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward with AMP
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        icd10_indices=icd10_indices,
                        labels=labels
                    )
                    loss = outputs["loss"] / self.args.gradient_accumulation
                
                # Backward
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
            
            # Evaluation
            self.evaluate(epoch)
            
            # Save checkpoint
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
        
        # Save LoRA adapters
        self.model.model.save_pretrained(output_dir / "lora_adapters")
        
        # Save ICD-10 components
        torch.save(self.model.icd10_embedding.state_dict(), output_dir / "icd10_embedding.pt")
        torch.save(self.model.fusion.state_dict(), output_dir / "fusion.pt")
        torch.save(self.model.icd10_head.state_dict(), output_dir / "icd10_head.pt")
        
        print(f"Saved checkpoint to {output_dir}")

# ============================================================
# MAIN
# ============================================================

@dataclass
class TrainArgs:
    icd10_artifacts_dir: str = "./icd10_tokenizer"
    model_name: str = "unsloth/gemma-3-4b-it-bnb-4bit"  # Unsloth MedGemma
    output_dir: str = "./medgemma-icd10"
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_text_length: int = 2048
    max_codes: int = 10
    num_train_samples: Optional[int] = None  # None = all
    num_eval_samples: int = 500
    wandb_project: str = "medgemma-icd10"
    wandb_run_name: Optional[str] = None

def main():
    parser = argparse.ArgumentParser(description="Train MedGemma with ICD-10 integration")
    parser.add_argument('--icd10_artifacts_dir', type=str, default="./icd10_tokenizer")
    parser.add_argument('--model_name', type=str, default="unsloth/gemma-3-4b-it-bnb-4bit")
    parser.add_argument('--output_dir', type=str, default="./medgemma-icd10")
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_text_length', type=int, default=2048)
    parser.add_argument('--max_codes', type=int, default=10)
    parser.add_argument('--num_train_samples', type=int, default=None)
    parser.add_argument('--num_eval_samples', type=int, default=500)
    parser.add_argument('--wandb_project', type=str, default="medgemma-icd10")
    parser.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Load ICD-10 tokenizer
    print("Loading ICD-10 tokenizer...")
    icd10_tokenizer = ICD10Tokenizer(args.icd10_artifacts_dir)
    print(f"  Loaded {len(icd10_tokenizer)} ICD-10 codes")

    # Load text tokenizer (MedGemma tokenizer)
    text_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    # Datasets
    print("Preparing datasets...")
    train_dataset = MedicalMeadowICD10Dataset(
        icd10_tokenizer, text_tokenizer, split="train",
        max_text_length=args.max_text_length,
        max_codes=args.max_codes,
        num_samples=args.num_train_samples
    )

    eval_dataset = MedicalMeadowICD10Dataset(
        icd10_tokenizer, text_tokenizer, split="validation",
        max_text_length=args.max_text_length,
        max_codes=args.max_codes,
        num_samples=args.num_eval_samples
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