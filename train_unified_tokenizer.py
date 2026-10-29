#!/usr/bin/env python3
"""
train_unified_tokenizer.py

Train MedGemma-3-4B with Unified Hierarchical ICD-10 Tokenizer (Option 1 + BPE)
================================================================================

This approach:
1. Builds hierarchical BPE vocabulary from ICD-10 codes (A00.0 → <CHAP>A <BLOCK>A00 <CAT>A00 <SUBCAT>0)
2. Merges with Gemma tokenizer (adds ~5-8k new tokens)
3. Single modality, single tokenizer, single embedding layer
4. ICD-10 codes in text are naturally tokenized as compositional hierarchy tokens
5. Model learns to GENERATE valid ICD-10 codes by predicting hierarchy tokens

Benefits:
- Compact vocabulary (vs 28k code-to-index or 50k special tokens)
- Compositional: A00.0, A00.1, A00.9 share <CHAP>A <BLOCK>A00 <CAT>A00
- Generalizes to unseen codes via shared prefixes
- No separate modality / fusion needed
"""

import os
os.environ["HF_HOME"] = "/mnt/data/medgemma_project/hf_cache"

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from unsloth import FastLanguageModel
import wandb


# ============================================================
# DATASET (Standard Causal LM with Unified Tokenizer)
# ============================================================
class UnifiedICD10Dataset(Dataset):
    """
    Causal LM dataset. In "icd10_custom" mode, ICD-10 codes in the assistant response are
    rewritten as hierarchical token strings (CHAP_A BLOCK_A00 CAT_A00 SUBCAT_0). These
    strings are then encoded by the UNIFIED tokenizer (Gemma vocab + merged ICD hierarchy
    tokens), which resolves every CHAP_/BLOCK_/CAT_/SUBCAT_/NUM token to a single stable ID.

    No standalone ICD-10 tokenizer is needed at runtime: the hierarchy tokens already live
    in the unified vocab, and the hierarchical string is constructed with plain string ops.
    The standalone icd10_hierarchical_bpe.json matters only at BUILD time (it produced the
    vocabulary that build_hierarchical_icd10_vocab.py merged into Gemma).
    """
    def __init__(self, data_list, tokenizer, max_text_length=2048, choose_response: str = "icd10"):
        self.dataset = data_list
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.response = choose_response
        # Memoized lookup: code -> hierarchical string (populated lazily, dict hit afterwards).
        self._hier_lookup = {}
        self._missing_warned = set()
        # The UNIFIED tokenizer's vocab is ground truth for which level tokens exist as
        # single stable IDs (it is the output of build_hierarchical_icd10_vocab.py).
        vocab = tokenizer.get_vocab() if tokenizer is not None else {}
        self._chap_tokens = {t for t in vocab if t.startswith("CHAP_")}
        self._block_tokens = {t for t in vocab if t.startswith("BLOCK_")}
        self._cat_tokens = {t for t in vocab if t.startswith("CAT_")}
        self._subcat_tokens = {t for t in vocab if t.startswith("SUBCAT_")}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        system_prompt = item.get("system", "")
        user_prompt = item.get("user", "")
        
        assistant_full = item.get("assistant", "")
        codes = item.get("codes", [])
        codes_str = ", ".join(codes)
        # Split assistant response
        assistant_response_icd10 = ""
        assistant_response_explanation = ""
        if codes_str and codes_str + "\n\n" in assistant_full:
            parts = assistant_full.split(codes_str + "\n\n", 1)
            assistant_response_icd10 = parts[0]
            assistant_response_explanation = parts[1] if len(parts) > 1 else ""
        else:
            assistant_response_icd10 = assistant_full
            assistant_response_explanation = assistant_full

        if self.response == "icd10_custom" and codes:
            codes_str = ", ".join(self._code_to_hierarchical(c) for c in codes)

        # Build conversation - hierarchy tokens are matched LITERALLY by the unified
        # tokenizer's added-vocabulary and encoded as single tokens.
        if self.response in ("icd10", "icd10_custom"):
            assistant_response = assistant_response_icd10 + codes_str
        else:
            assistant_response = assistant_response_explanation
        conversation = f"### Question:\n{user_prompt}\n\n### Answer:\n{assistant_response}"
        text_encoding = self.tokenizer(
            conversation,
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

    def _code_to_hierarchical(self, code: str) -> str:
        """A02.0 → 'CHAP_A BLOCK_A02 CAT_A02 SUBCAT_0'. Components are validated against
        the unified tokenizer's vocab (single-ID guarantee). Codes with missing level
        tokens (not covered by the tokenizer's training set) are still constructed, but
        Gemma's BPE will sub-decompose those pieces — warn once per code."""
        code = str(code).strip().upper()
        if not code:
            return ""
        cached = self._hier_lookup.get(code)
        if cached is not None:
            return cached
        if '.' in code:
            cat, sub = code.split('.', 1)
        else:
            cat, sub = code, ''
        result = " ".join(self._code_to_level_tokens(cat, sub))
        self._hier_lookup[code] = result
        return result

    def _code_to_level_tokens(self, cat: str, sub: str) -> List[str]:
        tokens = [
            f"CHAP_{cat[0] if cat else ''}",
            f"BLOCK_{cat[:3]}",
            f"CAT_{cat}",
        ]
        if sub:
            tokens.append(f"SUBCAT_{sub}")
        # Validate against vocab-derived token sets (O(1) membership)
        for level_tokens, tok in ((self._chap_tokens, tokens[0]), (self._block_tokens, tokens[1]),
                                  (self._cat_tokens, tokens[2])):
            if level_tokens and tok not in level_tokens:
                if tok not in self._missing_warned:
                    self._missing_warned.add(tok)
                    print(f"[hierarchy] '{tok}' not in unified vocab - "
                          f"codes using it will fall back to sub-token pieces. "
                          f"Rebuild with --codes_source union for full coverage.")
        if sub and self._subcat_tokens and f"SUBCAT_{sub}" not in self._subcat_tokens:
            if f"SUBCAT_{sub}" not in self._missing_warned:
                self._missing_warned.add(f"SUBCAT_{sub}")
                print(f"[hierarchy] 'SUBCAT_{sub}' not in unified vocab - "
                      f"codes using it will fall back to sub-token pieces. "
                      f"Rebuild with --codes_source union for full coverage.")
        return tokens


# ============================================================
# MODEL WRAPPER (Simple - Just MedGemma + LoRA)
# ============================================================

class MedGemmaUnified(nn.Module):
    """
    Standard MedGemma with LoRA. No extra modules!
    Unified tokenizer handles everything.
    """
    def __init__(self, model_name, lora_config=None, target_vocab_size=None):
        super().__init__()
        
        print(f"Loading {model_name} with Unsloth...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None, 
            load_in_4bit=True,
        )
        
        # CRITICAL: resize embeddings BEFORE get_peft_model. With
        # modules_to_save=["embed_tokens"], PEFT deep-copies the embedding module at
        # wrap time; resizing the base model afterwards would leave that copy at the
        # old vocab size and out-of-range token IDs would crash the forward pass.
        if target_vocab_size is not None and target_vocab_size != self.model.config.vocab_size:
            print(f"Resizing embeddings: {self.model.config.vocab_size} → {target_vocab_size}")
            self.model.resize_token_embeddings(target_vocab_size)
        
        if lora_config is None:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                # REQUIRED for the unified tokenizer: embed_tokens needs resizing to the
                # unified vocab and its new rows are randomly initialized. Without
                # modules_to_save, the embedding stays frozen under LoRA and the new
                # ICD hierarchy tokens can never be learned. Gemma-3 ties lm_head to
                # embed_tokens, so saving embed_tokens covers both.
                modules_to_save=["embed_tokens"],
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
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )


# ============================================================
# TRAINER
# ============================================================

class UnifiedTrainer:
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
        
        print(f"Saved checkpoint to {output_dir}")


# ============================================================
# INFERENCE HELPER
# ============================================================

def generate_icd10_codes(model, tokenizer, user_prompt: str, system_prompt: str = "",
                         max_new_tokens: int = 512, temperature: float = 0.7):
    """Generate ICD-10 codes using the unified tokenizer."""
    conversation = f"### Question:\n{user_prompt}\n\n### Answer:\n"
    
    inputs = tokenizer(
        conversation,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
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
    
    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


# ============================================================
# MAIN
# 
# 
# read datasetloader once again yourself : VERIFIED
# continue to line 565!!
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train MedGemma with Unified Hierarchical ICD-10 Tokenizer")
    parser.add_argument('--data_source', type=str, default="synth", choices=["synth", "jsonl"],
                        help="synth = FiscaAI/synth-ehr-icd10cm-prompt from HF hub, jsonl = --data_path file")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--synth_dataset', type=str, default="FiscaAI/synth-ehr-icd10cm-prompt")
    parser.add_argument('--synth_cache_dir', type=str, default="FiscaAI___synth-ehr-icd10cm-prompt")
    parser.add_argument('--unified_tokenizer_dir', type=str, 
                        help="Path to unified tokenizer from build_hierarchical_icd10_vocab.py")
    parser.add_argument('--model_name', type=str, default="unsloth/gemma-3-4b-it-bnb-4bit")
    parser.add_argument('--output_dir', type=str, 
                        )
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_text_length', type=int, default=2048)
    parser.add_argument('--response_mode', type=str, default="icd10_custom", choices=["icd10_custom", "explanation","icd10"])
    parser.add_argument('--icd10_tokenizer_path', type=str, default=None,
                        help="DEPRECATED/IGNORED: the standalone ICD-10 BPE is only a build-time "
                             "artifact; at runtime the unified tokenizer contains all hierarchy tokens.")
    args = parser.parse_args()

    wandb.init(project="medgemma-unified-icd10", config=vars(args))

    if args.data_source == "synth":
        from datasets import load_dataset, load_from_disk
        dataset = None
        try:
            dataset = load_dataset(args.synth_dataset, cache_dir=args.synth_cache_dir)
            print(f"Synth dataset loaded from HF hub: {args.synth_dataset}")
        except Exception as e:
            print(f"load_dataset failed ({e}), trying load_from_disk fallback...")
            try:
                dataset = load_from_disk(args.synth_cache_dir)
                print(f"Synth dataset loaded via load_from_disk: {args.synth_cache_dir}")
            except Exception as e2:
                raise RuntimeError(f"Could not load synth dataset from hub or disk: {e2}")
        split = 'train' if 'train' in dataset else list(dataset.keys())[0]
        all_data = list(dataset[split])
        # Normalize 'codes' (HF rows arrive as lists; be defensive about list-formatted strings)
        for item in all_data:
            c = item.get("codes")
            if isinstance(c, str):
                c = c.strip().lstrip('[').rstrip(']')
                item["codes"] = [x.strip(" '\"") for x in c.split(',') if x.strip(" '\"")]
            elif not isinstance(c, list):
                item["codes"] = []
        print(f"Synth dataset rows: {len(all_data)}")
    else:
        print(f"Loading data from {args.data_path}...")
        with open(args.data_path, "r", encoding="utf-8") as f:
            all_data = [json.loads(line) for line in f]
    
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    print(f"Loaded {len(train_data)} training samples and {len(eval_data)} evaluation samples.")

    # Load UNIFIED tokenizer (Gemma + ICD-10 hierarchical BPE)
    print(f"Loading unified tokenizer from {args.unified_tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.unified_tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Unified tokenizer vocab size: {len(tokenizer)}")
    
    # Show ICD-10 hierarchy tokens in vocab
    icd_tokens = [t for t in tokenizer.get_vocab().keys() if t.startswith(('CHAP_', 'BLOCK_', 'CAT_', 'SUBCAT_', '<ICD_'))] 
    print(f"ICD-10 hierarchy tokens in vocab: {len(icd_tokens)}")
    print(f"Examples: {icd_tokens[:10]}")

    # NOTE: No standalone ICD-10 tokenizer is loaded here. The hierarchy tokens
    # (CHAP_/BLOCK_/CAT_/SUBCAT_) live in the unified vocab, and the dataset builds
    # the hierarchical strings with plain string ops. icd10_hierarchical_bpe.json
    # is only a build-time artifact of build_hierarchical_icd10_vocab.py.

    # # DEBUG: Show input text and tokenization for one random sample
    # print("\n" + "="*80)
    # print("DEBUG: TOKENIZATION SAMPLE")
    # print("="*80)
    # import random
    # sample_idx = random.randint(0, len(train_data) - 1)
    # sample = train_data[sample_idx]
    # print(f"Sample index: {sample_idx}")
    # print(f"System: {sample.get('system', '')[:200]}...")
    # print(f"User: {sample.get('user', '')[:200]}...")
    # print(f"Assistant: {sample.get('assistant', '')[:200]}...")
    # print(f"Codes: {sample.get('codes', [])}")
    
    # # Reconstruct conversation like dataset does
    # system_prompt = sample.get("system", "")
    # user_prompt = sample.get("user", "")
    # assistant_full = sample.get("assistant", "")
    # codes = sample.get("codes", [])
    # codes_str = "".join(codes) # hope its not empty


    # #this should check if code is tokenizing but it will not work properly with normal tokenizers that fine for me
    # if icd10_tokenizer is not None and codes:
    #     hierarchical_codes = []
    #     for code in codes:
    #         tokens = code_to_hierarchical_tokens(code, icd10_tokenizer)
    #         hierarchical_codes.append(" ".join(tokens))
    #     codes_str = ", ".join(hierarchical_codes)
    #     print(f"\nHierarchical codes_str: {codes_str}")
    
    # assistant_response_icd10 = ""
    # assistant_response_explanation = ""
    # if codes_str and codes_str + "\n\n" in assistant_full:
    #     parts = assistant_full.split(codes_str + "\n\n", 1)
    #     assistant_response_icd10 = parts[0] + codes_str
    #     assistant_response_explanation = parts[1] if len(parts) > 1 else ""
    # else:
    #     assistant_response_icd10 = assistant_full
    #     assistant_response_explanation = assistant_full
    
    # assistant_response = assistant_response_icd10 if args.response_mode == "icd10_custom" or args.response_mode == "icd10" else assistant_response_explanation
    # conversation = f"### Question:\n{user_prompt}\n\n### Answer:\n{assistant_response}"
    
    # print(f"\n--- INPUT TEXT (conversation) ---")
    # print(conversation[:1000])
    # print("..." if len(conversation) > 1000 else "")
    
    # # Tokenize
    # text_encoding = tokenizer(
    #     conversation,
    #     truncation=True,
    #     max_length=args.max_text_length,
    #     padding="max_length",
    #     return_tensors="pt"
    # )
    
    # input_ids = text_encoding["input_ids"].squeeze(0).tolist()
    # tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # print(f"\n--- TOKENS (first 10) ---")
    # for i, (tid, tok) in enumerate(zip(input_ids[:10], tokens[:10])):
    #     print(f"  [{i}] ID={tid:6d} → {tok}")
    
    # print(f"\n--- TOKENS (all non-pad) ---")
    # non_pad = [(tid, tok) for tid, tok in zip(input_ids, tokens) if tid != tokenizer.pad_token_id]
    # for i, (tid, tok) in enumerate(non_pad):
    #     print(f"  [{i}] ID={tid:6d} → {tok}")
    
    # print(f"\nTotal tokens: {len(input_ids)}, Non-pad: {len(non_pad)}")
    # print("="*80)
    # import sys
    # sys.exit(0)  # COMMENT THIS OUT TO RUN FULL TRAINING
    
    # def code_to_hierarchical_tokens(code: str, icd10_tokenizer) -> List[str]:
    #     encoding = icd10_tokenizer.encode(code)
    #     return [icd10_tokenizer.id_to_token(id_) for id_ in encoding.ids 
    #             if icd10_tokenizer.id_to_token(id_) not in ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]]
    '''
    Review UnifiedDataset !!
    '''
    # Datasets
    print("Preparing datasets...")
    train_dataset = UnifiedICD10Dataset(
        train_data, tokenizer, 
        max_text_length=args.max_text_length, choose_response=args.response_mode
    )
    eval_dataset = UnifiedICD10Dataset(
        eval_data, tokenizer, 
        max_text_length=args.max_text_length, choose_response=args.response_mode
    )

    # Model (resizes embeddings to the unified vocab inside __init__, before PEFT wrapping)
    print("Initializing model...")
    model = MedGemmaUnified(model_name=args.model_name, target_vocab_size=len(tokenizer))
    model.tokenizer = tokenizer

    # Trainer
    trainer = UnifiedTrainer(model, train_dataset, eval_dataset, args)

    # Train
    print("Starting training...")
    trainer.train()

    # Final save
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(final_dir / "lora_adapters")
    model.tokenizer.save_pretrained(final_dir / "tokenizer")

    print(f"Training complete! Model saved to {final_dir}")
    wandb.finish()


if __name__ == '__main__':
    main()