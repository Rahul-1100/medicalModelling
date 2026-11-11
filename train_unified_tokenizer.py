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
# Point HF_HOME to your own directory instead of read-only /mnt/data
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# CRITICAL: Import unsloth BEFORE transformers and peft!
import unsloth
from unsloth import FastLanguageModel

import json
import random
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
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
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.response = choose_response

        # Filter out malformed rows before training starts
        valid_data = []
        skipped = 0
        for item in data_list:
            codes = item.get("codes", [])
            assistant_full = item.get("assistant", "")
            codes_str = ", ".join(codes)

            if codes and (codes_str + "\n\n" in assistant_full):
                valid_data.append(item)
            else:
                skipped += 1

        if skipped > 0:
            print(f"[Dataset Filter] Skipped {skipped}/{len(data_list)} malformed samples "
                  f"(missing codes or 'codes_str + \\n\\n' delimiter).")

        self.dataset = valid_data if valid_data else data_list

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

        # Split assistant response safely
        assistant_response_icd10 = ""
        assistant_response_explanation = ""
        if codes_str and codes_str + "\n\n" in assistant_full:
            parts = assistant_full.split(codes_str + "\n\n", 1)
            assistant_response_icd10 = parts[0]
            assistant_response_explanation = parts[1] if len(parts) > 1 else ""
        else:
            # Safe fallback: keep assistant_response_icd10 empty so codes_str is not duplicated
            assistant_response_icd10 = ""
            assistant_response_explanation = assistant_full

        if self.response == "icd10_custom" and codes:
            codes_str = ", ".join(self._code_to_hierarchical(c) for c in codes)

        # Build conversation - hierarchy tokens are matched LITERALLY by the unified
        # tokenizer's added-vocabulary and encoded as single tokens.
        if self.response in ("icd10", "icd10_custom"):
            assistant_response = assistant_response_icd10 + codes_str
        else:
            assistant_response = assistant_response_explanation

        prompt = f"### Question:\n{user_prompt}\n\n### Answer:\n"
        conversation = prompt + assistant_response
        text_encoding = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = text_encoding["input_ids"].squeeze(0)
        attention_mask = text_encoding["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": self._build_labels(prompt, input_ids, attention_mask)
        }

    def _build_labels(self, prompt: str, input_ids, attention_mask):
        """Loss only on the assistant response: prompt region and padding are masked
        with -100 so the model is not trained to re-generate the question text or pads."""
        labels = input_ids.clone()
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, truncation=True, max_length=self.max_text_length
        )["input_ids"]
        # Longest common prefix: the prompt's token ids must line up with the start of the
        # conversation ids. If BPE merges differ at the boundary (rare), fall back to the
        # shared prefix length instead of masking the wrong region.
        n = min(len(prompt_ids), len(input_ids))
        prefix = 0
        for i in range(n):
            if prompt_ids[i] == input_ids[i]:
                prefix += 1
            else:
                break
        labels[:prefix] = -100
        labels[attention_mask == 0] = -100
        # Guard: if truncation swallowed the whole response, every label would be -100
        # and cross-entropy over an empty target set yields NaN. Un-mask the last real
        # token so the sample still contributes a valid loss term.
        if (labels != -100).sum() == 0:
            non_pad = attention_mask.nonzero()
            if len(non_pad) > 0:
                labels[non_pad[-1].item()] = input_ids[non_pad[-1].item()]
        return labels

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
    def __init__(self, model_name, lora_config=None, target_vocab_size=None, resume_checkpoint=None):
        super().__init__()
        print(f"Loading {model_name} with Unsloth...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        # 1. Safely get current vocab size directly from the input embedding matrix
        current_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        if target_vocab_size is not None and target_vocab_size != current_vocab_size:
            print(f"Resizing embeddings: {current_vocab_size} -> {target_vocab_size}")
            self.model.resize_token_embeddings(target_vocab_size)
            # Update config in both places so PEFT/Unsloth see the new size
            if hasattr(self.model.config, "text_config"):
                self.model.config.text_config.vocab_size = target_vocab_size
            self.model.config.vocab_size = target_vocab_size

        # 2. Attach LoRA adapters with Unsloth
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                            "embed_tokens", "lm_head"],
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        # 3. If resuming from a 30k-step checkpoint, load the saved adapter weights
        if resume_checkpoint is not None:
            adapter_dir = Path(resume_checkpoint) / "lora_adapters"
            if adapter_dir.exists():
                print(f"Loading saved LoRA adapter weights from {adapter_dir}...")
                self.model.load_adapter(str(adapter_dir), adapter_name="default")
            else:
                print(f"[Warning] No lora_adapters directory found at {adapter_dir}")

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
    GEN_EVAL_SAMPLES = 64
    GEN_MAX_NEW_TOKENS = 96

    def __init__(self, model, train_dataset, eval_dataset, args):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args
        self.tokenizer = getattr(model, "tokenizer", None)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        num_training_steps = max(
            1, len(train_dataset) * args.num_epochs // (args.batch_size * args.gradient_accumulation)
        )
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
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    def train(self):
        test = True
        train_start_time = time.time()
        self.model.train()

        steps_per_epoch = len(self.train_loader)
        total_steps = self.args.num_epochs * steps_per_epoch

        # Resume optimizer, scheduler, scaler, and step counters if --resume_from_checkpoint is set
        start_epoch = 0
        resume_step = 0
        if getattr(self.args, "resume_from_checkpoint", None):
            ckpt_path = Path(self.args.resume_from_checkpoint)
            state_file = ckpt_path / "trainer_state.pt"
            if state_file.exists():
                print(f"Resuming optimizer/scheduler state from {state_file}...")
                state = torch.load(state_file, map_location=self.device)
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.scaler.load_state_dict(state["scaler"])
                start_epoch = state.get("epoch", 0)
                resume_step = (state.get("step") or 0) + 1
                if resume_step >= steps_per_epoch:
                    start_epoch += 1
                    resume_step = 0
                print(f"Resuming from Epoch {start_epoch + 1}, Step {resume_step}/{steps_per_epoch}")

        for epoch in range(start_epoch, self.args.num_epochs):
            total_loss = 0
            epoch_start_time = time.time()

            for step, batch in enumerate(self.train_loader):
                # Skip already-completed steps when resuming mid-epoch
                if epoch == start_epoch and step < resume_step:
                    continue

                if test:
                    print("Input: ", batch["input_ids"][0])
                    print("label: ", batch["labels"][0])
                    non_pad_inputs = [id_ for id_ in batch["input_ids"][0].tolist() if id_ != self.tokenizer.pad_token_id]
                    print("\nDecoded Input Text:\n", self.tokenizer.decode(non_pad_inputs))
                    valid_label_ids = [id_ for id_ in batch["labels"][0].tolist() if id_ != -100]
                    print("\nDecoded Target Label Text:\n", self.tokenizer.decode(valid_label_ids))
                    print("\nTarget Label Tokens:\n", self.tokenizer.convert_ids_to_tokens(valid_label_ids))
                    test = False

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
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

                step_loss = loss.item() * self.args.gradient_accumulation
                total_loss += step_loss
                global_step = epoch * steps_per_epoch + step + 1

                if step % 10 == 0 or (step + 1) == steps_per_epoch:
                    now = time.time()
                    epoch_done_pct = ((step + 1) / steps_per_epoch) * 100
                    epoch_elapsed = now - epoch_start_time
                    steps_in_this_epoch = max(1, (step + 1 - (resume_step if epoch == start_epoch else 0)))
                    avg_step_time_epoch = epoch_elapsed / steps_in_this_epoch
                    est_single_epoch_time = avg_step_time_epoch * steps_per_epoch
                    epoch_eta = avg_step_time_epoch * (steps_per_epoch - (step + 1))

                    total_elapsed = now - train_start_time
                    steps_done_this_run = max(1, global_step - (start_epoch * steps_per_epoch + resume_step))
                    avg_step_time_total = total_elapsed / steps_done_this_run
                    total_eta = avg_step_time_total * (total_steps - global_step)

                    print(
                        f"Epoch {epoch + 1}/{self.args.num_epochs} "
                        f"[{step + 1}/{steps_per_epoch} ({epoch_done_pct:.1f}%)] | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {step_loss:.4f} | "
                        f"Epoch Time: {self._format_time(epoch_elapsed)}/{self._format_time(est_single_epoch_time)} "
                        f"(ETA: {self._format_time(epoch_eta)}) | "
                        f"Total ETA: {self._format_time(total_eta)}"
                    )
                    if wandb.run:
                        wandb.log({
                            "train/loss": step_loss,
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/epoch_progress_pct": epoch_done_pct,
                            "train/est_epoch_minutes": est_single_epoch_time / 60,
                            "train/total_eta_minutes": total_eta / 60,
                        })

                # Save checkpoint every N steps (default: 30,000)
                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_checkpoint(epoch, step=step, global_step=global_step)

            epoch_train_time = time.time() - epoch_start_time
            active_epoch_steps = max(1, (steps_per_epoch - (resume_step if epoch == start_epoch else 0)))
            avg_loss = total_loss / active_epoch_steps
            print(
                f"\n---> Epoch {epoch + 1} finished in {self._format_time(epoch_train_time)} "
                f"({epoch_train_time:.1f}s) | Average Loss: {avg_loss:.4f}\n"
            )
            if wandb.run:
                wandb.log({"train/epoch_train_time_sec": epoch_train_time, "epoch": epoch + 1})

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
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    total_loss += outputs["loss"].item()
        avg_loss = total_loss / len(self.eval_loader)
        print(f"Epoch {epoch + 1} eval loss: {avg_loss:.4f}")
        metrics = {"eval/loss": avg_loss}
        if self.args.response_mode in ("icd10", "icd10_custom") and self.tokenizer is not None:
            metrics.update(self.evaluate_generation(epoch))
        if wandb.run:
            wandb.log(metrics)
        self.model.train()
        return avg_loss

    def evaluate_generation(self, epoch, num_samples=GEN_EVAL_SAMPLES, max_new_tokens=GEN_MAX_NEW_TOKENS):
        self.model.eval()
        subset = list(self.eval_dataset.dataset)[:num_samples]

        all_tp = all_fp = all_fn = 0
        per_code = {}   # code -> [tp, fp, fn]
        exact_matches = 0

        for item in subset:
            user_prompt = item.get("user", "")
            gold_codes = {str(c).strip().upper() for c in item.get("codes", []) if c}
            pred_codes = self._generate_codes(user_prompt, max_new_tokens)
            same = {c.upper() for c in pred_codes}

            if same == gold_codes:
                exact_matches += 1
            for c in same - gold_codes:
                per_code.setdefault(c, [0, 0, 0]); per_code[c][1] += 1
                all_fp += 1
            for c in gold_codes - same:
                per_code.setdefault(c, [0, 0, 0]); per_code[c][2] += 1
                all_fn += 1
            for c in same & gold_codes:
                per_code.setdefault(c, [0, 0, 0]); per_code[c][0] += 1
                all_tp += 1

        micro_p = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
        micro_r = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
        f1s = [tp / (tp + fp / 2 + fn / 2) for tp, fp, fn in per_code.values()
               if (tp + fp / 2 + fn / 2) > 0]
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        exact_match = exact_matches / len(subset) if subset else 0.0

        print(f"Epoch {epoch + 1} gen metrics (n={len(subset)}): "
              f"micro-F1={micro_f1:.4f} macro-F1={macro_f1:.4f} exact-set-match={exact_match:.4f}")
        return {
            "eval/gen_micro_f1": micro_f1,
            "eval/gen_macro_f1": macro_f1,
            "eval/gen_exact_match": exact_match,
        }

    def _generate_codes(self, user_prompt: str, max_new_tokens: int):
        prompt = f"### Question:\n{user_prompt}\n\n### Answer:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=self.amp_dtype):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][input_ids.shape[1]:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)

        codes, cur, cat, sub = [], [], "", ""
        for t in tokens:
            t = str(t).lstrip("▁ ")
            if not any(t.startswith(p) for p in ("CHAP_", "BLOCK_", "CAT_", "SUBCAT_")):
                continue
            if t.startswith("CHAP_") and cur:
                if cat:
                    codes.append(f"{cat}.{sub}" if sub else cat)
                cur, cat, sub = [], "", ""
            cur.append(t)
            if t.startswith("CAT_"):
                cat = t[4:]
            elif t.startswith("SUBCAT_"):
                sub = t[7:]
        if cur and cat:
            codes.append(f"{cat}.{sub}" if sub else cat)
        return codes

    def save_checkpoint(self, epoch, step=None, global_step=None):
        if global_step is not None:
            folder_name = f"checkpoint-step-{global_step}"
        else:
            folder_name = f"checkpoint-epoch-{epoch + 1}"

        output_dir = Path(self.args.output_dir) / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save LoRA adapters & unified tokenizer
        self.model.model.save_pretrained(output_dir / "lora_adapters", save_embedding_layers=True)
        self.model.tokenizer.save_pretrained(output_dir / "tokenizer")

        # 2. Save trainer state so you can resume after the 2-day limit
        state = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(state, output_dir / "trainer_state.pt")

        print(f"\n[Checkpoint] Saved model & trainer state to {output_dir}\n")


# ============================================================
# INFERENCE HELPER
# ============================================================
def generate_icd10_codes(model, tokenizer, user_prompt: str, system_prompt: str = "",
                         max_new_tokens: int = 512, temperature: float = 0.7):
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
    parser.add_argument('--output_dir', type=str, default="./checkpoints_unified")
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_text_length', type=int, default=2048)
    parser.add_argument('--save_steps', type=int, default=30000,
                        help="Save a checkpoint every N training steps (default: 30000)")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help="Path to a checkpoint folder (e.g., ./checkpoints_unified/checkpoint-step-30000) to resume")
    parser.add_argument('--response_mode', type=str, default="icd10_custom",
                        choices=["icd10_custom", "explanation", "icd10"])
    parser.add_argument('--icd10_tokenizer_path', type=str, default=None,
                        help="DEPRECATED/IGNORED: the standalone ICD-10 BPE is only a build-time artifact")
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

    # Fix seed so train/eval split is identical if resuming after 2-day limit
    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    print(f"Loaded {len(train_data)} training samples and {len(eval_data)} evaluation samples.")

    print(f"Loading unified tokenizer from {args.unified_tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.unified_tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Unified tokenizer vocab size: {len(tokenizer)}")
    icd_tokens = [t for t in tokenizer.get_vocab().keys() if t.startswith(('CHAP_', 'BLOCK_', 'CAT_', 'SUBCAT_', '<ICD_'))]
    print(f"ICD-10 hierarchy tokens in vocab: {len(icd_tokens)}")
    print(f"Examples: {icd_tokens[:10]}")

    print("Preparing datasets...")
    train_dataset = UnifiedICD10Dataset(
        train_data, tokenizer,
        max_text_length=args.max_text_length, choose_response=args.response_mode
    )
    eval_dataset = UnifiedICD10Dataset(
        eval_data, tokenizer,
        max_text_length=args.max_text_length, choose_response=args.response_mode
    )

    print("Initializing model...")
    model = MedGemmaUnified(
        model_name=args.model_name,
        target_vocab_size=len(tokenizer),
        resume_checkpoint=args.resume_from_checkpoint
    )
    model.tokenizer = tokenizer

    # print("Testing checkpoint save before training...")
    # try:
    #     model.model.save_pretrained("./dummy_test_save", save_embedding_layers=True)
    #     print("✅ SUCCESS! The save fix works. Starting training now...")
    # except Exception as e:
    #     print(f"❌ FAIL: The save still crashed. Error: {e}")

    trainer = UnifiedTrainer(model, train_dataset, eval_dataset, args)

    print("Starting training...")
    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(final_dir / "lora_adapters",save_embedding_layers=True)
    model.tokenizer.save_pretrained(final_dir / "tokenizer")

    print(f"Training complete! Model saved to {final_dir}")
    wandb.finish()


if __name__ == '__main__':
    main()