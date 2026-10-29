#!/usr/bin/env python3
"""
build_hierarchical_icd10_vocab.py

Build a compact ICD-10 vocabulary using hierarchical BPE.
Strategy:
1. Parse all ICD-10 codes into hierarchical components (chapter, block, category, subcategory)
   Code sources: FiscaAI/synth-ehr-icd10cm-prompt HF dataset (codes column),
   CSV/TXT registry files, or the union of both (--codes_source)
2. Train BPE on these components to learn shared prefixes
3. Merge with Gemma tokenizer vocabulary
4. Save unified tokenizer for training

This exploits ICD-10 structure: A00.0, A00.1, A00.9 share "A00" + ".0/.1/.9"
"""

import os
import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set, Tuple

import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


# ============================================================
# ICD-10 HIERARCHICAL DECOMPOSITION
# ============================================================

ICD10_PATTERN = re.compile(r'^([A-Z])(\d{2})(\.?)(\d{1,4}[A-TV-Z]?)$')

def decompose_icd10_code(code: str) -> Dict[str, str]:
    """
    Decompose ICD-10 code into hierarchical components.
    
    Examples:
        A00.0   → {'chapter': 'A', 'block': 'A00', 'category': 'A00', 'subcategory': '0'}
        A01.01  → {'chapter': 'A', 'block': 'A00', 'category': 'A01', 'subcategory': '01'}
        T80.40xA → {'chapter': 'T', 'block': 'T80', 'category': 'T80', 'subcategory': '40xA'}
        Z23     → {'chapter': 'Z', 'block': 'Z23', 'category': 'Z23', 'subcategory': ''}
    """
    code = str(code).strip().upper()
    
    # Handle codes without dots (category level only)
    if '.' not in code:
        if len(code) >= 3:
            return {
                'chapter': code[0],
                'block': code[:3],
                'category': code[:3],
                'subcategory': ''
            }
        return {'chapter': code[0] if code else '', 'block': '', 'category': '', 'subcategory': ''}
    
    category_part, subcategory_part = code.split('.', 1)
    
    # Chapter: first letter (A-TV-Z)
    chapter = category_part[0] if category_part else ''
    
    # Block: first 3 chars (e.g., A00, T80, Z23)
    block = category_part[:3] if len(category_part) >= 3 else category_part
    
    # Category: same as block for ICD-10-CM
    category = category_part
    
    return {
        'chapter': chapter,
        'block': block,
        'category': category,
        'subcategory': subcategory_part
    }


def code_to_hierarchical_tokens(code: str) -> List[str]:
    """
    Convert ICD-10 code to a sequence of hierarchical tokens for BPE training.
    
    Uses special markers to preserve hierarchy boundaries:
    A00.0  → ['CHAP_A', 'BLOCK_A00', 'CAT_A00', 'SUBCAT_0']
    T80.40xA → ['CHAP_T', 'BLOCK_T80', 'CAT_T80', 'SUBCAT_40xA']
    Z23    → ['CHAP_Z', 'BLOCK_Z23', 'CAT_Z23']

    NOTE: Markers use "PREFIX_level" format, NOT "<CHAP>A"-style. The tokenizers
    BpeTrainer treats '<' and '>' as special-token delimiters and forbids any
    learned merge containing them, so angle-bracket compounds can never become
    single BPE tokens.
    """
    parts = decompose_icd10_code(code)
    tokens = []
    
    if parts['chapter']:
        tokens.append(f"CHAP_{parts['chapter']}")
    if parts['block']:
        tokens.append(f"BLOCK_{parts['block']}")
    if parts['category']:
        tokens.append(f"CAT_{parts['category']}")
    if parts['subcategory']:
        tokens.append(f"SUBCAT_{parts['subcategory']}")
    
    return tokens


# ============================================================
# BPE TRAINING ON HIERARCHICAL COMPONENTS
# ============================================================

def train_hierarchical_bpe(
    codes: List[str],
    vocab_size: int = 8000,
    min_frequency: int = 2,
    special_tokens: List[str] = None
) -> Tokenizer:
    """
    Train BPE tokenizer on hierarchical ICD-10 components.
    
    Args:
        codes: List of ICD-10 codes
        vocab_size: Target vocabulary size (including special tokens)
        min_frequency: Minimum frequency for a merge
        special_tokens: Special tokens to include
    
    Returns:
        Trained tokenizers.Tokenizer object
    """
    if special_tokens is None:
        # NOTE: Do NOT include CHAP/BLOCK/CAT/SUBCAT markers here. Also never use
        # '<'/'>' characters in marker names: the tokenizers BpeTrainer treats
        # '<' and '>' as special-token delimiters and forbids merges containing
        # them, so compounds like "<CHAP>A" can never be learned as one token.
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>",
                          "<ICD_START>", "<ICD_END>"]
    
    # Decompose all codes into hierarchical token sequences
    print(f"Decomposing {len(codes)} ICD-10 codes...")
    all_sequences = []
    for code in codes:
        tokens = code_to_hierarchical_tokens(code)
        all_sequences.append(" ".join(tokens))
    
    # Show some examples
    print("\nExample decompositions:")
    for code in codes[:10]:
        tokens = code_to_hierarchical_tokens(code)
        print(f"  {code:12s} → {' '.join(tokens)}")
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    
    # Pre-tokenizer: split on whitespace (our hierarchical tokens are space-separated)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Decoder
    tokenizer.decoder = decoders.BPEDecoder()
    
    # Post-processor for special tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<BOS> $A <EOS>",
        pair="<BOS> $A <EOS> $B <EOS>",
        special_tokens=[("<BOS>", 2), ("<EOS>", 3)]
    )
    
    # Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Train from iterator
    print(f"\nTraining BPE with vocab_size={vocab_size}, min_frequency={min_frequency}...")
    tokenizer.train_from_iterator(all_sequences, trainer=trainer)
    
    print(f"Trained vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer


# ============================================================
# MERGE WITH GEMMA TOKENIZER
# ============================================================

def merge_with_gemma_tokenizer(
    icd10_tokenizer: Tokenizer,
    gemma_model_name: str = "unsloth/gemma-3-4b-it-bnb-4bit",
    output_dir: str = "./unified_icd10_gemma_tokenizer"
) -> PreTrainedTokenizerFast:
    """
    Merge ICD-10 hierarchical BPE vocabulary with Gemma tokenizer.
    
    Strategy:
    1. Load Gemma tokenizer
    2. Extract ICD-10 tokens that don't exist in Gemma
    3. Add them as new tokens to Gemma
    4. Save unified tokenizer
    """
    print(f"\nLoading Gemma tokenizer: {gemma_model_name}")
    gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
    
    if gemma_tokenizer.pad_token is None:
        gemma_tokenizer.pad_token = gemma_tokenizer.eos_token
    
    gemma_vocab = set(gemma_tokenizer.get_vocab().keys())
    print(f"Gemma vocab size: {len(gemma_vocab)}")
    
    # Get ICD-10 vocabulary
    icd10_vocab = set(icd10_tokenizer.get_vocab().keys())
    print(f"ICD-10 BPE vocab size: {len(icd10_vocab)}")
    
    # Find new tokens to add
    new_tokens = sorted([t for t in icd10_vocab if t not in gemma_vocab])
    print(f"New tokens to add: {len(new_tokens)}")
    
    # Show some examples
    print("\nExample new tokens:")
    for t in new_tokens[:20]:
        print(f"  {t}")
    if len(new_tokens) > 20:
        print(f"  ... and {len(new_tokens) - 20} more")
    
    # Add to Gemma tokenizer
    print("\nAdding tokens to Gemma tokenizer...")
    num_added = gemma_tokenizer.add_tokens(new_tokens)
    print(f"Successfully added: {num_added} tokens")
    print(f"New Gemma vocab size: {len(gemma_tokenizer)}")
    
    # Save unified tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    gemma_tokenizer.save_pretrained(output_path)
    print(f"\nSaved unified tokenizer to {output_path}")
    
    # Also save the ICD-10 BPE tokenizer separately for reference
    icd10_tokenizer.save(str(output_path / "icd10_hierarchical_bpe.json"))
    
    # Save mapping for reference
    mapping = {
        "gemma_original_vocab_size": len(gemma_vocab),
        "icd10_bpe_vocab_size": len(icd10_vocab),
        "new_tokens_added": num_added,
        "final_vocab_size": len(gemma_tokenizer),
        "new_tokens": new_tokens
    }
    with open(output_path / "merge_info.json", "w") as f:
        json.dump(mapping, f, indent=2)
    
    return gemma_tokenizer


def normalize_code(code: str) -> str:
    """Normalize ICD-10 code to standard format (shared by all loaders)."""
    code = str(code).strip(" '\"").upper()
    if not code or code == 'NAN':
        return ''
    if '.' in code:
        return code
    if len(code) >= 3:
        category = code[:3]
        subcategory = code[3:]
        if subcategory:
            return f"{category}.{subcategory}"
        else:
            return category
    return code


def filter_valid_codes(codes: Set[str]) -> List[str]:
    valid = [c for c in codes if c and len(c) >= 3 and ICD10_PATTERN.match(c)]
    return sorted(valid)


# ============================================================
# LOAD CODES FROM YOUR DATA SOURCES
# ============================================================

def load_codes_from_synth_dataset(
    dataset_name: str = "FiscaAI/synth-ehr-icd10cm-prompt",
    cache_dir: str = "FiscaAI___synth-ehr-icd10cm-prompt",
    split: str = None
) -> List[str]:
    """
    Load ICD-10-CM codes from the SynthEHR HF dataset (system/user/assistant/codes).

    The `codes` column entries may be actual Python lists (when loaded from arrow)
    or list-formatted strings like "['M24.541', 'M24.542']" - both are handled.
    """
    from datasets import load_dataset, load_from_disk

    dataset = None
    try:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        print(f"Dataset loaded from HF hub: {dataset_name}")
    except Exception as e:
        print(f"load_dataset failed ({e}), trying load_from_disk fallback...")
        try:
            dataset = load_from_disk(cache_dir)
            print(f"Dataset loaded via load_from_disk: {cache_dir}")
        except Exception as e2:
            raise RuntimeError(f"Could not load dataset from hub or disk: {e2}")

    if split is None:
        split = 'train' if 'train' in dataset else list(dataset.keys())[0]
    df = pd.DataFrame(dataset[split])

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    if 'codes' not in df.columns:
        raise ValueError(f"'codes' column not found. Available: {df.columns.tolist()}")

    all_codes = set()
    for raw in df['codes']:
        if raw is None:
            continue
        if isinstance(raw, (list, tuple, set)):
            candidates = list(raw)
        else:
            # Parse list-formatted string: "['M24.541', 'M24.542']"
            s = str(raw).strip()
            s = s.lstrip('[').rstrip(']')
            candidates = [c.strip(" '\"") for c in s.split(',') if c.strip(" '\"")]
        for c in candidates:
            norm = normalize_code(c)
            if norm:
                all_codes.add(norm)

    valid_codes = filter_valid_codes(all_codes)
    print(f"Unique normalized codes from synth dataset: {len(valid_codes)}")
    print(f"Sample codes: {valid_codes[:20]}")
    return valid_codes


def load_all_icd10_codes(csv_path: str, txt_path: str) -> List[str]:
    """Load and normalize all ICD-10 codes from your CSV/TXT data sources."""
    all_codes = set()
    
    # Load CSV
    print(f"Loading CSV: {csv_path}")
    df_csv = pd.read_csv(csv_path, header=None, names=[
        'category', 'subcategory', 'full_code', 'short_desc', 'long_desc', 'shortest_desc'
    ], dtype=str)
    for col in df_csv.columns:
        df_csv[col] = df_csv[col].astype(str).str.strip()
    df_csv['subcategory'] = df_csv['subcategory'].replace(['', 'nan', 'None', 'NA'], '')
    df_csv['code'] = df_csv['full_code'].apply(normalize_code)
    csv_codes = df_csv['code'].dropna().unique()
    all_codes.update(csv_codes)
    print(f"  CSV codes: {len(csv_codes)}")
    
    # Load TXT
    print(f"Loading TXT: {txt_path}")
    mappings = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|', 2)
            if len(parts) >= 2:
                icd10 = parts[1].strip()
                all_codes.add(normalize_code(icd10))
    print(f"  TXT codes: {len(all_codes) - len(csv_codes)} new")
    
    # Filter valid codes
    valid_codes = filter_valid_codes(all_codes)
    print(f"\nTotal unique valid ICD-10 codes: {len(valid_codes)}")
    
    return valid_codes


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build hierarchical ICD-10 BPE vocabulary and merge with Gemma")
    parser.add_argument('--csv_path', type=str, default="./2/ICD10codes.csv")
    parser.add_argument('--txt_path', type=str, default="./2/icd9to10dictionary.txt")
    parser.add_argument('--gemma_model', type=str, default="unsloth/gemma-3-4b-it-bnb-4bit")
    parser.add_argument('--output_dir', type=str, default="./unified_icd10_gemma_tokenizer")
    parser.add_argument('--icd10_vocab_size', type=int, default=8000, 
                        help="Target vocab size for ICD-10 BPE (before merging)")
    parser.add_argument('--min_frequency', type=int, default=2)
    parser.add_argument('--codes_source', type=str, default="synth",
                        choices=["files", "synth", "union"],
                        help="Which codes to train BPE on: 'synth' = FiscaAI synth-ehr dataset only, "
                             "'files' = CSV/TXT registry only, 'union' = both (recommended)")
    parser.add_argument('--synth_dataset', type=str, default="FiscaAI/synth-ehr-icd10cm-prompt")
    parser.add_argument('--synth_cache_dir', type=str, default="FiscaAI___synth-ehr-icd10cm-prompt")
    parser.add_argument('--synth_split', type=str, default=None,
                        help="Split to use for synth dataset (default: 'train' if present)")
    args = parser.parse_args()
    
    # Load codes from selected source
    if args.codes_source == "synth":
        codes = load_codes_from_synth_dataset(
            dataset_name=args.synth_dataset,
            cache_dir=args.synth_cache_dir,
            split=args.synth_split
        )
    elif args.codes_source == "files":
        codes = load_all_icd10_codes(args.csv_path, args.txt_path)
    else:  # union - guarantees both training-data coverage and full-registry generalization
        synth_codes = load_codes_from_synth_dataset(
            dataset_name=args.synth_dataset,
            cache_dir=args.synth_cache_dir,
            split=args.synth_split
        )
        file_codes = load_all_icd10_codes(args.csv_path, args.txt_path)
        codes = filter_valid_codes(set(synth_codes) | set(file_codes))
        print(f"\nUnion of synth + file codes: {len(codes)} unique valid codes")
    
    # Train hierarchical BPE on ICD-10 codes
    icd10_tokenizer = train_hierarchical_bpe(
        codes,
        vocab_size=args.icd10_vocab_size,
        min_frequency=args.min_frequency
    )
    
    # Merge with Gemma
    unified_tokenizer = merge_with_gemma_tokenizer(
        icd10_tokenizer,
        gemma_model_name=args.gemma_model,
        output_dir=args.output_dir
    )
    
    # Test the unified tokenizer
    print("\n" + "="*60)
    print("TESTING UNIFIED TOKENIZER")
    print("="*60)
    
    test_codes = ["A00.0", "A00.1", "A01.01", "T80.40xA", "Z23", "I10", "E11.9"]
    test_text = "Patient diagnosed with A00.0 and I10, also has E11.9"
    
    print(f"\nTest text: {test_text}")
    encoding = unified_tokenizer(test_text, return_tensors="pt")
    tokens = unified_tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {encoding['input_ids'][0].tolist()}")
    
    print("\nIndividual code tokenization:")
    for code in test_codes:
        tokens = unified_tokenizer.tokenize(code)
        ids = unified_tokenizer.convert_tokens_to_ids(tokens)
        print(f"  {code:12s} → {tokens} (IDs: {ids})")
    
    # Verify special tokens work
    print("\nSpecial ICD tokens in vocab:")
    for tok in ["CHAP_A", "BLOCK_A00", "CAT_A00", "SUBCAT_0", "<ICD_START>", "<ICD_END>"]:
        if tok in unified_tokenizer.get_vocab():
            print(f"  {tok}: ID={unified_tokenizer.convert_tokens_to_ids(tok)}")
        else:
            print(f"  {tok}: NOT FOUND")


if __name__ == "__main__":
    main()