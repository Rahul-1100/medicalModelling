#!/usr/bin/env python3
"""
build_hierarchical_icd10_vocab.py

Build a compact ICD-10 vocabulary using hierarchical BPE.
Strategy:
1. Parse all ICD-10 codes into hierarchical components (chapter, category, subcategory, extension)
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

ICD10_PATTERN = re.compile(r'^(?=[A-Z](?:\d{2}|\d[A-TV-Z])\.?[A-TV-Z0-9]{0,4}$)([A-Z])(\d{2}|\d[A-TV-Z])(\.?((?:[A-TV-Z][0-9A-TV-Z]{0,3})|(?:\d{1,3}[A-TV-Z0-9]{1,3})|(?:\d{1,4}))?)?$') # WTF is even this Need to reduce it if possible but this matches every case! 

def decompose_icd10_code(code: str) -> Dict[str, str]:
    """
    Decompose ICD-10 code into hierarchical components.

    Placeholder rule: ICD-10-CM reserves 'X' as a placeholder in positions 5/6,
    used only to push a 7th character into position 7. Placeholders are replaced
    with '_' so they are position information, not vocabulary content.

    A subcode of exactly 4 chars always ends in the 7th character (extension);
    subcodes of 1-3 chars carry no extension.

    Examples:
        A00.0    → {'chapter': 'A', 'category': 'A00', 'subcategory': '0',   'extension': ''}
        M48.54XA → {'chapter': 'M', 'category': 'M48', 'subcategory': '54_', 'extension': 'A'}
        O32.1XX1 → {'chapter': 'O', 'category': 'O32', 'subcategory': '1__', 'extension': '1'}
        T50.Z15A → {'chapter': 'T', 'category': 'T50', 'subcategory': 'Z15', 'extension': 'A'}
        M1A.9XX0 → {'chapter': 'M', 'category': 'M1A', 'subcategory': '9__', 'extension': '0'}
        D46.C    → {'chapter': 'D', 'category': 'D46', 'subcategory': 'C',   'extension': ''}
        N08      → {'chapter': 'N', 'category': 'N08', 'subcategory': '',    'extension': ''}
    """
    code = str(code).strip().upper()

    m = ICD10_PATTERN.match(code)
    if m:
        chapter = m.group(1)
        category = chapter + m.group(2)
        subcode = m.group(4) or ''

        if len(subcode) == 4:
            subcategory = subcode[:-1].replace('X', '_')
            extension = subcode[-1]
        else:
            subcategory = subcode.replace('X', '_')
            extension = ''

        return {
            'chapter': chapter,
            'category': category,
            'subcategory': subcategory,
            'extension': extension
        }

    # Regex failed (should not happen for codes passing filter_valid_codes):
    # best-effort fallback so downstream code still gets a usable shape.
    if len(code) >= 3:
        return {'chapter': code[0], 'category': code[:3], 'subcategory': '', 'extension': ''}
    return {'chapter': code[0] if code else '', 'category': '', 'subcategory': '', 'extension': ''}


def code_to_hierarchical_tokens(code: str) -> List[str]:
    """
    Convert ICD-10 code to a sequence of hierarchical tokens for BPE training.

    Factorized scheme: the chapter letter is NOT repeated in the category token,
    so `CAT_48` is shared across M48/D48/C48/K48/... and the model learns to
    condition on the preceding CHAP_ token. This collapses ~2000 CAT_* tokens
    into ~350 (digits 00-99 + digit+letter tails like CAT_1A for M1A) and makes
    every possible code representable (zero-OOV at this level), at the cost of
    CAT_ tokens being meaningful only in CHAP_ context.

    Uses special markers to preserve hierarchy boundaries:
    A00.0    → ['CHAP_A', 'CAT_00', 'SUBCAT_0']
    M48.54XA → ['CHAP_M', 'CAT_48', 'SUBCAT_54_', 'EXT_A']
    O32.1XX1 → ['CHAP_O', 'CAT_32', 'SUBCAT_1__', 'EXT_1']
    T50.Z15A → ['CHAP_T', 'CAT_50', 'SUBCAT_Z15', 'EXT_A']
    Z23      → ['CHAP_Z', 'CAT_23']

    NOTE: Markers use "PREFIX_level" format, NOT "<CHAP>A"-style. The tokenizers
    BpeTrainer treats '<' and '>' as special-token delimiters and forbids any
    learned merge containing them, so angle-bracket compounds can never become
    single BPE tokens.
    """
    parts = decompose_icd10_code(code)
    tokens = []

    if parts['chapter']:
        tokens.append(f"CHAP_{parts['chapter']}")
        if len(parts['category']) > 1:
            tokens.append(f"CAT_{parts['category'][1:]}")
    if parts['subcategory']:
        tokens.append(f"SUBCAT_{parts['subcategory']}")
    if parts['extension']:
        tokens.append(f"EXT_{parts['extension']}")

    return tokens


# Loose scanner for mid-text matching. Candidates are validated with
# ICD10_PATTERN before being rewritten, so junk like "H1N1" or codes glued
# into words ("XA99B") stay as plain text.
ICD10_CANDIDATE_PATTERN = re.compile(
    r'(?<![A-Z0-9])[A-Z](?:\d{2}|\d[A-TV-Z])\.?[A-TV-Z0-9]{0,4}(?![A-TV-Z0-9])'
)


def rewrite_codes_in_text(text: str) -> str:
    """
    Replace every ICD-10 code in raw text with its factorized marker string, so
    the unified tokenizer resolves each level to a single stable ID. Markers are
    joined WITHOUT separators - Gemma's SentencePiece turns every space between
    markers into a standalone '▁' token, so dropping the spaces saves ~1-3
    tokens per code (added-token matching is literal, boundaries are unambiguous):

        "diagnosed with M48.54XA and E11.9"
        → "diagnosed with CHAP_MCAT_48SUBCAT_54_EXT_A and CHAP_ECAT_11SUBCAT_9"

    Sentence punctuation is preserved: a dot FOLLOWING a code ("...with E11.9.")
    stays untouched because the candidate ends at the last code character.
    Candidates that fail validation (including bare-dot ones like "A09.") are
    left as-is. Limitation: lowercase placeholder 'x' (e.g. "T80.40xA" written
    in prose) is not scanned - normalize data to uppercase before rewriting.
    """
    s = str(text)
    out = []
    last = 0
    for m in ICD10_CANDIDATE_PATTERN.finditer(s):
        cand = m.group(0)
        if cand.endswith('.') or not ICD10_PATTERN.match(cand):
            continue
        out.append(s[last:m.start()])
        out.append("".join(code_to_hierarchical_tokens(cand)))
        last = m.end()
    out.append(s[last:])
    return "".join(out)


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
        # NOTE: Do NOT include CHAP/CAT/SUBCAT/EXT markers here. Also never use
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
        all_sequences.append("".join(tokens))
    
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
    codes: List[str] = None,
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

    # IMPORTANT: only the enumerated LEVEL tokens are added to Gemma - never
    # the raw BPE vocab. Trained on separator-free marker strings, BPE learns
    # spanning merges ('CHAP_ACAT_0', '0SUBCAT_0'); if those were added as
    # tokens, the added-token trie's longest-match rule would shadow the level
    # tokens and destroy single-ID encoding.
    LEVEL_TOKEN_PATTERN = re.compile(
        r'^(?:CHAP_[A-TV-Z]|CAT_[0-9A-TV-Z]+|SUBCAT_[0-9A-TV-Z_]+|EXT_[0-9A-TV-Z])$'
    )

    # Factorized level tokens form a closed set: enumerate them all so no
    # BPE frequency cut can ever drop one. CAT tails are shared across chapters
    # (CHAP_M + CAT_48 = M48), so the whole set is tiny and gives zero-OOV at
    # the chapter/category/extension levels by construction. SUBCAT_ stays
    # frequency-driven (open-ended inventory).
    import string
    letters = [c for c in string.ascii_uppercase if c != "U"]
    guaranteed = {f"CHAP_{c}" for c in letters}
    guaranteed |= {f"CAT_{n:02d}" for n in range(100)}
    guaranteed |= {f"CAT_{d}{l}" for d in string.digits for l in letters}
    guaranteed |= {f"EXT_{c}" for c in string.digits + "".join(letters)}
    # Code-block delimiters for generation-time parsing (see _generate_codes
    # in train_unified_tokenizer.py). Added as plain added-tokens to Gemma -
    # the old BpeTrainer '<'/'>' merge restriction doesn't apply since these
    # bypass BPE entirely.
    guaranteed |= {"<ICD_START>", "<ICD_END>"}
    missing_guaranteed = guaranteed - icd10_vocab
    print(f"Guaranteed factorized tokens: {len(guaranteed)} "
          f"(not learned by BPE: {len(missing_guaranteed)})")

    # SUBCAT_ inventory is open-ended, so guarantee the OBSERVED ones instead:
    # every subcat that appears in the training codes is added directly,
    # bypassing BPE frequency cuts (singleton subcodes would otherwise fall
    # back to meaningless subword pieces).
    observed_subcats = set()
    if codes:
        for c in codes:
            p = decompose_icd10_code(c)
            if p['subcategory']:
                observed_subcats.add(f"SUBCAT_{p['subcategory']}")
        missing_subcats = observed_subcats - icd10_vocab
        icd10_vocab |= observed_subcats
        print(f"Observed SUBCAT tokens added: {len(missing_subcats)} "
              f"(total distinct observed: {len(observed_subcats)})")
    
    # Find new tokens to add: ONLY enumerated level tokens, never BPE merges
    level_tokens = guaranteed | observed_subcats
    orphan_bpe = {t for t in icd10_vocab
                  if LEVEL_TOKEN_PATTERN.match(t) and t not in level_tokens}
    if orphan_bpe:
        print(f"[warn] {len(orphan_bpe)} BPE-learned level-shaped tokens ignored "
              f"(not in enumerated sets): {sorted(orphan_bpe)[:5]} ...")
    new_tokens = sorted(level_tokens - gemma_vocab)
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
    # "A09." - a bare trailing dot is never valid ICD-10; the regex accepts it
    # only because the dot is optional, so clean it here instead.
    code = code.rstrip('.')
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


def strip_code_dots(text: str) -> str:
    """
    Remove the dot only inside ICD-10 codes, leaving other punctuation alone.

    The dot's position is fully deterministic (always right after the 3-char
    category), so the model learns it implicitly from CHAP_/CAT_ tokens and the
    dot never needs to be a token of its own. "E11.9" -> "E119" and
    "M4854XA"/"M48.54XA" become identical strings before tokenization.
    Sentence-ending periods after a code ("...diagnosed with E11.9.") are kept,
    since they follow a digit, not the category.
    """
    return re.sub(r'(?<=[A-Z](?:\d{2}|\d[A-TV-Z]))\.(?=[A-TV-Z0-9])', '', str(text))


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
        codes=codes,
        gemma_model_name=args.gemma_model,
        output_dir=args.output_dir
    )
    
    # Test the unified tokenizer
    print("\n" + "="*60)
    print("TESTING UNIFIED TOKENIZER")
    print("="*60)
    
    test_codes = ["A00.0", "A00.1", "A01.01", "T80.40xA", "Z23", "I10", "E11.9",
                  "M48.54XA", "O32.1XX1", "T50.Z15A", "M1A.9XX0", "D46.C", "N08"]
    test_text = "Patient diagnosed with A00.0 and I10, also has E11.9"

    print(f"\nTest text (codes rewritten to hierarchy tokens): {rewrite_codes_in_text(test_text)}")
    encoding = unified_tokenizer(rewrite_codes_in_text(test_text), return_tensors="pt")
    tokens = unified_tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {encoding['input_ids'][0].tolist()}")

    print("\nIndividual code tokenization:")
    for code in test_codes:
        tokens = unified_tokenizer.tokenize("".join(code_to_hierarchical_tokens(code)))
        ids = unified_tokenizer.convert_tokens_to_ids(tokens)
        print(f"  {code:12s} → {tokens} (IDs: {ids})")
    
    # Verify special tokens work
    print("\nSpecial ICD tokens in vocab:")
    for tok in ["CHAP_A", "CAT_00", "SUBCAT_0", "<ICD_START>", "<ICD_END>"]:
        if tok in unified_tokenizer.get_vocab():
            print(f"  {tok}: ID={unified_tokenizer.convert_tokens_to_ids(tok)}")
        else:
            print(f"  {tok}: NOT FOUND")


if __name__ == "__main__":
    main()