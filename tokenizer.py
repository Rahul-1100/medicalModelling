#!/usr/bin/env python3
"""
tokenizer.py - Fixed for your exact CSV/TXT format
CSV: category,subcategory,full_code,short_desc,long_desc (compact codes: A000, A0100, A011)
TXT: ICD9|ICD10|description (standard codes: T80.40xA, Y99.0, Y93.01)

Produces artifacts in ./icd10_tokenizer/:
- code_to_idx.json       # Code → index (for embedding layer)
- idx_to_code.json       # Index → code
- descriptions.json      # Code → clinical description
- hierarchy.json         # Code → {chapter, block, category, subcategory}
- tokenizer_config.json  # Config for reproducibility
- hf_tokenizer/          # HuggingFace tokenizer with 8 special tokens

Usage:
    python tokenizer.py --data_dir /path/to/kaggle/download --output_dir ./icd10_tokenizer
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer


def normalize_code(code):
    """
    Convert compact ICD-10 to standard format with dot.
    Rules: First 3 chars = category (letter + 2 digits), rest = subcategory
    A000 → A00.0
    A001 → A00.1
    A009 → A00.9
    A0100 → A01.00
    A0101 → A01.01
    A011 → A01.1
    A012 → A01.2
    A020 → A02.0
    T8040xA → T80.40xA (already handled by TXT parser - already has dot)
    """
    code = str(code).strip().upper()
    if not code or code == 'NAN':
        return ''
    
    # Already has dot? Return as-is (TXT file codes like T80.40xA, Y99.0)
    if '.' in code:
        return code
    
    # Compact format: first 3 chars = category, rest = subcategory
    if len(code) >= 3:
        category = code[:3]  # e.g., A00, A01, A02, T80
        subcategory = code[3:]  # e.g., 0, 1, 9, 00, 01, 1, 2, 40xA
        
        if subcategory:
            return f"{category}.{subcategory}"
        else:
            return category  # Just category level (e.g., A00)
    
    return code


def parse_icd10_csv(csv_path):
    """Parse your exact CSV format: 5 columns, comma-separated"""
    # Read with comma separator, no header
    df = pd.read_csv(csv_path, header=None, names=[
        'category', 'subcategory', 'full_code', 'short_desc', 'long_desc','shortest_desc'
    ], dtype=str,
    index_col=False )
    # return df
    # Clean whitespace
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Handle empty subcategory
    df['subcategory'] = df['subcategory'].replace(['', 'nan', 'None', 'NA'], '')
    
    # Normalize the full_code (column 2) to standard format
    df['code'] = df['full_code'].apply(normalize_code)
    
    # Use long_desc as primary description
    df['description'] = df['long_desc']
    
    print(f"CSV parsed: {len(df)} rows")
    print(f"Sample codes (raw → normalized):")
    for _, row in df.head(15).iterrows():
        print(f"  {row['full_code']} → {row['code']} | {row['description'][:60]}")
    
    print(f"Unique normalized codes: {df['code'].nunique()}")
    return df[['code', 'description', 'category', 'subcategory']].copy()


def parse_icd9_to_10_dict(txt_path):
    """Parse your exact TXT format: ICD9|ICD10|description"""
    mappings = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|', 2)
            if len(parts) == 3:
                icd9, icd10, desc = parts
                mappings.append({
                    'icd9': icd9.strip(),
                    'icd10': icd10.strip(),  # Already standard format (T80.40xA, Y99.0)
                    'description': desc.strip()
                })
            elif len(parts) == 2:
                mappings.append({
                    'icd9': parts[0].strip(),
                    'icd10': parts[1].strip(),
                    'description': ''
                })
    
    df = pd.DataFrame(mappings)
    print(f"TXT parsed: {len(df)} mappings")
    print(f"Sample ICD-10 codes: {df['icd10'].head(10).tolist()}")
    print(f"Unique ICD-10 codes: {df['icd10'].nunique()}")
    return df


def build_unified_registry(csv_df, map_df):
    """Combine both sources, CSV priority for descriptions"""
    # CSV codes (already normalized)
    csv_codes = csv_df[['code', 'description']].copy()
    csv_codes['source'] = 'csv'
    csv_codes['category'] = csv_df['category']
    csv_codes['subcategory'] = csv_df['subcategory']
    
    # TXT mapping codes (already standard format)
    map_codes = map_df[['icd10', 'description']].copy()
    map_codes.columns = ['code', 'description']
    map_codes['source'] = 'icd9_mapping'
    map_codes['category'] = map_codes['code'].str[:3]  # First 3 chars (e.g., T80, Y99)
    map_codes['subcategory'] = ''
    
    # Combine, CSV priority
    combined = pd.concat([csv_codes, map_codes], ignore_index=True)
    combined = combined.drop_duplicates(subset=['code'], keep='first')
    combined = combined.sort_values('code').reset_index(drop=True)
    
    return combined


def build_hierarchy(registry_df):
    """Build hierarchy from standard format codes (with dots)"""
    hierarchy = {}
    
    for _, row in registry_df.iterrows():
        code = str(row['code'])
        if not code:
            continue
        
        # Parse standard format: CATEGORY.SUBCATEGORY
        if '.' in code:
            category, subcategory = code.split('.', 1)
        else:
            category = code
            subcategory = ''
        
        # Chapter: first letter
        chapter = category[0] if category else ''
        
        # Block: first 3 chars (chapter + 2 digits)
        block = category[:3] if len(category) >= 3 else category
        
        # Category: full category part (e.g., A00, A01, T80, Y99)
        # Subcategory: full code with dot
        full_code = code
        
        hierarchy[code] = {
            'chapter': chapter,
            'block': block,
            'category': category,
            'subcategory': full_code,
            'path': [chapter, block, category, full_code]
        }
    
    return hierarchy


def create_tokenizer_artifacts(registry_df, hierarchy, output_dir, base_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
    """Create all tokenizer artifacts"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    codes = registry_df['code'].tolist()
    code_to_idx = {code: i for i, code in enumerate(codes)}
    idx_to_code = {i: code for code, i in code_to_idx.items()}
    descriptions = dict(zip(registry_df['code'], registry_df['description']))
    
    # Save JSON artifacts
    with open(output_path / 'code_to_idx.json', 'w') as f:
        json.dump(code_to_idx, f)
    with open(output_path / 'idx_to_code.json', 'w') as f:
        json.dump(idx_to_code, f)
    with open(output_path / 'descriptions.json', 'w') as f:
        json.dump(descriptions, f)
    with open(output_path / 'hierarchy.json', 'w') as f:
        json.dump(hierarchy, f)
    
    # Create HF tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    special_tokens = [
        '[ICD_START]', '[ICD_END]',
        '[ICD_CHAP]', '[ICD_BLOCK]', '[ICD_CAT]', '[ICD_SUBCAT]',
        '[ICD_DESC]', '[ICD_SEP]'
    ]
    
    new_tokens = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        print(f"Added {len(new_tokens)} special tokens to tokenizer")
    
    hf_tokenizer_dir = output_path / 'hf_tokenizer'
    tokenizer.save_pretrained(hf_tokenizer_dir)
    
    # Config
    config = {
        'base_model': base_model,
        'vocab_size': len(tokenizer),
        'special_tokens': special_tokens,
        'num_codes': len(codes),
        'strategies': ['description', 'hierarchical', 'hybrid', 'code_only', 'code_with_desc'],
        'hierarchy_levels': ['chapter', 'block', 'category', 'subcategory']
    }
    with open(output_path / 'tokenizer_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved all artifacts to {output_path}")
    return tokenizer, code_to_idx, idx_to_code, descriptions, hierarchy


class ICD10Tokenizer:
    """Runtime tokenizer for ICD-10 codes with multiple strategies."""
    
    def __init__(self, artifacts_dir, base_model=None):
        self.artifacts_dir = Path(artifacts_dir)
        
        with open(self.artifacts_dir / 'code_to_idx.json') as f:
            self.code_to_idx = json.load(f)
        with open(self.artifacts_dir / 'idx_to_code.json') as f:
            self.idx_to_code = {int(k): v for k, v in json.load(f).items()}
        with open(self.artifacts_dir / 'descriptions.json') as f:
            self.descriptions = json.load(f)
        with open(self.artifacts_dir / 'hierarchy.json') as f:
            self.hierarchy = json.load(f)
        
        if base_model is None:
            with open(self.artifacts_dir / 'tokenizer_config.json') as f:
                config = json.load(f)
            base_model = config['base_model']
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.artifacts_dir / 'hf_tokenizer')
        self.base_model = base_model
    
    def get_description(self, code):
        return self.descriptions.get(str(code), '')
    
    def get_hierarchy(self, code):
        return self.hierarchy.get(str(code), {})
    
    def get_indices(self, codes):
        """Convert code strings to indices (handles both formats)"""
        indices = []
        for c in codes:
            c_norm = normalize_code(c)
            indices.append(self.code_to_idx.get(c_norm, 0))
        return indices
    
    def tokenize_hierarchical(self, code, max_length=32):
        """Fixed-length hierarchical tokens (RT-2 style)"""
        code = normalize_code(str(code))
        h = self.hierarchy.get(code)
        if not h:
            return self.tokenize_code_only(code, max_length)
        
        tokens = []
        tokens.append(self.tokenizer.convert_tokens_to_ids('[ICD_START]'))
        
        for level_name, level_token in [
            ('chapter', '[ICD_CHAP]'),
            ('block', '[ICD_BLOCK]'),
            ('category', '[ICD_CAT]'),
            ('subcategory', '[ICD_SUBCAT]')
        ]:
            tokens.append(self.tokenizer.convert_tokens_to_ids(level_token))
            tokens.extend(self.tokenizer.encode(h[level_name], add_special_tokens=False))
        
        tokens.append(self.tokenizer.convert_tokens_to_ids('[ICD_END]'))
        return tokens[:max_length]
    
    def tokenize_hybrid(self, code, max_length=128):
        """Hierarchy + description (recommended)"""
        code = normalize_code(str(code))
        h = self.hierarchy.get(code)
        desc = self.get_description(code)
        
        if not h:
            return self.tokenize_description(code, max_length)
        
        tokens = []
        tokens.append(self.tokenizer.convert_tokens_to_ids('[ICD_START]'))
        
        for level_name, level_token in [
            ('chapter', '[ICD_CHAP]'),
            ('block', '[ICD_BLOCK]'),
            ('category', '[ICD_CAT]'),
            ('subcategory', '[ICD_SUBCAT]')
        ]:
            tokens.append(self.tokenizer.convert_tokens_to_ids(level_token))
            tokens.extend(self.tokenizer.encode(h[level_name], add_special_tokens=False))
        
        tokens.append(self.tokenizer.convert_tokens_to_ids('[ICD_SEP]'))
        tokens.append(self.tokenizer.convert_tokens_to_ids('[ICD_DESC]'))
        tokens.extend(self.tokenizer.encode(desc, add_special_tokens=False))
        tokens.append(self.tokenizer.convert_tokens_to_ids('[ICD_END]'))
        return tokens[:max_length]
    
    def tokenize_description(self, code, max_length=128):
        desc = self.get_description(normalize_code(str(code)))
        text = f"[ICD_START] {desc} [ICD_END]"
        return self.tokenizer.encode(text, max_length=max_length, truncation=True)
    
    def tokenize_code_only(self, code, max_length=16):
        text = f"[ICD_START] {normalize_code(str(code))} [ICD_END]"
        return self.tokenizer.encode(text, max_length=max_length, truncation=True)
    
    def encode_batch(self, codes, strategy='hybrid', max_length=128, padding=True):
        method = getattr(self, f'tokenize_{strategy}')
        batch = [method(c, max_length) for c in codes]
        
        if padding:
            max_len = max(len(x) for x in batch) if batch else 0
            pad_id = self.tokenizer.pad_token_id
            batch = [x + [pad_id] * (max_len - len(x)) for x in batch]
        
        return batch
    
    def __len__(self):
        return len(self.code_to_idx)
    
    def __contains__(self, code):
        return normalize_code(str(code)) in self.code_to_idx


def main():
    parser = argparse.ArgumentParser(description="Build ICD-10 tokenizer artifacts")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Kaggle dataset download')
    parser.add_argument('--output_dir', type=str, default='icd10_tokenizer',
                        help='Output directory for artifacts')
    parser.add_argument('--base_model', type=str, 
                        default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                        help='Base tokenizer model')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    csv_path = data_dir / 'ICD10codes.csv'
    dict_path = data_dir / 'icd9to10dictionary.txt'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"ICD10codes.csv not found in {data_dir}")
    if not dict_path.exists():
        raise FileNotFoundError(f"icd9to10dictionary.txt not found in {data_dir}")
    
    print("=" * 60)
    print("PARSING CSV (5 columns: cat, subcat, code, short, long)")
    print("=" * 60)
    csv_df = parse_icd10_csv(csv_path)
    
    print("\n" + "=" * 60)
    print("PARSING TXT (ICD9|ICD10|description)")
    print("=" * 60)
    map_df = parse_icd9_to_10_dict(dict_path)
    
    print("\n" + "=" * 60)
    print("BUILDING UNIFIED REGISTRY")
    print("=" * 60)
    registry = build_unified_registry(csv_df, map_df)
    print(f"Total unique ICD-10 codes: {len(registry)}")
    print(f"Sample codes: {registry['code'].head(20).tolist()}")
    
    print("\n" + "=" * 60)
    print("BUILDING HIERARCHY")
    print("=" * 60)
    hierarchy = build_hierarchy(registry)
    
    print("\n" + "=" * 60)
    print("CREATING TOKENIZER ARTIFACTS")
    print("=" * 60)
    tokenizer, code_to_idx, idx_to_code, descriptions, hierarchy = create_tokenizer_artifacts(
        registry, hierarchy, args.output_dir, args.base_model
    )
    
    # Verification - use ICD10Tokenizer wrapper class (has tokenize_* methods)
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Codes in registry: {len(registry)}")
    print(f"Codes in hierarchy: {len(hierarchy)}")
    print(f"Codes in code_to_idx: {len(code_to_idx)}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Create ICD10Tokenizer instance for testing (has the tokenize_* methods)
    icd10_tokenizer = ICD10Tokenizer(args.output_dir, args.base_model)
    
    # Test with codes from YOUR data
    test_codes = ['A00.0', 'A00.1', 'A00.9', 'A01.00', 'A01.01', 'A01.1', 'A01.2', 'A02.0', 
                  'T80.40xA', 'Y99.0', 'Y93.01', 'I21.9', 'E11.9', 'J44.1', 'Z00.00']
    print("\nTest encoding (hierarchical):")
    for code in test_codes:
        if code in icd10_tokenizer.code_to_idx:
            tokens = icd10_tokenizer.tokenize_hierarchical(code)
            tok_str = icd10_tokenizer.tokenizer.convert_ids_to_tokens(tokens)
            print(f"  ✓ {code}: {len(tokens)} tokens → {tok_str}")
        else:
            print(f"  ✗ {code}: NOT FOUND")
    
    # Test compact format input (what your annotator might output)
    print("\nTest with compact format input (A000, A0100, A011):")
    for code in ['A000', 'A001', 'A0100', 'A0101', 'A011', 'A012', 'A020']:
        norm = normalize_code(code)
        if norm in icd10_tokenizer.code_to_idx:
            tokens = icd10_tokenizer.tokenize_hierarchical(code)
            print(f"  ✓ {code} → {norm}: {len(tokens)} tokens")
        else:
            print(f"  ✗ {code} → {norm}: NOT FOUND")


if __name__ == '__main__':
    main()
