#!/usr/bin/env python3
"""
test_tokenizer.py - Test script to verify ICD10Tokenizer works with:
1. User/System/Assistant conversation parts
2. ICD-10 code tokenization (using tokenizer.py's ICD10Tokenizer class)
3. Integration with the text tokenizer (MedGemma)

Run:
    python test_tokenizer.py --artifacts_dir ./icd10_tokenizer
"""

import argparse
from pathlib import Path
import sys

# Add the current directory to path to import tokenizer.py's ICD10Tokenizer
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import ICD10Tokenizer as TokenizerPyICD10Tokenizer
from train import ICD10Tokenizer as TrainICD10Tokenizer, MedicalMeadowICD10Dataset
from transformers import AutoTokenizer


def test_tokenizer_py_icd10(tokenizer_artifacts_dir):
    """Test the tokenizer.py's ICD10Tokenizer class directly"""
    print("=" * 60)
    print("TEST 1: tokenizer.py ICD10Tokenizer (rich strategies)")
    print("=" * 60)
    
    icd10_tok = TokenizerPyICD10Tokenizer(tokenizer_artifacts_dir)
    print(f"Vocab size: {len(icd10_tok)}")
    print(f"Special tokens: {icd10_tok.tokenizer.additional_special_tokens}")
    
    # Test codes from your CSV (compact format) and TXT (standard format)
    test_codes = [
        'A000',      # Compact → A00.0
        'A001',      # Compact → A00.1
        'A009',      # Compact → A00.9
        'A0100',     # Compact → A01.00
        'A0101',     # Compact → A01.01
        'A011',      # Compact → A01.1
        'A012',      # Compact → A01.2
        'A020',      # Compact → A02.0
        'T80.40xA',  # Standard (from TXT)
        'Y99.0',     # Standard (from TXT)
        'Y93.01',    # Standard (from TXT)
        'I21.9',     # Standard
        'E11.9',     # Standard
        'J44.1',     # Standard
        'Z00.00',    # Standard
    ]
    
    print("\n--- Tokenization Strategies ---")
    for code in test_codes:
        print(f"\nCode: {code}")
        
        # Check if code exists (after normalization)
        normalized = code if '.' in code else ''
        if not normalized:
            # Apply normalization manually to check
            from tokenizer import normalize_code
            normalized = normalize_code(code)
        
        if normalized in icd10_tok.code_to_idx:
            print(f"  Normalized: {normalized} (index: {icd10_tok.code_to_idx[normalized]})")
            
            # Test different strategies
            for strategy in ['hierarchical', 'hybrid', 'description', 'code_only']:
                method = getattr(icd10_tok, f'tokenize_{strategy}')
                tokens = method(code)
                tok_str = icd10_tok.tokenizer.convert_ids_to_tokens(tokens)
                print(f"  {strategy:15s}: {len(tokens):3d} tokens → {tok_str[:10]}...")
        else:
            print(f"  Normalized: {normalized} → NOT FOUND in vocabulary")


def test_train_icd10_tokenizer(tokenizer_artifacts_dir):
    """Test the train.py's ICD10Tokenizer wrapper"""
    print("\n" + "=" * 60)
    print("TEST 2: train.py ICD10Tokenizer (simple encode)")
    print("=" * 60)
    
    icd10_tok = TrainICD10Tokenizer(tokenizer_artifacts_dir)
    print(f"Vocab size: {len(icd10_tok)}")
    
    test_codes = ['A00.0', 'A01.00', 'T80.40xA', 'I21.9', 'E11.9']
    
    for code in test_codes:
        indices = icd10_tok.get_indices(code)
        tokens = icd10_tok.tokenizer.convert_ids_to_tokens(indices)
        print(f"  {code:12s} → {len(indices):3d} tokens: {tokens}")


def test_text_tokenizer_conversation(model_name="unsloth/gemma-3-4b-it-bnb-4bit"):
    """Test MedGemma text tokenizer with user/system/assistant parts"""
    print("\n" + "=" * 60)
    print("TEST 3: Text Tokenizer (User/System/Assistant parts)")
    print("=" * 60)
    
    # Use a smaller model for testing if the main one isn't available
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        print(f"Could not load {model_name}, using microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        text_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    
    print(f"Text tokenizer vocab size: {len(text_tokenizer)}")
    print(f"Pad token: {text_tokenizer.pad_token} (id: {text_tokenizer.pad_token_id})")
    print(f"EOS token: {text_tokenizer.eos_token} (id: {text_tokenizer.eos_token_id})")
    
    # Sample conversation parts
    system_prompt = "You are a medical coding assistant. Provide accurate ICD-10 codes."
    user_message = "Patient presents with acute myocardial infarction and type 2 diabetes."
    assistant_response = "Based on the clinical presentation, the appropriate ICD-10 codes are:\n- I21.9: Acute myocardial infarction, unspecified\n- E11.9: Type 2 diabetes mellitus without complications"
    
    # Tokenize each part separately
    print("\n--- Separate Tokenization ---")
    for name, text in [("System", system_prompt), ("User", user_message), ("Assistant", assistant_response)]:
        encoding = text_tokenizer(text, add_special_tokens=False)
        tokens = text_tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        print(f"{name:10s}: {len(encoding['input_ids']):4d} tokens → {tokens[:15]}...")
    
    # Tokenize full conversation (typical format)
    full_conversation = f"### System:\n{system_prompt}\n\n### User:\n{user_message}\n\n### Assistant:\n{assistant_response}"
    
    print("\n--- Full Conversation ---")
    encoding = text_tokenizer(full_conversation, add_special_tokens=True, max_length=2048, truncation=True)
    tokens = text_tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    print(f"Full conv: {len(encoding['input_ids']):4d} tokens → {tokens[:20]}...")
    
    return text_tokenizer


def test_dataset_integration(tokenizer_artifacts_dir, text_tokenizer):
    """Test the MedicalMeadowICD10Dataset integration"""
    print("\n" + "=" * 60)
    print("TEST 4: Dataset Integration (MedicalMeadowICD10Dataset)")
    print("=" * 60)
    
    icd10_tok = TrainICD10Tokenizer(tokenizer_artifacts_dir)
    
    # Create a sample data item
    sample_data = [{
        "question": "Patient with acute MI and diabetes",
        "answer": "I21.9, E11.9",
        "icd10_codes": ["I21.9", "E11.9", "A00.0", "T80.40xA"]
    }]
    
    dataset = MedicalMeadowICD10Dataset(
        sample_data, 
        icd10_tok, 
        text_tokenizer, 
        max_text_length=512, 
        max_codes=10, 
        max_code_tokens=18
    )
    
    item = dataset[0]
    
    print(f"input_ids shape: {item['input_ids'].shape}")
    print(f"attention_mask shape: {item['attention_mask'].shape}")
    print(f"icd10_indices shape: {item['icd10_indices'].shape}")
    print(f"labels shape: {item['labels'].shape}")
    
    # Decode text
    decoded_text = text_tokenizer.decode(item['input_ids'], skip_special_tokens=True)
    print(f"\nDecoded text (first 200 chars):\n{decoded_text[:200]}...")
    
    # Show ICD-10 indices matrix
    print(f"\nICD-10 indices matrix [10 codes x 18 tokens]:")
    for i in range(min(5, item['icd10_indices'].shape[0])):
        row = item['icd10_indices'][i].tolist()
        non_zero = [(j, v) for j, v in enumerate(row) if v != 0]
        if non_zero:
            tokens = text_tokenizer.convert_ids_to_tokens([v for _, v in non_zero])
            print(f"  Code {i}: non-zero at positions {non_zero} → {tokens}")


def test_end_to_end(tokenizer_artifacts_dir, model_name="unsloth/gemma-3-4b-it-bnb-4bit"):
    """End-to-end test combining everything"""
    print("\n" + "=" * 60)
    print("TEST 5: End-to-End Pipeline")
    print("=" * 60)
    
    # Load both tokenizers
    icd10_tok_py = TokenizerPyICD10Tokenizer(tokenizer_artifacts_dir)
    icd10_tok_train = TrainICD10Tokenizer(tokenizer_artifacts_dir)
    
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        text_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    
    # Simulate a medical conversation with ICD-10 codes
    conversation = """### System:
You are a medical coding assistant. Provide accurate ICD-10 codes for the clinical scenario.

### User:
A 65-year-old male presents with crushing chest pain radiating to left arm, 
diaphoresis, and shortness of breath. ECG shows ST elevation in leads II, III, aVF. 
History of type 2 diabetes mellitus and hypertension.

### Assistant:
Based on the clinical presentation, this is an acute inferior ST-elevation myocardial infarction.
The appropriate ICD-10 codes are:
- I21.1: Acute transmural myocardial infarction of inferior wall
- I25.10: Atherosclerotic heart disease of native coronary artery without angina pectoris
- E11.9: Type 2 diabetes mellitus without complications
- I10: Essential (primary) hypertension"""
    
    # Tokenize the full conversation
    text_encoding = text_tokenizer(
        conversation,
        max_length=2048,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    print(f"Text tokens: {text_encoding['input_ids'].shape}")
    
    # Simulate extracted ICD-10 codes from the assistant response
    extracted_codes = ["I21.1", "I25.10", "E11.9", "I10"]
    
    # Create ICD-10 indices matrix (as done in dataset)
    max_codes = 10
    max_code_tokens = 18
    padded_matrix = torch.zeros((max_codes, max_code_tokens), dtype=torch.long)
    
    import torch
    
    for i, code in enumerate(extracted_codes[:max_codes]):
        code_tokens = icd10_tok_train.get_indices(code)
        code_tokens = code_tokens[:max_code_tokens]
        padded_matrix[i, :len(code_tokens)] = torch.tensor(code_tokens, dtype=torch.long)
        
        # Show what tokens were generated
        tokens = icd10_tok_py.tokenizer.convert_ids_to_tokens(code_tokens)
        print(f"  {code:10s} → {len(code_tokens):2d} tokens: {tokens}")
    
    print(f"\nICD-10 matrix shape: {padded_matrix.shape}")
    print(f"Non-zero entries: {(padded_matrix != 0).sum().item()}")
    
    # Test hierarchical tokenization for one code
    print("\n--- Hierarchical tokenization (rich strategy) ---")
    for code in ["I21.1", "E11.9"]:
        tokens = icd10_tok_py.tokenize_hierarchical(code, max_length=32)
        tok_str = icd10_tok_py.tokenizer.convert_ids_to_tokens(tokens)
        print(f"  {code}: {len(tokens)} tokens → {tok_str}")


def main():
    parser = argparse.ArgumentParser(description="Test ICD-10 tokenizer")
    parser.add_argument('--artifacts_dir', type=str, default='./icd10_tokenizer',
                        help='Path to tokenizer artifacts directory')
    parser.add_argument('--model_name', type=str, default='unsloth/gemma-3-4b-it-bnb-4bit',
                        help='Text tokenizer model name')
    parser.add_argument('--test', type=str, choices=['all', 'py', 'train', 'text', 'dataset', 'e2e'],
                        default='all', help='Which test to run')
    args = parser.parse_args()
    
    artifacts_dir = Path(args.artifacts_dir)
    
    if not artifacts_dir.exists():
        print(f"Error: Artifacts directory not found: {artifacts_dir}")
        print("Run tokenizer.py first to generate artifacts:")
        print(f"  python tokenizer.py --data_dir ./2 --output_dir ./icd10_tokenizer")
        return
    
    if args.test in ['all', 'py']:
        test_tokenizer_py_icd10(artifacts_dir)
    
    if args.test in ['all', 'train']:
        test_train_icd10_tokenizer(artifacts_dir)
    
    if args.test in ['all', 'text']:
        text_tok = test_text_tokenizer_conversation(args.model_name)
    else:
        try:
            text_tok = AutoTokenizer.from_pretrained(args.model_name)
        except:
            text_tok = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        if text_tok.pad_token is None:
            text_tok.pad_token = text_tok.eos_token
    
    if args.test in ['all', 'dataset']:
        test_dataset_integration(artifacts_dir, text_tok)
    
    if args.test in ['all', 'e2e']:
        test_end_to_end(artifacts_dir, args.model_name)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()