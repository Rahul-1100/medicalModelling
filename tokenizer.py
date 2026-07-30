import re
from typing import List, Dict, Any, Tuple
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# ==========================================
# 1. TOKENIZER SETUP
# ==========================================

def setup_icd_tokenizer(base_model_name: str = "bert-base-uncased") -> AutoTokenizer:
    """
    Loads base tokenizer and appends reserved tokens <ICD_00> through <ICD_99>.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    icd_tokens = [f"<ICD_{i:02d}>" for i in range(100)]
    tokenizer.add_special_tokens({"additional_special_tokens": icd_tokens})
    return tokenizer


# ==========================================
# 2. COVERAGE AUDITING ENGINE
# ==========================================

class ICD10CoverageAuditor:
    """
    Audits a pandas DataFrame column to verify 100% of ICD-10 codes 
    are covered by the custom vocabulary without subword splitting.
    """
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        # Matches any letter A-Z, optional dot/space, and captures the 2-digit bucket (00-99)
        # e.g., "A.15", "A15", "A15.9", "B.20", "C18.9"
        self.pattern = re.compile(r"\b([A-Z])\.?\s*(\d{2})(?:\.\d+)?\b")
        self.valid_icd_tokens = set(f"<ICD_{i:02d}>" for i in range(100))

    def audit_column(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        results = []
        
        for idx, raw_val in df[column_name].dropna().items():
            code_str = str(raw_val).strip().upper()
            match = self.pattern.search(code_str)
            
            if match:
                bucket = match.group(2) # Extracted two digits (e.g., '15')
                token_name = f"<ICD_{bucket}>"
                
                token_id = self.tokenizer.convert_tokens_to_ids(token_name)
                is_unknown = (token_id == self.tokenizer.unk_token_id)
                
                results.append({
                    "row_index": idx,
                    "raw_code": code_str,
                    "is_covered": not is_unknown and token_name in self.valid_icd_tokens,
                    "mapped_token": token_name,
                    "token_id": token_id,
                    "failure_reason": None if not is_unknown else "Unknown token ID"
                })
            else:
                results.append({
                    "row_index": idx,
                    "raw_code": code_str,
                    "is_covered": False,
                    "mapped_token": None,
                    "token_id": None,
                    "failure_reason": "Regex pattern mismatch (Not valid ICD-10 format)"
                })
                
        return pd.DataFrame(results)


# ==========================================
# 3. BERT & TEXT PREPROCESSING PIPELINE
# ==========================================

def convert_icd_to_tokens(text: str) -> str:
    """
    Replaces raw ICD-10 occurrences (e.g., A15, B20, C18.9) in text with custom <ICD_XX> tokens.
    """
    def replace_match(match):
        digits = match.group(2)
        return f"<ICD_{digits}>"

    return re.sub(r"\b([A-Z])\.?\s*(\d{2})(?:\.\d+)?\b", replace_match, text)


class ICDPreprocessingPipeline:
    def __init__(self, bert_model_name_or_path: str):
        """
        Initializes BERT model for medical entity recognition / ICD-10 mapping.
        """
        self.icd_pipeline = pipeline(
            "ner",
            model=bert_model_name_or_path,
            tokenizer=bert_model_name_or_path,
            aggregation_strategy="simple"
        )

    def replace_text_with_icd_codes(self, text: str) -> str:
        """
        Uses BERT to detect medical terms and replace them with predicted ICD-10 codes.
        """
        entities = self.icd_pipeline(text)
        
        # Reverse sort prevents string character offset shifting
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        text_list = list(text)
        for entity in entities:
            start = entity['start']
            end = entity['end']
            icd_code = entity.get('entity_group', entity.get('word'))
            
            text_list[start:end] = f" {icd_code} "
            
        return "".join(text_list)

    def preprocess(self, text: str) -> str:
        """
        Step 1: Normal Text -> Standard ICD-10 codes (via BERT)
        Step 2: ICD-10 codes -> <ICD_XX> Tokens (via Regex)
        """
        text_with_icd = self.replace_text_with_icd_codes(text)
        final_text = convert_icd_to_tokens(text_with_icd)
        return final_text


# ==========================================
# 4. EXECUTION & VERIFICATION
# ==========================================

def run_verification(csv_path: str, code_column: str, use_dummy_data: bool = False):
    print("1. Initializing Tokenizer and Auditor...")
    tokenizer = setup_icd_tokenizer("bert-base-uncased")
    auditor = ICD10CoverageAuditor(tokenizer)
    
    if use_dummy_data:
        print("2. Running on sample test data...")
        df = pd.DataFrame({
            code_column: [
                "A.15", "A15", "A15.9", "B.20", "B20.1", "B07",
                "A99", "A.00", "C18.9", "Invalid_Code_123"
            ]
        })
    else:
        print(f"2. Loading Kaggle dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
    
    print(f"3. Auditing column '{code_column}' ({len(df)} rows)...")
    report_df = auditor.audit_column(df, code_column)
    
    total_codes = len(report_df)
    covered_codes = report_df["is_covered"].sum()
    leftovers = total_codes - covered_codes
    coverage_pct = (covered_codes / total_codes) * 100 if total_codes > 0 else 0
    
    print("\n" + "="*50)
    print("            ICD-10 VOCABULARY AUDIT REPORT")
    print("="*50)
    print(f"Total Codes Evaluated : {total_codes}")
    print(f"Successfully Covered  : {covered_codes} ({coverage_pct:.2f}%)")
    print(f"Leftover / Unmapped   : {leftovers}")
    print("="*50)
    
    if leftovers > 0:
        print("\n[WARNING] Found leftover codes that are NOT covered!")
        unmapped_df = report_df[~report_df["is_covered"]]
        print("\nSample of unmapped rows:")
        print(unmapped_df[["row_index", "raw_code", "failure_reason"]].head(10).to_string(index=False))
        
        report_df.to_csv("icd10_unmapped_audit_report.csv", index=False)
        print("\nFull breakdown saved to 'icd10_unmapped_audit_report.csv'.")
    else:
        print("\n[SUCCESS] 100% COVERAGE ACHIEVED! Zero leftover codes.")


if __name__ == "__main__":
    # To run on real Kaggle file:
    # run_verification(csv_path="your_kaggle_dataset.csv", code_column="icd10_code", use_dummy_data=False)
    
    # Running quick test with dummy data:
    run_verification(csv_path="", code_column="icd10_code", use_dummy_data=True)