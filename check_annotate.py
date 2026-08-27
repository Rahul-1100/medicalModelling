#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

def main():
    # Path to your generated dataset
    input_file = Path("./data/annotated_medical_meadow_cleaned.jsonl")
    
    if not input_file.exists():
        print(f"Error: Could not find {input_file}")
        return

    total_records = 0
    fallback_count = 0
    # To count how many records have 1 code, 2 codes, 3 codes, etc.
    code_length_counts = Counter()
    # To optionally see the most common codes overall
    all_codes_counter = Counter()

    print(f"Reading {input_file}...\n")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
                
            record = json.loads(line)
            codes = record.get("icd10_codes", [])
            
            total_records += 1
            num_codes = len(codes)
            code_length_counts[num_codes] += 1
            
            for code in codes:
                all_codes_counter[code] += 1
                
            # Check if this record used the fallback we set in the previous script
            if num_codes == 1 and codes[0] == "Z0000":
                fallback_count += 1

    if total_records == 0:
        print("No records found in the file.")
        return

    # --- Print Statistics ---
    print(f"Total records analyzed: {total_records}")
    print("-" * 40)
    
    print("Distribution of ICD-10 code counts per record:")
    # Sort by the number of codes (e.g., 0, 1, 2, 3...)
    for length in sorted(code_length_counts.keys()):
        count = code_length_counts[length]
        percentage = (count / total_records) * 100
        print(f"  • {length} code(s): {count:,} records ({percentage:.2f}%)")
        
        # If showing the 1-code stats, note how many were the fallback
        if length == 1 and fallback_count > 0:
            fallback_pct = (fallback_count / count) * 100
            print(f"      -> {fallback_count:,} of these were the 'Z0000' fallback ({fallback_pct:.2f}% of 1-code records)")

    print("-" * 40)
    print("Top 10 most frequently assigned codes overall:")
    for code, count in all_codes_counter.most_common(10):
        percentage = (count / total_records) * 100
        print(f"  • {code}: {count:,} times ({percentage:.2f}%)")

if __name__ == "__main__":
    main()