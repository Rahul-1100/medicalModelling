# import kagglehub
# import shutil
from pathlib import Path

# Download latest version
# path = kagglehub.dataset_download("arashnic/icd10-codes-and-descriptions")
destination = Path(__file__).resolve().parent
# shutil.move(path, destination)

# print("Path to dataset files:", destination / Path(path).name)

from datasets import load_dataset

ds = load_dataset("FiscaAI/synth-ehr-icd10cm-prompt", cache_dir=destination)
