import sys
from pathlib import Path
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

CORPUS_PATH = Path("../register-corpus") # path to repo register corpus
if not CORPUS_PATH.exists() or not CORPUS_PATH.is_dir() or not any(CORPUS_PATH.iterdir()):
    print(f"current filepath to register-corpus repo ({CORPUS_PATH}) does not exist, is not a directory, or is empty", file=sys.stderr)
    print("edit the CORPUS_PATH variable in utils/corpus_load.py to the proper directory", file=sys.stderr)
    sys.exit(1)

reg_abbv_path = CORPUS_PATH / "info/reg_abbv.json"
if reg_abbv_path.exists():
    with open(reg_abbv_path) as reg_abbv_file:
        REG_ABBV2NAME = json.load(reg_abbv_file)
else:
    print(f"register abbreviation json file not found at {reg_abbv_path}, check if path is correct or register-corpus path {CORPUS_PATH} is correct", file=sys.stderr)
    sys.exit(1)

REGISTERS = list(REG_ABBV2NAME.keys())
REG2ID = {reg : REGISTERS.index(reg) for reg in REGISTERS}

class RegisterDataset(Dataset):
    def __init__(self, encoded_texts : dict[str, list[torch.Tensor]], registers : list[int]) -> None:
        self.encoded_texts = encoded_texts
        self.registers = registers

    def __len__(self) -> int:
        return len(self.registers)

    def __getitem__(self, index : int) -> dict[str, torch.Tensor]:
        encoded_text = {k : v[index] for k, v in self.encoded_texts.items()}
        register = self.registers[index]
        return {**encoded_text, 'labels': torch.tensor(register)}


def load_data(path : Path, model_checkpoint : str) -> Dataset:
    path = CORPUS_PATH / "corpus" / path
    
    if path.exists():
        dataset = pd.read_csv(path, sep='\t')
    else:
        print("corpus file not found, incorrect language specification or bad corpus filepath (see CORPUS_PATH in utils/corpus_load.py)", file=sys.stderr)
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    texts = dataset.iloc[:,1].tolist()
    encoded_texts = dict(tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True))
    registers = dataset.iloc[:,0].tolist()
    registers = [REG2ID[reg] for reg in registers]
    
    return RegisterDataset(encoded_texts, registers)