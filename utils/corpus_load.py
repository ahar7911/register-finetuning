import sys
from pathlib import Path
from random import sample
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# load register-corpus repository and perform checks
CORPUS_PATH = Path("../register-corpus") # path to register-corpus repository
if not CORPUS_PATH.exists() or not CORPUS_PATH.is_dir() or not any(CORPUS_PATH.iterdir()):
    print(f"Current filepath to register-corpus repository ({CORPUS_PATH}) does not exist, is not a directory, or is empty", file=sys.stderr)
    print("Edit the CORPUS_PATH variable in register-finetuning/utils/corpus_load.py to the proper directory", file=sys.stderr)
    sys.exit(1)

# load list of registers from register-corpus
reg_abbv_path = CORPUS_PATH / "info/reg_abbv.json"
if reg_abbv_path.exists():
    with open(reg_abbv_path) as reg_abbv_file:
        REG_ABBV2NAME = json.load(reg_abbv_file)
else:
    print(f"Register abbreviation JSON file not found at {reg_abbv_path}, check if path is correct or register-corpus path {CORPUS_PATH} is correct", file=sys.stderr)
    sys.exit(1)

REGISTERS = list(REG_ABBV2NAME.keys())
REG2ID = {reg : REGISTERS.index(reg) for reg in REGISTERS}


# custom dataset class for tokenized register-labeled texts
class RegisterDataset(Dataset):
    def __init__(self, encoded_texts : dict[str, list[torch.Tensor]], registers : list[int]) -> None:
        self.encoded_texts = encoded_texts
        self.registers = registers

    def __len__(self) -> int:
        return len(self.registers)

    def __getitem__(self, index : int) -> dict[str, torch.Tensor]:
        encoded_text = {k : v[index] for k, v in self.encoded_texts.items()}
        register = self.registers[index]
        return {**encoded_text, "labels": torch.tensor(register)}
    

def get_texts_regs(path : Path) -> tuple[list[str], list[str]]:
    path = CORPUS_PATH / "corpus" / path # path assumed to be relative to register-corpus/corpus/
    if path.exists():
        dataset = pd.read_csv(path, sep="\t")
    else:
        print(f"Corpus file not found at {path}, incorrect language specification or bad corpus filepath (see CORPUS_PATH in utils/corpus_load.py)", file=sys.stderr)
        sys.exit(1)

    texts = dataset.iloc[:,1].tolist()
    registers = dataset.iloc[:,0].tolist()
    return texts, registers


def load_data(paths : list[Path], model_checkpoint : str) -> Dataset:
    all_texts_regs = [get_texts_regs(path) for path in paths] # list of tuples of lists of texts and of registers
    min_len = min([len(texts) for texts, _ in all_texts_regs]) # smallest number of texts one path/corpus has

    texts = []
    registers = []
    for lang_texts, lang_regs in all_texts_regs: # random samples so equal amounts of texts from each path/corpus
        texts += sample(lang_texts, min_len)
        registers += sample(lang_regs, min_len)
    
    # tokenize texts
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    encoded_texts = dict(tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True))
    registers = [REG2ID[reg] for reg in registers]
    
    return RegisterDataset(encoded_texts, registers)