import sys
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

REGISTERS = ('IN', 'IP/OP', 'RN', 'JN', 'HI', 'LY', 'NID', 'AID')
REG2ID = {reg : REGISTERS.index(reg) for reg in REGISTERS}
CORPUS_FILEPATH = "../register-corpus/corpus"

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


def load_data(filepath : str, model_checkpoint : str) -> Dataset:
    try:
        dataset = pd.read_csv(CORPUS_FILEPATH + filepath, sep='\t')
    except FileNotFoundError as e:
        print(f"{str(e)}\n Corpus file not found, incorrect language specification", file=sys.stderr)
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    texts = dataset.iloc[:,1].tolist()
    encoded_texts = dict(tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True))
    registers = dataset.iloc[:,0].tolist()
    registers = [REG2ID[reg] for reg in registers]
    
    return RegisterDataset(encoded_texts, registers)

load_data("/test/al.tsv", "cis-lmu/glot500-base")
print("success!")