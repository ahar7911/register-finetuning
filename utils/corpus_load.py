import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

REGISTERS = ['IN', 'IP/OP', 'RN', 'JN', 'HI', 'LY', 'NID', 'AID']
REG2ID = {reg : REGISTERS.index(reg) for reg in REGISTERS}

class RegisterDataset(Dataset):
    def __init__(self, texts, registers, tokenizer):
        self.texts = texts
        self.registers = registers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index : int):
        text = str(self.texts[index])
        register = self.registers[index]
        encoded_text = self.tokenizer(text, return_tensors = "pt", padding = "max_length")
        return {**encoded_text, 'labels' : torch.tensor(register, dtype=torch.long)}


def load_data(filepath : str, model_checkpoint : str, batch_size : int=16) -> DataLoader:
    dataset = pd.read_csv(filepath, sep='\t')
    
    X_texts = dataset.iloc[:,1].tolist()
    y_regs = dataset.iloc[:,0].tolist()
    y_regs = [REG2ID[reg] for reg in y_regs]

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    dataset = RegisterDataset(texts=X_texts, registers=y_regs, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader