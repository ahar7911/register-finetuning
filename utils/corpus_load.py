import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

REGISTERS = ['IN', 'IP/OP', 'RN', 'JN', 'HI', 'LY', 'NID', 'AID']
REG2ID = {reg : REGISTERS.index(reg) for reg in REGISTERS}

class RegisterDataset(Dataset):
    def __init__(self, encoded_texts : dict[str, list[torch.Tensor]], registers : list[int]):
        self.encoded_texts = encoded_texts
        self.registers = registers

    def __len__(self):
        return len(self.registers)

    def __getitem__(self, index : int):
        encoded_text = {k : v[index] for k, v in self.encoded_texts.items()}
        register = self.registers[index]
        return {**encoded_text, 'labels': torch.tensor(register, dtype=torch.long)}


def load_data(filepath : str, model_checkpoint : str, batch_size : int=16) -> DataLoader:
    dataset = pd.read_csv(filepath, sep='\t')
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    texts = dataset.iloc[:,1].tolist()
    encoded_texts = dict(tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True))
    regs = dataset.iloc[:,0].tolist()
    regs = [REG2ID[reg] for reg in regs]
    
    dataset = RegisterDataset(encoded_texts, regs)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader