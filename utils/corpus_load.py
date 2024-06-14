import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

REGISTERS = ['IN', 'IP/OP', 'RN', 'JN', 'HI', 'LY', 'NID', 'AID']
REG2ID = {reg : REGISTERS.index(reg) for reg in REGISTERS}

class RegisterDataset(Dataset):
    def __init__(self, texts, registers, tokenizer, max_len):
        self.texts = texts
        self.registers = registers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index : int):
        text = str(self.texts[index])
        register = self.registers[index]

        # https://huggingface.co/docs/transformers/v4.41.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__
        encoded_text = self.tokenizer(text,
                                  max_length = self.max_len,
                                  return_token_type_ids = False,
                                  return_attention_mask = True,
                                  return_tensors = "pt",
                                  padding = "max_length",
                                  truncation=True)
        return {'input_ids': encoded_text['input_ids'][0],
            'attention_mask': encoded_text['attention_mask'][0],
            'labels': torch.tensor(register, dtype=torch.long)}


def load_data(filepath : str, model_checkpoint : str, local : bool=False, batch_size : int=16) -> DataLoader:
    dataset = pd.read_csv(filepath, sep='\t')
    
    X_texts = dataset.iloc[:,1].tolist()
    y_regs = dataset.iloc[:,0].tolist()
    y_regs = [REG2ID[reg] for reg in y_regs]

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, local_files_only=local)
    dataset = RegisterDataset(texts=X_texts, registers=y_regs, tokenizer=tokenizer, max_len=512)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader