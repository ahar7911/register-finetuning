from argparse import ArgumentParser

import pandas as pd

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler

import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

#TODO arguments, early stopping and checkpointing

# CONSTANTS
registers = ['IN', 'IP/OP', 'RN', 'JN', 'HI', 'LY', 'NID', 'AID']
reg2id = {reg : registers.index(reg) for reg in registers}
id2reg = {v: k for k, v in reg2id.items()}

CORE_filepath = '../corpus/register-corpus/CORE/'
lang2tsv = {'en': f'{CORE_filepath}english/CORE_en.tsv',
            'fr': f'{CORE_filepath}french/CORE_fr.tsv',
            'sw': f'{CORE_filepath}swedish/CORE_sw.tsv',
            'fi': f'{CORE_filepath}finnish/CORE_fi.tsv',}
LANGS = lang2tsv.keys()

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


def load_data(filepath : str, model_checkpoint : str, local : bool=False, seed : int=42) -> tuple[DataLoader, DataLoader]:
    dataset = pd.read_csv(filepath, sep='\t')
    
    X_texts = dataset.iloc[:,1].tolist()
    y_regs = dataset.iloc[:,0].tolist()
    y_regs = [reg2id[reg] for reg in y_regs]
    
    X_train, X_test, y_train, y_test = train_test_split(X_texts, y_regs, test_size=0.2, random_state=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, local_files_only=local)
    train_dataset = RegisterDataset(texts=X_train, registers=y_train, tokenizer=tokenizer, max_len=512)
    test_dataset = RegisterDataset(texts=X_test, registers=y_test, tokenizer=tokenizer, max_len=512)

    train_dataloader = DataLoader(train_dataset, batch_size=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    return train_dataloader, test_dataloader


def get_metrics(num_classes : int, device : torch.device) -> dict[str, torchmetrics.Metric]:
    metrics = {'accuracy': MulticlassAccuracy(num_classes=num_classes),
               'precision': MulticlassPrecision(num_classes=num_classes),
               'recall': MulticlassRecall(num_classes=num_classes),
               'f1': MulticlassF1Score(num_classes=num_classes)}
    for metric in metrics.values():
        metric.to(device)
    return metrics


def train(model : transformers.PreTrainedModel, train_dataloader : DataLoader, num_epochs: int, 
          device : torch.device, optimizer : torch.optim.Optimizer, lr_scheduler, metrics : dict[str, torchmetrics.Metric]):
    print('TRAINING')
    for epoch in range(num_epochs):
        model.train()

        print(f"Epoch {epoch + 1} training:")
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            for metric in metrics.values():
                metric(preds, batch['labels'])

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()

        metric_summary = ""
        for name, metric in metrics.items():
            metric_summary += name + str(metric.compute().item()) + ", "
        print(f"Epoch {epoch+1}: {metric_summary}")
        
        for metric in metrics.values():
            metric.reset()


def evaluate(model : transformers.PreTrainedModel, test_dataloader : DataLoader, 
             device : torch.device, metrics : dict[str, torchmetrics.Metric]):
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k,v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
        
        outputs = outputs.logits
        preds = torch.argmax(outputs, dim=-1)
        
        for metric in metrics.values():
            metric(preds, batch['labels'])
    
    metric_summary = ""
    for name, metric in metrics.items():
        metric_summary += name + str(metric.compute().item()) + ", "
    print(f"EVALUATING: {metric_summary}")
        
    for metric in metrics.values():
        metric.reset()


def main(model : str, eval_lang : str, num_epochs=4):
    model2chckpt = {'mbert': 'google-bert/bert-base-multilingual-cased', 
                    'xlm-r': 'FacebookAI/xlm-roberta-base',
                    'glot500': 'cis-lmu/glot500-base'}
    
    num_classes = len(registers)
    checkpoint = model2chckpt[model]

    train_dataloader, test_dataloader = load_data(lang2tsv[eval_lang], checkpoint)

    classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    classifier.to(device)

    metrics = get_metrics(num_classes, device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    train(classifier, train_dataloader, num_epochs, device, optimizer, lr_scheduler, metrics)
    evaluate(classifier, test_dataloader, device, metrics)


if __name__ == '__main__':
    parser = ArgumentParser(prog='Register fine-tuning',
                            description='Finetinuning LLMs for multilingual classification of registers')
    parser.add_argument('--model', choices=["mbert", "xlm-r", "glot500"], required=True,
                        help="LLM to finetune")
    parser.add_argument('--train_langs', choices=LANGS,
                        help="Language(s) to finetune register on")
    parser.add_argument('--eval_lang', choices=LANGS, required=True,
                        help='Language to evaluate model on')
    parser.add_argument('--num_epochs',
                        help='Number of epochs to finetune model for')
    parser.add_argument('-freeze', action='store_true', 
                        help='Freeze all model layers except last couple and classification head')
    args = parser.parse_args()

    if args.num_epochs is not None:
        main(args.model, args.eval_lang, args.num_epochs)
    else:
        main(args.model, args.eval_lang)