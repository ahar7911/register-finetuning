from argparse import ArgumentParser

import json

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoModelForSequenceClassification, get_scheduler

import torchmetrics

from utils.corpus_load import load_data, REGISTERS
from utils.metrics import get_metrics

#TODO arguments, early stopping and checkpointing


def train(model : transformers.PreTrainedModel, train_dataloader : DataLoader, num_epochs: int, 
          device : torch.device, optimizer : torch.optim.Optimizer, lr_scheduler, metrics : dict[str, torchmetrics.Metric],
          output_file_str : str):
    print('TRAINING')
    train_summary = {}
    for epoch in range(num_epochs):
        model.train()
        epoch_str = f'epoch {epoch + 1}'

        print(f"{epoch_str} training")
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

        metric_summary = {}
        for name, metric in metrics.items():
            metric_summary = {**metric_summary, name : metric.compute()}
        train_summary[epoch_str] = metric_summary
        
        for metric in metrics.values():
            metric.reset()
    
    with open(output_file_str, 'w') as file:
        json.dump(file, train_summary)


def main(model_name : str, train_langs : str, lang2tsv : dict[str, str], num_epochs=4):
    with open('utils/model2chckpt.json') as file:
        model2chckpt = json.load(file)
    checkpoint = model2chckpt[model_name]

    num_classes = len(REGISTERS)
    train_lang_tsv = lang2tsv[train_langs]
    output_filepath = f'output/output-{train_langs}.json'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_classes)
    classifier.to(device)

    train_dataloader = load_data(train_lang_tsv, checkpoint, batch_size=16)
    metrics = get_metrics(num_classes, device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    train(classifier, train_dataloader, num_epochs, device, optimizer, lr_scheduler, metrics, output_filepath)
    classifier.save_pretrained(f'./models/mbert-{train_langs}/', from_pt=True)


if __name__ == '__main__':
    with open('utils/lang2tsv.json') as file:
        lang2tsv = json.load(file)
    
    parser = ArgumentParser(prog='Register fine-tuning',
                            description='Fine-tuning LLMs for multilingual classification of registers')
    parser.add_argument('--model', choices=["mbert", "xlm-r", "glot500"], required=True,
                        help="LLM to finetune")
    parser.add_argument('--train_langs', choices=lang2tsv.keys(), required=True,
                        help="Language(s) to finetune register on")
    parser.add_argument('--num_epochs',
                        help='Number of epochs to finetune model for')
    parser.add_argument('-freeze', action='store_true', 
                        help='Freeze all model layers except last couple and classification head')
    args = parser.parse_args()

    if args.num_epochs is not None:
        main(args.model, args.train_langs, lang2tsv, args.num_epochs)
    else:
        main(args.model, args.train_langs, lang2tsv)