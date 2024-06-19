from argparse import ArgumentParser
import json

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForSequenceClassification, get_scheduler
import torchmetrics

from utils.corpus_load import load_data, REGISTERS, CORPUS_FILEPATH
from utils.metrics import get_metrics, add_batch, get_metric_summary, reset_metrics

def train(model : transformers.PreTrainedModel, train_dataloader : DataLoader, num_epochs: int, 
          device : torch.device, optimizer : torch.optim.Optimizer, lr_scheduler : torch.optim.lr_scheduler.LambdaLR, 
          metrics : dict[str, torchmetrics.Metric], output_file_str : str):

    train_summary = {}
    for epoch in range(num_epochs):
        model.train()
        epoch_str = f'epoch {epoch + 1}'

        print(f"{epoch_str} training")
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            add_batch(metrics, preds, batch['labels'])

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()

        train_summary[epoch_str] = get_metric_summary(metrics)
        reset_metrics(metrics)
    
    with open(output_file_str, 'w') as file:
        json.dump(train_summary, file, indent=4)


def main(model_name : str, train_langs : str, num_epochs : int):
    with open('utils/model2chckpt.json') as file:
        model2chckpt = json.load(file)
    checkpoint = model2chckpt[model_name]

    num_labels = len(REGISTERS)
    train_lang_tsv = f"{CORPUS_FILEPATH}/train/{train_langs}.tsv"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    classifier.to(device)

    train_dataloader = load_data(train_lang_tsv, checkpoint, is_train=True, batch_size=16)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    metrics = get_metrics(num_labels, device)
    output_filepath = f'output/{model_name}-train-{train_langs}.json'

    train(classifier, train_dataloader, num_epochs, device, optimizer, lr_scheduler, metrics, output_filepath)
    classifier.save_pretrained(f'./models/mbert-{train_langs}/', from_pt=True)


if __name__ == '__main__':
    parser = ArgumentParser(prog='Register fine-tuning',
                            description='Fine-tuning LLMs for multilingual classification of registers')
    parser.add_argument('--model', choices=["mbert", "xlm-r", "glot500"], required=True,
                        help="LLM to finetune")
    parser.add_argument('--train_langs', required=True,
                        help="Language(s) to finetune register classification on")
    parser.add_argument('--num_epochs', default=4,
                        help='Number of epochs to finetune model for.')
    parser.add_argument('--freeze', action='store_true', 
                        help='Freeze all model layers except last couple and classification head')
    args = parser.parse_args()

    if args.num_epochs is not None:
        main(args.model, args.train_langs, args.num_epochs)
    else:
        main(args.model, args.train_langs)