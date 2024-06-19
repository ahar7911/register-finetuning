from argparse import ArgumentParser
import json

import transformers
from transformers import AutoModelForSequenceClassification
import torch
import torchmetrics
from torch.utils.data import DataLoader

from utils.corpus_load import load_data, REGISTERS, CORPUS_FILEPATH
from utils.metrics import get_metrics, add_batch, get_metric_summary, reset_metrics

def evaluate(model : transformers.PreTrainedModel, test_dataloader : DataLoader, 
             device : torch.device, metrics : dict[str, torchmetrics.Metric],
             output_filepath : str):
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k,v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
        
        outputs = outputs.logits
        preds = torch.argmax(outputs, dim=-1)
        add_batch(metrics, preds, batch['labels'])
    
    metric_summary = get_metric_summary(metrics)

    with open(output_filepath, 'w') as file:
        json.dump(metric_summary, file, indent=4)
        
    reset_metrics(metrics)

def main(model_name : str, train_langs : str, eval_lang : str):
    with open('utils/model2chckpt.json') as file:
        model2chckpt = json.load(file)
    checkpoint = model2chckpt[model_name]

    num_labels = len(REGISTERS)
    eval_lang_tsv = f"{CORPUS_FILEPATH}/test/{eval_lang}.tsv"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if train_langs is not None:
        classifier = AutoModelForSequenceClassification.from_pretrained(f'./models/mbert-{train_langs}')
    else:
        classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    classifier.to(device)

    test_dataloader = load_data(eval_lang_tsv, checkpoint, batch_size=64)
    metrics = get_metrics(num_labels, device)
    if train_langs is not None:
        output_filepath = f'output/{model_name}-{train_langs}-eval-{eval_lang}.json'
    else:
        output_filepath = f'output/{model_name}-eval-{eval_lang}.json'
    

    evaluate(classifier, test_dataloader, device, metrics, output_filepath)
    

if __name__ == '__main__':
    parser = ArgumentParser(prog="Evaluate register classification",
                            description="Evaluates multilingual model's ability to classify registers in one language")
    parser.add_argument('--model', required=True,
                        help='Name of model to evaluate')
    parser.add_argument('--train_langs',
                        help='Language(s) model was fine-tuned on; untrained model used if not specified')
    parser.add_argument('--eval_lang', required=True,
                        help='Language to evaluate fine-tuned model on')
    args = parser.parse_args()
    
    main(args.model, args.train_langs, args.eval_lang)