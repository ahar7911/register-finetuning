from argparse import ArgumentParser

import json

import transformers
from transformers import AutoModelForSequenceClassification
import torch
import torchmetrics
from torch.utils.data import DataLoader

from utils.corpus_load import load_data
from utils.metrics import get_metrics

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
        
        for metric in metrics.values():
            metric(preds, batch['labels'])
    
    metric_summary = {}
    for name, metric in metrics.items():
        metric_summary = {**metric_summary, name : metric.compute()}

    with open(output_filepath, 'w') as file:
        json.dump(file, metric_summary)
        
    for metric in metrics.values():
        metric.reset()

def main(model : str, train_langs : str, eval_lang : str, lang2tsv : dict[str, str]):
    eval_lang_tsv = lang2tsv[eval_lang]
    output_filepath = f'output/output-{train_langs}-{eval_lang}.json'

    with open('utils/model2chckpt.json') as file:
        model2chckpt = json.load(file)
    checkpoint = model2chckpt[model]
    
    model = AutoModelForSequenceClassification.from_pretrained(f'./models/mbert-{train_langs}')
    test_dataloader = load_data(eval_lang_tsv, checkpoint, local=True, batch_size=64)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    metrics = get_metrics()

    evaluate(model, test_dataloader, device, metrics, output_filepath)
    

if __name__ == '__main__':
    with open('utils/lang2tsv.json') as file:
        lang2tsv = json.load(file)
    
    parser = ArgumentParser(prog="Evaluate register classification",
                            description="Given a model and a language, evaluates the model's ability to classify registers in that language")
    parser.add_argument('--model', required=True,
                        help='Name of fine-tuned model to evaluate')
    parser.add_argument('--train_langs', required=True,
                        help='Languages model was fine-tuned on')
    parser.add_argument('--eval_lang', choices=lang2tsv.keys(), required=True,
                        help='Language to evaluate fine-tuned model on')
    args = parser.parse_args()
    
    
    main(args.model, args.train_langs, args.eval_lang, lang2tsv)