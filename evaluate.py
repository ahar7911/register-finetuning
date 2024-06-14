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

def main(model_path : str, lang_tsv : str):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    test_dataloader = load_data(lang_tsv, model_path, local=True, batch_size=64)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    metrics = get_metrics()

    evaluate(model, test_dataloader, device, metrics)
    

if __name__ == '__main__':
    with open('utils/lang2tsv.json') as file:
        lang2tsv = json.load(file)
    
    parser = ArgumentParser(prog="Evaluate register classification",
                            description="Given a model and a language, evaluates the model's ability to classify registers in that language")
    parser.add_argument('--model', required=True,
                        help='Local path to fine-tuned model to evaluate')
    parser.add_argument('--lang', choices=lang2tsv.keys(), required=True,
                        help='Language to evaluate fine-tuned model on')
    args = parser.parse_args()
    
    lang_tsv = lang2tsv[args.lang]
    main(args.model, lang_tsv)