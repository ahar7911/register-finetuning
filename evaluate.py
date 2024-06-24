from argparse import ArgumentParser
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import transformers
from transformers import AutoModelForSequenceClassification
import torch
import torchmetrics
from sklearn.metrics import confusion_matrix
import seaborn as sn

from utils.corpus_load import load_data, REGISTERS, CORPUS_FILEPATH
from utils.metrics import get_metrics, add_batch, get_metric_summary, reset_metrics

def evaluate(model : transformers.PreTrainedModel, test_dataloader : torch.utils.data.DataLoader, 
             device : torch.device, metrics : dict[str, torchmetrics.Metric],
             output_filepath : str, lang : str):
    model.eval()
    all_labels = []
    all_preds = []

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        outputs = outputs.logits
        preds = torch.argmax(outputs, dim=-1)
        add_batch(metrics, preds, batch["labels"])

        all_labels.append(batch["labels"].cpu())
        all_preds.append(preds.cpu())
    
    metric_summary = get_metric_summary(metrics)
    with open(output_filepath + f"{lang}.json", "w") as file:
        json.dump(metric_summary, file, indent=4)
    reset_metrics(metrics)

    # modified from https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
    cf_matrix = confusion_matrix(torch.cat(all_labels), torch.cat(all_preds), labels=range(len(REGISTERS)))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=REGISTERS, columns=REGISTERS)
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(output_filepath + f"{lang}.png")

def main(model_name : str, train_langs : str, eval_lang : str):
    with open("utils/model2chckpt.json") as file:
        model2chckpt = json.load(file)
    checkpoint = model2chckpt[model_name]

    num_labels = len(REGISTERS)
    eval_lang_tsv = f"{CORPUS_FILEPATH}/test/{eval_lang}.tsv"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    try:
        classifier = AutoModelForSequenceClassification.from_pretrained(f"./models/{model_name}-{train_langs}")
    except Exception as e:
        print(f"{str(e)}\n Model not found, no saved model has been trained on specified model/language", file=sys.stderr)
        sys.exit(1)
    classifier.to(device)

    test_dataloader = load_data(eval_lang_tsv, checkpoint, batch_size=64)
    metrics = get_metrics(num_labels, device)
    output_filepath = f"output/{model_name}/{train_langs}/eval/"
    
    evaluate(classifier, test_dataloader, device, metrics, output_filepath, eval_lang)

    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = ArgumentParser(prog="Evaluate register classification",
                            description="Evaluates multilingual model's ability to classify registers in one language")
    parser.add_argument("--model", choices=["mbert", "xlm-r", "glot500"], required=True,
                        help="Name of model to evaluate")
    parser.add_argument("--train_langs", required=True, 
                        help="Language(s) model was fine-tuned on; untrained model used if not specified")
    parser.add_argument("--eval_lang", required=True,
                        help="Language to evaluate fine-tuned model on")
    args = parser.parse_args()
    
    main(args.model, args.train_langs, args.eval_lang)