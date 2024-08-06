from argparse import ArgumentParser
import sys
import time
from pathlib import Path
import json

import transformers
from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader

from utils.corpus_load import load_data, REGISTERS
from utils.metrics import Metrics, save_cfm

def evaluate(model : transformers.PreTrainedModel, 
             test_dataloader : DataLoader, 
             device : torch.device, 
             metrics : Metrics,
             out_path : str,
             metric_key : str,
             store_cfm : bool = True,
             eval_lang : str = None
             ) -> None:
    if store_cfm:
        if eval_lang is None:
            print("eval_lang cannot be None when store_cfm is true, ending evaluation", file=sys.stderr)
            return None
        all_labels = []
        all_preds = []
    
    model.eval()
    eval_start_time = time.time()
    metrics.reset()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
        
        outputs = outputs.logits
        preds = torch.argmax(outputs, dim=-1)
        metrics.add_batch(preds, batch["labels"])

        if store_cfm:
            all_labels.append(batch["labels"].cpu())
            all_preds.append(preds.cpu())

    total_time = time.time() - eval_start_time
    print(f"end evaluation | total time: {int(total_time // 60)}m{total_time % 60:.2f}s")
    
    metrics.write_summary(out_path, metric_key)
    print("metrics saved")

    if store_cfm:
        cfm_path = out_path.parent / f"cfm/{eval_lang}.json"
        save_cfm(torch.cat(all_preds), torch.cat(all_labels), cfm_path)
        print("confusion matrices saved")


def main(model_name : str, 
         train_langs : str, 
         eval_lang : str, 
         subfolder : str
         ) -> None:
    with open(Path("utils/model2chckpt.json")) as file:
        model2chckpt = json.load(file)
    
    checkpoint = model2chckpt[model_name]
    num_labels = len(REGISTERS)
    model_str = f"{model_name}-{train_langs}"
    if subfolder is not None:
        model_str = f"{subfolder}/{model_str}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        classifier = AutoModelForSequenceClassification.from_pretrained(f"./models/{model_str}")
    except:
        print(f"""model not found, incorrect model name {model_name} 
              or no saved model has been trained on specified language(s) {train_langs}. 
              maybe check the order of your language strings?""", file=sys.stderr)
        sys.exit(1)
    classifier.to(device)

    eval_lang_tsv = Path(f"test/{eval_lang}.tsv")
    test_dataset = load_data([eval_lang_tsv], checkpoint)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    
    metrics = Metrics(num_labels, device)
    out_path = Path(f"output/{model_str}/eval.json")
    
    evaluate(classifier, test_dataloader, device, metrics, out_path, metric_key=eval_lang, eval_lang=eval_lang)
    

if __name__ == "__main__":
    parser = ArgumentParser(prog="Evaluate register classification",
                            description="Evaluates multilingual model's ability to classify registers in one language")
    parser.add_argument("--model_name", choices=["mbert", "xlmr", "glot500"], required=True,
                        help="Name of model to evaluate")
    parser.add_argument("--train_langs", required=True, 
                        help="Language(s) model was fine-tuned on, multiple languages must be separated by '-'")
    parser.add_argument("--eval_lang", required=True,
                        help="Language to evaluate fine-tuned model on")
    parser.add_argument("--subfolder",
                        help="Evaluation outputs will be saved to output/subfolder/ and model will be taken from models/subfolder/")
    args = parser.parse_args()
    args_dict = vars(args)
    
    main(**args_dict)