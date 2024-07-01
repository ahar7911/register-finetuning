from argparse import ArgumentParser
import time
import json

import transformers
from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader

from utils.corpus_load import load_data, REGISTERS
from utils.metrics import Metrics, save_cf_matrix

def evaluate(model : transformers.PreTrainedModel, 
             test_dataloader : DataLoader, 
             device : torch.device, 
             metrics : Metrics,
             output_filepath : str,
             eval_lang : str
             ) -> None:
    all_labels = []
    all_preds = []
    
    model.eval()
    eval_start_time = time.time()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
        
        outputs = outputs.logits
        preds = torch.argmax(outputs, dim=-1)
        metrics.add_batch(preds, batch["labels"])

        all_labels.append(batch["labels"].cpu())
        all_preds.append(preds.cpu())

    total_time = time.time() - eval_start_time
    print(f"end evaluation | total time: {int(total_time // 60)}m{total_time % 60:.2f}s")
    
    metrics.write_summary(output_filepath + "eval.json", eval_lang)
    metrics.reset()

    save_cf_matrix(torch.cat(all_preds), torch.cat(all_labels), output_filepath + f"cfm/{eval_lang}.png")

    print("metrics and cf matrix saved")


def main(model_name : str, train_langs : str, eval_lang : str) -> None:
    with open("utils/model2chckpt.json") as file:
        model2chckpt = json.load(file)
    
    checkpoint = model2chckpt[model_name]
    num_labels = len(REGISTERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        classifier = AutoModelForSequenceClassification.from_pretrained(f"./models/{model_name}-{train_langs}")
    except Exception as e:
        raise RuntimeError(f"Model not found, incorrect model name {model_name} or no saved model has been trained on specified language(s) {train_langs}") from e
    classifier.to(device)

    eval_lang_tsv = f"/test/{eval_lang}.tsv"
    test_dataset = load_data(eval_lang_tsv, checkpoint)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    metrics = Metrics(num_labels, device)
    output_filepath = f"output/{model_name}-{train_langs}/"
    
    evaluate(classifier, test_dataloader, device, metrics, output_filepath, eval_lang)
    

if __name__ == "__main__":
    parser = ArgumentParser(prog="Evaluate register classification",
                            description="Evaluates multilingual model's ability to classify registers in one language")
    parser.add_argument("--model", choices=["mbert", "xlm-r", "glot500"], required=True,
                        help="Name of model to evaluate")
    parser.add_argument("--train_langs", required=True, 
                        help="Language(s) model was fine-tuned on")
    parser.add_argument("--eval_lang", required=True,
                        help="Language to evaluate fine-tuned model on")
    args = parser.parse_args()
    
    main(args.model, args.train_langs, args.eval_lang)