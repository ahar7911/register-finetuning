from argparse import ArgumentParser
import sys
import os
import time
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForSequenceClassification, get_scheduler

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.corpus_load import load_data, CORPUS_PATH, REGISTERS
from utils.metrics import Metrics
from evaluate import evaluate

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train(model : DDP, 
          train_dataloader : DataLoader,
          val_dataloader : DataLoader, 
          rank : int,
          num_epochs: int,
          optimizer : torch.optim.Optimizer, 
          lr_scheduler : torch.optim.lr_scheduler.LambdaLR, 
          metrics : Metrics, 
          out_path : Path,
          loss_fn : torch.nn.CrossEntropyLoss = None
          ) -> None:

    train_start_time = time.time()
    print(f"gpu{rank}: start training")

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        metrics.reset()

        for batch in train_dataloader:       
            batch = {k: v.to(rank) for k, v in batch.items()}

            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            metrics.add_batch(preds, batch["labels"])

            if loss_fn is None:
                loss = outputs.loss
            else:
                loss = loss_fn(outputs.logits, batch["labels"])
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
        
        epoch_str = f"epoch {epoch + 1}"
        epoch_time = time.time() - epoch_start_time
        print(f"gpu{rank}: {epoch_str} | loss: {loss} | time: {int(epoch_time // 60)}m{epoch_time % 60:.2f}s")

        metrics.write_summary(out_path, epoch_str)
        
        evaluate(model, val_dataloader, rank, metrics, out_path, metric_key=epoch_str + ": validation", store_cfm=False)

    total_time = time.time() - train_start_time
    print(f"gpu{rank}: end training | total time: {int(total_time // 60)}m{total_time % 60:.2f}s")


def get_weights(train_lang):
    summary_path = CORPUS_PATH / f"summaries/{train_lang}.json"
    if summary_path.exists():
        with open(summary_path) as summary_file:
            summary_dict = json.load(summary_file)["counts"]
    else:
        print(f"given train lang {train_lang}, but could not find corresponding summary file at {summary_path}.", file=sys.stderr)
        print("please double check that CORPUS_PATH in utils/corpus_load.py correctly directs to register-corpus location and/or that register-corpus/analyze_dist.py has been run", file=sys.stderr)
        sys.exit(1)

    total = sum(summary_dict.values())
    weights = []
    for count in summary_dict.values():
        if count == 0:
            weights.append(0)
        else:
            weights.append(total / count)
    
    return torch.tensor(weights)


def main(rank : int, 
         world_size: int, 
         model_name : str, 
         train_langs : str, 
         balanced : bool,
         num_epochs : int,
         batch_size : int,
         ) -> None:
    ddp_setup(rank, world_size)

    with open(Path("utils/model2chckpt.json")) as file:
        model2chckpt = json.load(file)

    checkpoint = model2chckpt[model_name]
    num_labels = len(REGISTERS)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    train_lang_tsv = Path(f"train/{train_langs}.tsv")
    dataset = load_data(train_lang_tsv, checkpoint)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  sampler=DistributedSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size, 
                                pin_memory=True, 
                                sampler=DistributedSampler(val_dataset))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    metrics = Metrics(num_labels, rank)

    loss_fn = None
    if balanced:
        weights = get_weights(train_langs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights).to(rank)

    out_path = Path(f"output/{model_name}-{train_langs}/train.json")
    out_path.parent.mkdir(parents=True, exist_ok=True) # makes output dir
    out_path.unlink(missing_ok=True) # removes train.json if it already exists

    train(model, train_dataloader, val_dataloader, rank, num_epochs, optimizer, lr_scheduler, metrics, out_path, loss_fn)
    model.module.save_pretrained(Path(f"./models/{model_name}-{train_langs}/"), from_pt=True) # creates necessary subfolders if required

    destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(prog="Register fine-tuning",
                            description="Fine-tuning LLMs for multilingual classification of registers")
    parser.add_argument("--model", choices=["mbert", "xlmr", "glot500"], required=True,
                        help="LLM to finetune")
    parser.add_argument("--train_langs", required=True,
                        help="Language(s) to finetune register classification on")
    parser.add_argument("--balanced", action="store_true",
                        help="Whether model will train such that each class is weighted equally or not")
    parser.add_argument("--num_epochs", default=5, type=int,
                        help="Number of epochs to finetune model for (default: 5)")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Size of each training batch (default: 16)")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"world size (# of gpus): {world_size}")

    mp.spawn(main, args=(world_size, args.model, args.train_langs, args.balanced, args.num_epochs, args.batch_size), nprocs=world_size)