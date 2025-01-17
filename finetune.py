from argparse import ArgumentParser
import os
import time
from pathlib import Path
import json
import math
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler
from transformers import AutoModelForSequenceClassification, get_scheduler
from sklearn.utils.class_weight import compute_class_weight

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.corpus_load import load_data, REGISTERS
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

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        metrics.reset()

        for batch in train_dataloader:       
            batch = {k: v.to(rank) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda"):
                outputs = model(**batch)
                if loss_fn is None:
                    loss = outputs.loss
                else:
                    loss = loss_fn(outputs.logits, batch["labels"])
            
            preds = torch.argmax(outputs.logits, dim=-1)
            metrics.add_batch(preds, batch["labels"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        lr_scheduler.step() # once per epoch
        
        epoch_str = f"gpu{rank}: epoch {epoch + 1}"
        epoch_time = time.time() - epoch_start_time
        print(f"{epoch_str} | loss: {loss} | time: {int(epoch_time // 60)}m{epoch_time % 60:.2f}s")

        metrics.write_summary(out_path, epoch_str)
        
        evaluate(model, val_dataloader, rank, metrics, out_path, metric_key=epoch_str + ": validation", store_cfm=False)
        print()

    total_time = time.time() - train_start_time
    print(f"gpu{rank}: end training | total time: {int(total_time // 60)}m{total_time % 60:.2f}s")
    print()


def get_weights(train_labels : list[int], num_labels : int) -> torch.tensor:
    present_classes = sorted(np.unique(train_labels))
    weights = compute_class_weight("balanced", classes=present_classes, y=train_labels)

    if len(present_classes) != num_labels: # one or more classes not present in training data
        class2weight = dict(zip(present_classes, weights))
        for curr_class in range(num_labels):
            if curr_class not in class2weight:
                class2weight[curr_class] = 0
        weights = [class2weight[c] for c in range(num_labels)]

    return torch.tensor(weights, dtype=torch.float)


def main(rank : int, 
         world_size: int, 
         model_name : str, 
         train_langs : str,
         subfolder : str, 
         balanced : bool,
         num_epochs : int,
         batch_size : int,
         lr : float
         ) -> None:
    ddp_setup(rank, world_size)

    with open(Path("utils/model2chckpt.json")) as file:
        model2chckpt = json.load(file)
    checkpoint = model2chckpt[model_name]

    # load model info
    num_labels = len(REGISTERS)
    model_str = f"{model_name}-{train_langs}"
    if subfolder is not None:
        model_str = f"{subfolder}/{model_str}"

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    train_lang_tsvs = [Path(f"train/{train_lang}.tsv") for train_lang in train_langs.split("-")]
    dataset = load_data(train_lang_tsvs, checkpoint)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  sampler=DistributedSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,  
                                sampler=DistributedSampler(val_dataset))

    if num_epochs is None: # if number of epochs unspecified, run as many epochs for the model to see 50000 examples/texts
        num_examples = len(train_dataset)
        num_epochs = math.ceil(50000 / num_examples)
        print(f"{num_examples} examples in training dataset, running for {num_epochs} epochs")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    metrics = Metrics(num_labels, rank)

    out_path = Path(f"output/{model_str}/train.json")
    out_path.unlink(missing_ok=True) # removes train.json if it already exists
    (out_path.parent / "eval.json").unlink(missing_ok=True) # removes eval.json if it already exists

    loss_fn = None
    if balanced:
        train_labels = [item["labels"].item() for item in train_dataset]
        weights = get_weights(train_labels, num_labels)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss_fn.to(rank)

    train(model, train_dataloader, val_dataloader, rank, num_epochs, optimizer, lr_scheduler, metrics, out_path, loss_fn)
    model.module.save_pretrained(Path(f"./models/{model_str}/"), from_pt=True) # creates necessary subfolders if required

    destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(prog="Register fine-tuning",
                            description="Fine-tuning LLMs for multilingual classification of registers")
    parser.add_argument("--model_name", choices=["mbert", "xlmr", "glot500"], required=True,
                        help="LLM to finetune")
    parser.add_argument("--train_langs", required=True,
                        help="Language(s) to finetune register classification on, multiple languages must be separated by '-'")
    parser.add_argument("--subfolder",
                        help="Training outputs will be saved to output/subfolder/ and model will be saved to models/subfolder/")
    parser.add_argument("--balanced", action="store_true",
                        help="Whether model will train such that each class is weighted equally or not")
    parser.add_argument("--num_epochs", type=int,
                        help="Number of epochs to finetune model for (default: variable such that model sees 50,000 examples)")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Size of each training batch (default: 16)")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="Learning rate for AdamW optimizer when training (default: 1e-5)")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"world size (# of gpus): {world_size}")

    main_args = (world_size,) + tuple(vars(args).values())
    mp.spawn(main, args=main_args, nprocs=world_size)