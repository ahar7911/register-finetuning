from argparse import ArgumentParser
import os
import time
import json

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForSequenceClassification, get_scheduler
import torchmetrics

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.corpus_load import load_data, REGISTERS
from utils.metrics import get_metrics, add_batch, get_metric_summary, reset_metrics

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
          rank : int,
          num_epochs: int,
          optimizer : torch.optim.Optimizer, 
          lr_scheduler : torch.optim.lr_scheduler.LambdaLR, 
          metrics : dict[str, torchmetrics.Metric], 
          output_file_str : str
          ) -> None:

    train_summary = {}
    train_start_time = time.time()
    print(f"gpu{rank}: start training")

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()

        for batch in train_dataloader:       
            batch = {k: v.to(rank) for k, v in batch.items()}

            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            add_batch(metrics, preds, batch["labels"])

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
        
        epoch_str = f"epoch {epoch + 1}"
        epoch_time = time.time() - epoch_start_time
        print(f"gpu{rank}: {epoch_str} | loss: {loss} | time: {int(epoch_time // 60)}m{epoch_time % 60:.2f}s")

        train_summary[epoch_str] = get_metric_summary(metrics)
        reset_metrics(metrics)

    total_time = time.time() - train_start_time
    print(f"gpu{rank}: end training | total time: {int(total_time // 60)}m{total_time % 60:.2f}s")
    
    with open(output_file_str, "w") as file:
        json.dump(train_summary, file, indent=4)


def main(rank : int, 
         world_size: int, 
         model_name : str, 
         train_langs : str, 
         num_epochs : int,
         batch_size : int,
         ) -> None:
    ddp_setup(rank, world_size)

    with open("utils/model2chckpt.json") as file:
        model2chckpt = json.load(file)

    checkpoint = model2chckpt[model_name]
    num_labels = len(REGISTERS)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    train_lang_tsv = f"/train/{train_langs}.tsv"
    train_dataset = load_data(train_lang_tsv, checkpoint)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                                 pin_memory=True, 
                                                 sampler=DistributedSampler(train_dataset))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    metrics = get_metrics(num_labels, rank)
    output_filepath = f"output/{model_name}-{train_langs}/train.json"

    train(model, train_dataloader, rank, num_epochs, optimizer, lr_scheduler, metrics, output_filepath)
    model.module.save_pretrained(f"./models/{model_name}-{train_langs}/", from_pt=True)

    destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(prog="Register fine-tuning",
                            description="Fine-tuning LLMs for multilingual classification of registers")
    parser.add_argument("--model", choices=["mbert", "xlm-r", "glot500"], required=True,
                        help="LLM to finetune")
    parser.add_argument("--train_langs", required=True,
                        help="Language(s) to finetune register classification on")
    parser.add_argument("--num_epochs", default=5, type=int,
                        help="Number of epochs to finetune model for (default: 5)")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Size of each training batch (default: 16)")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"world size (# of gpus): {world_size}")

    mp.spawn(main, args=(world_size, args.model, args.train_langs, args.num_epochs, args.batch_size), nprocs=world_size)