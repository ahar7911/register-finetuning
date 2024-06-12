import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

#TODO arguments, early stopping and checkpointing

# CONSTANTS
DATA_FILEPATH = '../corpus/register-corpus/CORE/french/CORE_fr.tsv'
NUM_EPOCHS = 2
registers = ['IN', 'IP/OP', 'RN', 'JN', 'HI', 'LY', 'NID', 'AID']
reg2id = {reg : registers.index(reg) for reg in registers}
id2reg = {v: k for k, v in reg2id.items()}

print("DATA PROCESSING")

dataset = pd.read_csv(DATA_FILEPATH, sep='\t')
X_texts = dataset.iloc[:,1].tolist()
y_regs = dataset.iloc[:,0].tolist()
y_regs = [reg2id[reg] for reg in y_regs]

X_train, X_test, y_train, y_test = train_test_split(X_texts, y_regs,
                                                    test_size=0.2, random_state=42)

class RegisterDataset(Dataset):
  def __init__(self, texts, registers, tokenizer, max_len):
    self.texts = texts
    self.registers = registers
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, index : int):
    text = str(self.texts[index])
    register = self.registers[index]

    # https://huggingface.co/docs/transformers/v4.41.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__
    encoded_text = self.tokenizer(text,
                                  max_length = self.max_len,
                                  return_token_type_ids = False,
                                  return_attention_mask = True,
                                  return_tensors = "pt",
                                  padding = "max_length",
                                  truncation=True)
    return {'input_ids': encoded_text['input_ids'][0],
            'attention_mask': encoded_text['attention_mask'][0],
            'labels': torch.tensor(register, dtype=torch.long)}

#'distilbert/distilbert-base-multilingual-cased'
# google-bert/bert-base-multilingual-cased
#"cis-lmu/glot500-base"
checkpoint = "./models/mbert" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

train_dataset = RegisterDataset(texts=X_train, registers=y_train, tokenizer=tokenizer, max_len=512)
test_dataset = RegisterDataset(texts=X_test, registers=y_test, tokenizer=tokenizer, max_len=512)

train_dataloader = DataLoader(train_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

print("FINE-TUNING")

classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(registers), local_files_only=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classifier.to(device)

accuracy = MulticlassAccuracy(num_classes=len(registers))
precision = MulticlassPrecision(num_classes=len(registers))
recall = MulticlassRecall(num_classes=len(registers))
f1 = MulticlassRecall(num_classes=len(registers))

metrics = [accuracy, precision, recall, f1]
for metric in metrics:
  metric.to(device)

optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
  "linear",
  optimizer=optimizer,
  num_warmup_steps=50,
  num_training_steps=len(train_dataloader) * NUM_EPOCHS
)

for epoch in range(NUM_EPOCHS):
  classifier.train()

  print(f"Epoch {epoch + 1} training:")

  for i, batch in enumerate(train_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = classifier(**batch)

    preds = torch.argmax(outputs.logits, dim=-1)
    for metric in metrics:
      metric(preds, batch['labels'])

    loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()

    optimizer.zero_grad()

  print(f"Epoch {epoch+1}: accuracy={metrics[0].compute()}, precision={metrics[1].compute()}, recall={metrics[2].compute()}, f1={metrics[3].compute()},")
  for metric in metrics:
    metric.reset()

print("EVALUATION")

classifier.eval()

for batch in test_dataloader:
  batch = {k: v.to(device) for k,v in batch.items()}

  with torch.no_grad():
    outputs = classifier(**batch)

  outputs = outputs.logits
  preds = torch.argmax(outputs, dim=-1)

  for metric in metrics:
    metric(preds, batch['labels'])

print(f"accuracy={metrics[0].compute()}, precision={metrics[1].compute()}, recall={metrics[2].compute()}, f1={metrics[3].compute()},")
for metric in metrics:
  metric.reset()