import json
from huggingface_hub import snapshot_download

with open('model2chckpt.json') as file:
    model2chckpt = json.load(file)

for chckpt in model2chckpt.values():
    snapshot_download(repo_id=chckpt)