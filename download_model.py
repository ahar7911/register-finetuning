from huggingface_hub import snapshot_download

models = ['google-bert/bert-base-multilingual-cased', 
          'FacebookAI/xlm-roberta-base', 
          'cis-lmu/glot500-base']

for chckpt in models:
    snapshot_download(repo_id=chckpt)