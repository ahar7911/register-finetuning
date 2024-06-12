from huggingface_hub import snapshot_download

snapshot_download(repo_id="google-bert/bert-base-multilingual-cased", local_dir="models/mbert")