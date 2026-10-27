"""Build-time only: fetch model weights into the image."""
import sys
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=dest,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model",
                                  "tokenizer*", "merges.txt", "vocab*"])
print("downloaded", repo, "->", dest)
