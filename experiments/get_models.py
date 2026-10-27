from huggingface_hub import snapshot_download, list_repo_files
repo = "PINEAPPLEANANAS/SpeechCorrection"
files = list_repo_files(repo)
print(len(files), "files in repo; top-level:", sorted({f.split('/')[0] for f in files}))
print([f for f in files if "checkpoint" not in f])
p = snapshot_download(repo, local_dir="models/SpeechCorrection",
                      allow_patterns=["whisper-torgo/*", "gec-model/*"],
                      ignore_patterns=["*checkpoint-*", "*optimizer.pt", "*.bin"])
print("saved to", p)
