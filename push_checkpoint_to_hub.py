from huggingface_hub import HfApi
import os

from huggingface_hub import login

api = HfApi()
hf_token = os.getenv('HF_TOKEN')
login(token=hf_token)

REPO_ID = "milanakdj/indic_parler_tts_nepal_v1_450_steps_lr_1e_6_32_eff_batch"   # ← change to your repo name
CKPT_PATH = r"C:\Users\Alex\Downloads\results (6)\checkpoints\best_checkpoint"

# Create repo if it doesn't exist
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

# Push the main checkpoint folder
api.upload_folder(
    folder_path=CKPT_PATH,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="checkpoint_step_3500 — Nepali finetuned Indic Parler - ",
    ignore_patterns=["training_state.pt"],  # skip 1.7GB optimizer state if not needed
)

print(f"✅ Pushed → https://huggingface.co/{REPO_ID}")