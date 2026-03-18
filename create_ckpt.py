from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="lal-Joey/MedSAM3_v1",
    filename="best_lora_weights.pt",
    local_dir="weights/MedSAM3/checkpoints"
)