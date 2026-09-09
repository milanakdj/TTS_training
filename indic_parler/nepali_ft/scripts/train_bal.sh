#!/bin/bash
# Retry #2. Run #1 made emotion WORSE (SER mean 56.2% -> 43.8%, angry 87.5% -> 50%)
# for two reasons, both fixed here:
#   1. the mix was 82% neutral (25,529 neutral vs 5,711 emotion clips), so the model
#      learned the dominant mode -- neutral went to 100% and fear/sad collapsed onto
#      it. data_bal caps neutral at 13 h against 12.3 h of emotion, which is also
#      what the Rasa paper's own recipe looks like (~10 h neutral, 1-3 h per emotion).
#   2. lr 8e-5 with 250 warmup steps spiked train loss 2.31 -> 3.80 at step 900 and
#      only recovered as cosine decayed it; eval loss finished at 3.88 vs train 2.38.
#      Now 2e-5 with 500 warmup.
# Baseline to beat, measured on the BASE model with Srijana captions: mean 56.2%,
# angry 87.5%, happy 62.5%, sad 0%.
#
# --description_tokenizer_name IS NOT OPTIONAL. The trainer defaults it to
# model_name_or_path, which tokenizes captions with the repo's Llama tokenizer while
# the text encoder is flan-t5-large -- the exact silent bug that made emotion look
# broken in the first place. Training that way would fit the model to gibberish captions.
#
# --freeze_text_encoder keeps flan-t5 intact so the caption space the base model
# already understands is not overwritten by 27k Nepali captions.
set -euo pipefail
cd /workspace/milan_nepali_parler_ft
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
# RANK/LOCAL_RANK leak in from the shell env; transformers then tries a distributed
# rendezvous and dies on the missing WORLD_SIZE. This is single-GPU.
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
V=/root/tts/TTS_training/synthetic_pipeline/tts_service/.venv/bin/python

$V -m training.run_parler_tts_training \
  --model_name_or_path "ai4bharat/indic-parler-tts" \
  --feature_extractor_name "ylacombe/dac_44khz" \
  --description_tokenizer_name "google/flan-t5-large" \
  --prompt_tokenizer_name "ai4bharat/indic-parler-tts" \
  --train_dataset_name "/workspace/milan_nepali_parler_ft/data_bal" \
  --train_dataset_config_name "default" \
  --train_split_name "train" \
  --eval_dataset_name "/workspace/milan_nepali_parler_ft/data_bal" \
  --eval_dataset_config_name "default" \
  --eval_split_name "validation" \
  --target_audio_column_name "audio" \
  --description_column_name "description" \
  --prompt_column_name "text" \
  --max_duration_in_seconds 25 \
  --min_duration_in_seconds 0.6 \
  --max_text_length 400 \
  --output_dir "/workspace/milan_nepali_parler_ft/out_bal" \
  --temporary_save_to_disk "/workspace/milan_nepali_parler_ft/tmp_codec_bal" \
  --save_to_disk "/workspace/milan_nepali_parler_ft/prep_bal" \
  --preprocessing_num_workers 1 \
  --audio_encoder_per_device_batch_size 32 \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --freeze_text_encoder true \
  --do_train true \
  --num_train_epochs 6 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing false \
  --learning_rate 2e-5 \
  --lr_scheduler_type "cosine" \
  --warmup_steps 500 \
  --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.01 \
  --logging_steps 25 \
  --do_eval true \
  --evaluation_strategy "steps" --eval_steps 500 \
  --per_device_eval_batch_size 8 \
  --compute_clap_similarity_metric false \
  --compute_noise_level_metric false \
  --save_strategy "steps" --save_steps 500 --save_total_limit 4 \
  --dataloader_num_workers 8 \
  --group_by_length true \
  --report_to "none" \
  --seed 20260905 \
  --overwrite_output_dir false
