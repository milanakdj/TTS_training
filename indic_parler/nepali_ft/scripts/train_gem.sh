#!/bin/bash
# Finetune on the PRE-VC Gemini Nepali audio -- the only expressive Nepali here
# that indic-parler was NOT already trained on, and the only one the user confirmed
# sounds natively Nepali (Rasa's Srijana was rejected as "a Hindi person speaking
# Nepali"; indicvoices-r Nepali is 100% West Bengal).
#
# 3,123 clips / 10.2 h / 9 voices, near gender parity (1,597 F / 1,526 M), captions
# taken from the corpus's own per-turn `voice_direction` strings.
#
# Two Rasa finetunes already failed (56.2% -> 43.8% and -> 50.0% SER) because
# indic-parler had seen that data; the point here is that this data is new.
# Checkpoint every 100 steps: on 10 h the useful window may be short, and the best
# checkpoint is likely not the last one.
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
  --train_dataset_name "/workspace/milan_nepali_parler_ft/data_gem" \
  --train_dataset_config_name "default" \
  --train_split_name "train" \
  --eval_dataset_name "/workspace/milan_nepali_parler_ft/data_gem" \
  --eval_dataset_config_name "default" \
  --eval_split_name "validation" \
  --target_audio_column_name "audio" \
  --description_column_name "description" \
  --prompt_column_name "text" \
  --max_duration_in_seconds 25 \
  --min_duration_in_seconds 0.6 \
  --max_text_length 400 \
  --output_dir "/workspace/milan_nepali_parler_ft/out_gem" \
  --temporary_save_to_disk "/workspace/milan_nepali_parler_ft/tmp_codec_gem" \
  --save_to_disk "/workspace/milan_nepali_parler_ft/prep_gem" \
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
  --learning_rate 3e-5 \
  --lr_scheduler_type "cosine" \
  --warmup_steps 500 \
  --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.01 \
  --logging_steps 25 \
  --do_eval true \
  --evaluation_strategy "steps" --eval_steps 100 \
  --per_device_eval_batch_size 8 \
  --compute_clap_similarity_metric false \
  --compute_noise_level_metric false \
  --save_strategy "steps" --save_steps 100 --save_total_limit 8 \
  --dataloader_num_workers 8 \
  --group_by_length true \
  --report_to "none" \
  --seed 20260905 \
  --overwrite_output_dir false
