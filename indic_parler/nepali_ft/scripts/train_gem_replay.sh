#!/bin/bash
# FINAL parler experiment: native Gemini expressive Nepali + REPLAY, training only
# the caption->audio conditioning path.
#
# Why this shape, given everything that failed:
#  * 3 full finetunes (427M decoder trained, text encoder frozen) all made emotion
#    WORSE and the user reported "very bad quality". Backwards: that trains the
#    audio-rendering weights to change caption control.
#  * No replay in any of them. The official parler expressive recipe
#    (parler-tts-mini-expresso) mixes generic speech in "to preserve generic voice
#    capabilities". data_replay = 14 h Rasa Nepali neutral, i.e. the model's own
#    prior distribution.
#  * CSP-FT (2 probe-selected layers) is NOT usable here: the emotion probe scored
#    0.165-0.200 across all decoder layers against a 0.167 chance floor, so there is
#    no emotion-bearing layer to select. Falling back to Voicebox-Adapter's finding
#    (arXiv 2406.06251): train the conditioning path, freeze the acoustic stack.
#  * PARLER_TRAIN_ONLY keeps encoder_attn (cross-attention, 100.7M) + embed_prompts
#    trainable and freezes everything else outside the encoders. Note enc_to_dec_proj
#    does not exist on this checkpoint (encoder and decoder hidden sizes both 1024).
#  * lr 1e-5, ~5% of pretraining LR, per CSP-FT. 8e-5 spiked loss 2.31->3.80 before.
#
# Mix: data_gem 10.4 h expressive (NEW signal, native-sounding) + data_replay 14 h
# neutral (known) = 57% neutral, in the 50-70% band the Rasa paper recommends.
set -u
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
cd /workspace/milan_nepali_parler_ft
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PARLER_TRAIN_ONLY="encoder_attn,embed_prompts"
V=/root/tts/TTS_training/synthetic_pipeline/tts_service/.venv/bin/python
G=/workspace/milan_nepali_parler_ft/data_gem
R=/workspace/milan_nepali_parler_ft/data_replay

$V -m training.run_parler_tts_training \
  --model_name_or_path "ai4bharat/indic-parler-tts" \
  --feature_extractor_name "ylacombe/dac_44khz" \
  --description_tokenizer_name "google/flan-t5-large" \
  --prompt_tokenizer_name "ai4bharat/indic-parler-tts" \
  --train_dataset_name "$G+$R" \
  --train_dataset_config_name "default+default" \
  --train_split_name "train+train" \
  --train_dataset_samples "3123+8621" \
  --eval_dataset_name "$G" --eval_dataset_config_name "default" --eval_split_name "validation" \
  --target_audio_column_name "audio" --description_column_name "description" --prompt_column_name "text" \
  --max_duration_in_seconds 20 --min_duration_in_seconds 1.0 --max_text_length 400 \
  --output_dir "/workspace/milan_nepali_parler_ft/out_gemr" \
  --temporary_save_to_disk "/workspace/milan_nepali_parler_ft/tmp_codec_gemr" \
  --save_to_disk "/workspace/milan_nepali_parler_ft/prep_gemr" \
  --preprocessing_num_workers 1 --audio_encoder_per_device_batch_size 32 \
  --dtype "bfloat16" --attn_implementation "sdpa" --freeze_text_encoder true \
  --do_train true --num_train_epochs 4 \
  --gradient_accumulation_steps 4 --per_device_train_batch_size 8 \
  --gradient_checkpointing false --learning_rate 1e-5 \
  --lr_scheduler_type "cosine" --warmup_steps 400 \
  --adam_beta1 0.9 --adam_beta2 0.99 --weight_decay 0.01 --logging_steps 25 \
  --do_eval true --evaluation_strategy "steps" --eval_steps 250 --per_device_eval_batch_size 8 \
  --compute_clap_similarity_metric false --compute_noise_level_metric false \
  --save_strategy "steps" --save_steps 250 --save_total_limit 6 \
  --dataloader_num_workers 8 --group_by_length true --report_to "none" \
  --seed 20260906 --overwrite_output_dir false
