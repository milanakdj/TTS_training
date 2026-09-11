#!/usr/bin/env python3
"""Finetune bodhan-ai/indic-transcribe-flex on HUMAN-transcribed Nepali.

WHY THIS TRAINS ON GOLD DATA ONLY
---------------------------------
flex is a nvidia/canary-1b-v2 derivative. Our large Nepali corpora are pseudo-
labelled (`transcript_used` saaras/canary), and measured 2026-09-11 flex already
scores 0.015-0.022 WER on them -- because it is being compared to its own family's
output. On the mahadhwani sample its hypotheses match `canary_transcript` at WER
0.019 but `saaras_transcript` at 0.060, and 64% of those labels are Canary's.
Training there teaches flex to imitate an ASR it already is.

Against HUMAN transcripts flex is genuinely weak: rasa 0.129, indicvoices-r 0.177.
That is the headroom, so gold data is the training set and the gold test split is
the only number worth quoting.

WHAT HAD TO BE BUILT
--------------------
The shipped modelling code is an inference port: `forward()` raises
NotImplementedError when `labels` is passed, and there is no text encoder on the
tokenizer (decode + prompt building only). So this file adds:
  * a training forward with teacher forcing + cross-entropy
  * text -> ids  (multilingual SP ids are offset by spl_size = 1152)
  * the 10-token canary2 prompt as a loss-masked prefix
The decoder already builds a causal mask on the seq>1 path, so teacher forcing is
correct as-is; `use_cache` is off so no KV cache is allocated.

  python3 flex_finetune.py --smoke        # 20 steps, no save
  python3 flex_finetune.py
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

BASE = os.environ.get("FLEX_BASE", "/workspace/models/indic-transcribe-flex")
OUT = os.environ.get("FLEX_OUT", "/workspace/flex-nepali")
DATA = "/root/tts/TTS_training/whisper/flex"
sys.path.insert(0, BASE)

from transformers import AutoModelForSpeechSeq2Seq, Seq2SeqTrainingArguments, Trainer
from transformers.modeling_outputs import Seq2SeqLMOutput
from feature_extraction_indic_canary import IndicCanaryFeatureExtractor
from tokenization_indic_canary import IndicCanaryTokenizer

FE = None          # set in main(); global so it is not saved into the checkpoint


def training_forward(self, waveforms=None, wav_lens=None, decoder_input_ids=None,
                     labels=None, **kw):
    """Teacher-forced forward. Mel is computed here, on the GPU, because the box's
    spare CPU is already committed (8 of 16 cores are held by wekanode)."""
    feats, feat_lens = FE(waveforms, wav_lens)
    mask = (torch.arange(feats.size(2), device=feats.device)[None, :]
            < feat_lens[:, None]).long()
    enc = self.model.encoder(feats, attention_mask=mask)
    cross = self._cross_mask_from_lengths(enc.lengths, enc.last_hidden_state.size(1))
    # start_pos=0 and seq>1 -> the decoder builds its own causal mask. No cache.
    hidden = self.model.decoder(decoder_input_ids, enc.last_hidden_state, cross,
                                past_key_values=None, start_pos=0)
    logits = self.lm_head(hidden)
    loss = None
    if labels is not None:
        # float() so the softmax is computed in fp32 under bf16 autocast
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               labels.reshape(-1), ignore_index=-100)
    return Seq2SeqLMOutput(loss=loss, logits=logits)


class GoldSet(Dataset):
    def __init__(self, path, tok, limit=None):
        self.rows = [json.loads(l) for l in open(path, errors="replace")]
        if limit: self.rows = self.rows[:limit]
        self.tok = tok
        self.prompt = tok.encode_prompt("ne", itn=False, romanized=False)
        self.offset = tok.spl_size          # text pieces live at [1152, 7152)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        import soundfile as sf
        r = self.rows[i]
        wav, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))
        if sr != FE.sample_rate:
            wav = FE.resample(wav, sr)
        ids = [self.offset + t for t in self.tok.multi.encode(r["text"])]
        return {"wav": wav, "ids": ids}


class Collator:
    def __init__(self, tok, prompt, max_target=448):
        self.pad, self.eos = tok.pad_id, tok.eos_id
        self.prompt, self.max_target = prompt, max_target

    def __call__(self, batch):
        batch = [b for b in batch if b["wav"].numel() > 0]
        wl = [b["wav"].shape[0] for b in batch]
        W = max(wl)
        waveforms = torch.zeros(len(batch), W)
        for i, b in enumerate(batch): waveforms[i, :wl[i]] = b["wav"]

        # full = prompt + text + eos ; input is full[:-1], labels are full[1:].
        # The prompt is context, not a prediction target, so the first
        # len(prompt)-1 label positions are masked out.
        seqs = [self.prompt + b["ids"][: self.max_target] + [self.eos] for b in batch]
        L = max(len(s) for s in seqs) - 1
        dec = torch.full((len(batch), L), self.pad, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            dec[i, : len(s) - 1] = torch.tensor(s[:-1])
            tgt = torch.tensor(s[1:])
            lab[i, : len(s) - 1] = tgt
            lab[i, : len(self.prompt) - 1] = -100
        return {"waveforms": waveforms, "wav_lens": torch.tensor(wl),
                "decoder_input_ids": dec, "labels": lab}


def main(a):
    global FE
    dev = "cuda"
    tok = IndicCanaryTokenizer.from_pretrained(BASE)
    FE = IndicCanaryFeatureExtractor.from_pretrained(BASE, device=dev)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)
    type(model).forward = training_forward        # the port refuses labels otherwise
    model.config.use_cache = False
    model = model.to(dev)
    model.train()
    n = sum(p.numel() for p in model.parameters())
    print(f"[flex-ft] {n/1e9:.2f}B params  base={BASE}  out={OUT}", flush=True)

    train = GoldSet(f"{DATA}/gold_train.jsonl", tok, limit=a.limit)
    val = GoldSet(f"{DATA}/gold_val.jsonl", tok, limit=64 if a.smoke else None)
    coll = Collator(tok, train.prompt)
    print(f"[data] train {len(train):,}  val {len(val):,}  (gold/human transcripts only)",
          flush=True)

    args = Seq2SeqTrainingArguments(
        output_dir=OUT,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        per_device_eval_batch_size=a.batch,
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        max_steps=20 if a.smoke else -1,
        # transformers 5.x dropped warmup_ratio; warmup_steps is the survivor
        warmup_steps=a.warmup_steps,
        lr_scheduler_type="linear",
        bf16=True,
        optim="adamw_torch",
        dataloader_num_workers=a.workers,
        logging_steps=25,
        eval_strategy="no" if a.smoke else "steps",
        eval_steps=a.eval_steps,
        save_strategy="no" if a.smoke else "steps",
        save_steps=a.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=not a.smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        # the collator emits waveforms/wav_lens, which Trainer's column pruning
        # would otherwise drop as "unused"
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(model=model, args=args, train_dataset=train,
                      eval_dataset=val, data_collator=coll)
    t0 = time.time()
    trainer.train()
    print(f"[flex-ft] training done in {(time.time()-t0)/3600:.2f} h", flush=True)
    if not a.smoke:
        trainer.save_model(OUT)
        # the custom code + tokenizer + mel buffers must travel with the weights
        import shutil
        for f in ("configuration_indic_canary.py", "modeling_indic_canary.py",
                  "tokenization_indic_canary.py", "feature_extraction_indic_canary.py",
                  "indic_transcribe.py", "tokenizer_multilingual.model",
                  "tokenizer_spl_tokens.model", "tokenizer_config.json",
                  "feature_extractor.safetensors", "generation_config.json"):
            src = os.path.join(BASE, f)
            if os.path.isfile(src): shutil.copy2(src, os.path.join(OUT, f))
        print(f"[flex-ft] saved to {OUT}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--warmup-steps", type=int, default=120)
    p.add_argument("--limit", type=int)
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
