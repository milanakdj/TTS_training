# Architecture Decision Record — Nepali Pocket-TTS

Every decision that shaped this project, why it was taken, what it cost, and what
would have to change for it to be revisited. Read top to bottom the first time;
after that use it as a lookup so a decision is not silently re-litigated.

Status legend: **Accepted** (in force) · **Superseded** · **Open** (known risk, no
decision yet).

| # | Decision | Status |
|---|---|---|
| [001](#adr-001) | Use Kyutai Pocket-TTS as the Nepali TTS stack | Accepted |
| [002](#adr-002) | Two stages: finetune a 24L teacher, then distil a 6L student | Accepted |
| [003](#adr-003) | Fresh Nepali BPE-4000 tokenizer and a reset text embedding | Accepted |
| [004](#adr-004) | Train the student from scratch, not from pretrained weights | Accepted |
| [005](#adr-005) | Bake classifier-free guidance into the student (`distill_cfg_coef: 2.0`) | Accepted |
| [006](#adr-006) | Freeze and copy the flow head and Mimi; distil the backbone only | Accepted |
| [007](#adr-007) | Gate training data on pre-computed quality metadata, not hours | Accepted |
| [008](#adr-008) | Evaluate against a real-human ceiling with a Nepali-finetuned ASR | Accepted |
| [009](#adr-009) | Pin CPU thread count for every latency measurement | Accepted |
| [010](#adr-010) | Ship the student; retire the teacher to a distillation source | Accepted |
| [011](#adr-011) | Keep checkpoints on `/workspace`, not `/` | Accepted |
| [012](#adr-012) | Publish public-but-gated on Hugging Face | Accepted |
| [013](#adr-013) | Train on AI4Bharat sources despite the accent finding | **Open** |
| [014](#adr-014) | No emotion conditioning in this model | Accepted |

---

## ADR-001 {#adr-001}
### Use Kyutai Pocket-TTS as the Nepali TTS stack

**Status:** Accepted (2026-09-05)

**Context.** Three earlier approaches to Nepali expressive TTS in this project ran
out of road:

- **RVC** (voice conversion) was dropped in favour of Seed-VC; neither carried
  emotion together with timbre.
- **Indic Parler-TTS** was pursued hardest. It ended in a documented negative
  result: a linear probe on all 24 decoder layers scored 0.158–0.200 against a
  0.167 chance floor, so emotion is not linearly decodable from that model's pooled
  states for Nepali. Caption engineering plateaued, activation steering collapsed
  every emotion onto "surprise", and three separate finetunes all scored *below*
  the untouched base model.
- The metric driving that work turned out to be invalid: `emotion2vec_plus_large`
  scores *real, professionally acted* human Nepali at 30.4% mean top-1 (sad 3.1%).

**Decision.** Move to Kyutai Pocket-TTS: a flow-matching language model over Mimi
codec latents, with zero-shot voice cloning from a short audio prompt and a design
target of CPU inference.

**Why this and not more Parler work.** Pocket-TTS changes the axis. It has no
emotion conditioning at all, so it does not depend on the broken emotion metric; it
is judged on intelligibility, speaker similarity and latency, all of which can be
measured with instruments that were validated against real human audio. It is also
two orders of magnitude cheaper to run.

**Consequences.** Emotional control is out of scope for this model (see
[ADR-014](#adr-014)). Prosody comes entirely from the voice prompt. In exchange we
got a model that is deployable on a CPU and measurably good at the thing it does.

**Revisit if** a valid Nepali emotion metric appears — one demonstrated to separate
real human emotional Nepali from real human neutral Nepali. Until then, any emotion
claim about any Nepali TTS in this project is unfalsifiable.

---

## ADR-002 {#adr-002}
### Two stages: finetune a 24-layer teacher, then distil a 6-layer student

**Status:** Accepted (2026-09-05)

**Context.** The released Pocket-TTS English checkpoints come in a 6-layer
"pocket" size and a 24-layer size. Finetuning the 6-layer model directly on Nepali
was the obvious one-stage option.

**Decision.** Finetune the 24-layer model on Nepali first (stage 1, the *teacher*),
then depth-distil it into a 6-layer *student* (stage 2). Both stages ran 200,000
steps at `batch_size` 64 on one H100.

**Why.** A 24-layer backbone has far more capacity to absorb 2,215 hours of a new
language with a freshly initialised text embedding. But measured on CPU it
generates at **1.53x real-time** — and at the time of the decision it measured
0.73x, i.e. slower than real time and undeployable. Distillation lets the capacity
be spent during training and then thrown away at inference: the student learns to
reproduce the teacher's backbone activations with a quarter of the layers.

**Consequences.** Doubled training cost (two 200k-step runs, roughly 36 hours of
H100 each). The payoff was larger than expected — see [ADR-005](#adr-005): the
student did not merely approach the teacher, it beat it.

---

## ADR-003 {#adr-003}
### Fresh Nepali BPE-4000 tokenizer and a reset text embedding

**Status:** Accepted (2026-09-05)

**Context.** The English checkpoint's SentencePiece vocabulary contains no
Devanagari. Every Nepali character would encode to an unknown token, so the model
could not represent the target text at all.

**Decision.** Train a 4,000-token Nepali SentencePiece BPE on this corpus
(`tokenizer/nepali_bpe4000.model`) and set `reset_text_embedding: true` for the
teacher so the text embedding is reinitialised rather than inherited.

**Why 4,000.** Nepali is written with a small alphabet and rich agglutination;
4,000 subwords covers the corpus without giving the model a huge embedding table to
learn from scratch. The same reasoning appears in this project's Parakeet ASR work,
where extending a vocabulary with Devanagari was a precondition rather than a knob.

**Consequences.** The model has **no English ability whatsoever** — the embedding
that knew English was discarded. Mixed-script input is unsupported. The tokenizer
file is part of the released artefact and must travel with the weights; a mismatch
produces fluent nonsense rather than an error.

---

## ADR-004 {#adr-004}
### Train the student from scratch, not from pretrained weights

**Status:** Accepted (2026-09-07)

**Context.** `start_from_pretrained` could have initialised the student from the
released English 6-layer checkpoint.

**Decision.** `start_from_pretrained: false`. The student's backbone starts random.

**Why.** The student's only objective is to reproduce *this teacher's* backbone
activations. English pretrained weights encode a different text embedding, a
different tokenizer and a different language; as an initialisation for an
activation-matching objective they are closer to noise than to a head start. The
non-backbone weights are not random, though — they are copied from the teacher (see
[ADR-006](#adr-006)), which is where the useful initialisation actually comes from.

**Consequences.** Nothing English-derived survives in the student's backbone. The
learning-rate screen (`{1,2,4}e-4` on full mini-cosines) picked 4e-4, higher than
the teacher's 2e-4, which is consistent with training a fresh backbone rather than
nudging a trained one.

---

## ADR-005 {#adr-005}
### Bake classifier-free guidance into the student (`distill_cfg_coef: 2.0`)

**Status:** Accepted (2026-09-07) — **this is why the student beat its teacher**

**Context.** Classifier-free guidance (CFG) improves conditional generation by
extrapolating away from an unconditioned prediction. It costs a second forward pass
per step. Crucially, **CFG is not implemented anywhere in the Pocket-TTS inference
package** — `grep -rn cfg pocket_tts/` returns nothing outside the word "config".
It exists only in the training code.

**Decision.** Compute the distillation targets *with* guidance at coefficient 2.0,
so the student learns the teacher's guided output distribution rather than its raw
one.

**Why.** The student then produces guided-quality output from a single forward
pass. The teacher cannot do this at all: the inference path has no null branch, so a
deployed teacher is always unguided.

**Measured consequence.** On 100 held-out utterances scored with a
Nepali-finetuned Whisper:

| system | WER | CER |
|---|---|---|
| real human (ceiling) | 0.343 | 0.122 |
| teacher 24L | 0.402 | 0.177 |
| **student 6L** | **0.322** | **0.139** |

The student is ahead of the teacher on CER in all six corpus sources individually,
so this is not one source carrying an average.

**Important corollary.** Because the teacher can only ever be sampled unguided,
"student beats teacher" is a statement about *deployable* configurations, not a
claim that 6 layers hold more knowledge than 24. A CFG-capable teacher would need
two forward passes per frame, which would put it near 0.77x real-time — slower than
real time and outside the point of this project.

**Related config note.** `text_dropout` and `voice_dropout` are both 0.0 here on
purpose. The teacher's targets are computed fully-conditioned on both branches, so
dropping conditioning on the student's input would ask it to predict a conditioned
target from an unconditioned input.

---

## ADR-006 {#adr-006}
### Freeze and copy the flow head and Mimi; distil the backbone only

**Status:** Accepted (2026-09-07)

**Context.** A Pocket-TTS model is three parts: a transformer *backbone* over text
and latent history, a small *flow head* that turns a backbone state into a Mimi
latent, and the *Mimi codec* that turns latents into a waveform.

**Decision.** Keep the student's `d_model` identical to the teacher's (1024) so
every non-backbone weight can be copied verbatim. The flow head stays frozen; only
`transformer.num_layers` changes, 24 → 6. Mimi is copied unchanged.

**Why.** It makes the distillation objective a clean, well-posed regression: the
student must produce backbone activations the *existing, unmodified* flow head can
already consume. Nothing downstream is moving while the backbone learns.

**Measured consequence — parameter counts, straight from the safetensors exports:**

| model | total | flow_lm (backbone + flow head) | Mimi codec |
|---|---|---|---|
| **student 6L** | **109.5M** | 89.4M | 20.1M |
| teacher 24L | 336.1M | 316.0M | 20.1M |

Three things worth drawing out of those numbers:

- **The 20.1M Mimi codec is identical in both**, because it is copied from the
  teacher and frozen. That is also why the speedup has no hidden Amdahl ceiling:
  Mimi is only ~19% of the teacher's per-frame cost (7.5 ms/frame against 39.9 ms
  for the autoregressive stage), so cutting the backbone pays almost in full.
- **The distil cut the backbone, not the head.** Both models share `flow.depth: 6`,
  `flow.dim: 512` and `d_model: 1024`; only `num_layers` differs. So the
  316M → 89.4M drop is almost entirely the 18 removed transformer layers.
- **On disk:** 1.34 GB for the teacher against **438 MB** for the student, float32.

This is a *million*-scale model, nowhere near a billion. That matters because the
whole point of "pocket" TTS is CPU deployment, and 109.5M is what buys 5.04x
real-time on 4 pinned threads. The Indic Parler-TTS work in this project was a
different scale of model entirely, which is part of why the two were never a fair
speed comparison.

---

## ADR-007 {#adr-007}
### Gate training data on pre-computed quality metadata, not hours

**Status:** Accepted (2026-09-05)

**Context.** A scan of all 80 directories under `/workspace/proc_data_new` found
~8,500 hours of Nepali. Most of it is the *same audio in different processing
states* — `indic_voices_long` → `IndicVoices` → `indicvoices-r` are one corpus —
so a naive hour count triple-counts.

**Decision.** Pick one representative per corpus and gate on the quality metadata
the team had already computed, rather than on duration:

- `ans_snr*`: three independent ASRs (saaras / canary / conformer) transcribed each
  clip; keep only clips where they agree. `spk_overlap_percent` removes clips with
  more than one talker, which a TTS must never learn.
- `mahadhwani` / `podcast-index`: same three-ASR setup exposed as `low_cer_count`;
  `podcast-index` sits at p50 = 1, so it needs the `>= 2` gate to be usable at all.
- `indicvoices-r`: already restored; `dnsmos_mean` / SNR / C50 gate residual noise.
- `rasa`: studio quality, ungated.

Result: **2,215.7 hours over 749,610 clips**, 100% word-aligned, audio referenced
in place with nothing copied.

**Consequences.** The kept set is dominated by spontaneous in-the-wild speech
(1,830 h of the 2,216 is `ans_*` YouTube), which is why the model clones
conversational voices better than polished narration. Alignment used
`gagan3012/wav2vec2-xlsr-nepali` and was resumable — the v2 run seeded its output
with v1's rows so only the ~1,990 new hours were aligned.

---

## ADR-008 {#adr-008}
### Evaluate against a real-human ceiling with a Nepali-finetuned ASR

**Status:** Accepted (2026-09-08)

**Context.** This project has been burned twice by trusting a metric that could not
see the thing it was measuring. `emotion2vec` scored real human acted Nepali at
30.4%. Then off-the-shelf `whisper-large-v3-turbo`, pinned to `language="ne"`,
scored *real human* held-out Nepali at **0.917 WER / 0.344 CER** — useless for
ranking anything.

**Decision.** Three rules for every evaluation here:

1. **Always score real human audio of the same utterances through the identical
   pipeline** and report it as a ceiling row. A metric that cannot handle real
   speech cannot rank synthetic speech.
2. **Use a Nepali-finetuned ASR.** `himalaya-ai/whisper-large-v3-nepali-final`
   scores that same human audio at **0.343 WER / 0.122 CER**. It requires a
   **doubled** `<|startoftranscript|>` decoder prefix; with a single one it emits
   endless nonsense (WER > 400%), and the prefix cannot be expressed through
   `pipeline()`, faster-whisper or a plain `generate()`.
3. **Detect and discard broken measurements, not clips.** Whisper silently
   romanizes a few percent of Nepali audio, producing CER ≈ 1.0 on fine audio.
   Filter on a Devanagari-character ratio; that is why `n` varies per row.

**Protocol.** 100 held-out utterances, 77 speaker identities, all 6 sources. The
target utterance is strictly held out; the voice prompt is a *different* clip of the
same identity taken from the train split, because a prompt is conditioning, not a
target. Speaker identity is keyed on **(directory, diarization label)** — a bare
`SPEAKER_01` is per-video, so keying on the label alone silently pairs unrelated
people.

**Two results that must be reported with their caveats:**

- The student's WER (0.322) came out *below* the real human row (0.343). This is
  **not** superhuman quality. The human audio is largely spontaneous speech with
  disfluencies scored against a reference transcript, and synthetic read speech is
  simply easier for an ASR. The honest reading is that **WER has saturated as a
  discriminator** at this quality level.
- Resemblyzer speaker similarity rates both TTS systems *above* genuine human
  cross-utterance similarity (0.826 / 0.841 against 0.807). A clone sits
  unnaturally close to its own prompt while two real recordings of one person differ
  in content, session and channel. Use that column for teacher-vs-student only.

**Consequence.** The remaining instrument with no known failure mode is the human
ear. `infer/final/listen/` holds 12 blind A/B triples for exactly that.

---

## ADR-009 {#adr-009}
### Pin CPU thread count for every latency measurement

**Status:** Accepted (2026-09-08)

**Context.** The first CPU benchmark reported the teacher at 0.15x real-time and
the student at 0.17x — a 1.13x speedup where cutting 24 layers to 6 should give
about 3x. The conclusion "the distil bought nothing" was wrong.

**Cause.** The benchmark called `torch.set_num_threads(os.cpu_count())` = 16 on a
16-core machine that the training job already held at load ~13.6 (six dataloader
workers plus sixteen inductor compile workers). Both models thrashed the same
oversubscribed threads, so wall time measured contention, not compute, and
compressed a 3x gap into 1.13x.

**Decision.** Every latency measurement pins its thread count
(`torch.set_num_threads(4)` plus `OMP_NUM_THREADS`) and records `/proc/loadavg` in
its log. Where possible, measure when the GPU job is finished.

**Measured consequence.** Same machine, same load, threads pinned to 4:

| | first run (16 threads) | pinned (4 threads) |
|---|---|---|
| teacher 24L | 0.15x RT | **1.53x RT** |
| student 6L | 0.17x RT | **5.04x RT** |

Confirmed by a stage profiler that times the two pipeline threads separately (the
autoregressive loop and the Mimi decoder run concurrently, so wall time is roughly
the slower of the two): AR 39.9 → 12.7 ms/frame, a **3.14x** backbone speedup,
against an 80 ms/frame real-time budget.

**Generalisation.** When a benchmark says an architectural change bought nothing,
suspect the harness before the change.

---

## ADR-010 {#adr-010}
### Ship the student; retire the teacher to a distillation source

**Status:** Accepted (2026-09-08)

**Decision.** The 6-layer student is the deployable model. The teacher is kept only
to distil from.

**Why.** The student is better on every measured axis — WER, CER, speaker
similarity — and 3.3x faster, 3x smaller. There is no configuration in which
deploying the teacher is preferable.

**Artefacts.** Student weights at
`/workspace/nepali_student_6l/model_final_200k.safetensors` (438 MB, EMA export at
step 200,000); inference config at `infer/nepali_student_6l.yaml`. Teacher at
`runs/nepali_teacher_24l/model.safetensors` (1.34 GB).

---

## ADR-011 {#adr-011}
### Keep checkpoints on `/workspace`, not `/`

**Status:** Accepted (2026-09-07)

**Context.** `ckpt_freq: 2500` over 200,000 steps with `num_ckpt_keep: 999` keeps
about 80 checkpoints at 676 MB each — roughly 54 GB per run. The root filesystem
was 88% full with 205 GB free.

**Decision.** `run_dir: /workspace/nepali_student_6l`. `/workspace` is a 2.3 PB
shared filesystem with 850 TB free.

**Consequence.** Training artefacts and the repo live on different filesystems, so
paths in configs are absolute. Optimiser state is a sidecar file kept only for the
newest checkpoint, so older snapshots stay loadable for eval and distillation while
demotion is a plain unlink.

---

## ADR-012 {#adr-012}
### Publish public-but-gated on Hugging Face

**Status:** Accepted (2026-09-08)

**Decision.** Publish the student as a public repository with **gated** access,
under CC-BY-4.0.

**Why gated.** The model clones a voice from three to five seconds of audio. Gating
puts a name, an affiliation, a stated purpose and an explicit consent acknowledgement
in front of that capability. This mirrors the decision already taken for
`milanakdj/nepali-tts-synthetic-v2`.

**Why CC-BY-4.0.** The upstream weights (`kyutai/pocket-tts`) are CC-BY-4.0 and the
Mimi codec weights in the release are Kyutai's, unmodified. Matching the upstream
licence is the clean, compliant choice; attribution to Kyutai is in the card.

**Operational note.** `extra_gated_*` metadata in the card YAML does **not** by
itself turn gating on, and neither does a `gated:` key. Gating must be set through
the settings API (`HfApi.update_repo_settings(..., gated="manual")`). Reading it
back needs `?expand[]=gated`, or the API reports `gated: false` regardless.

**No reference audio ships with the release.** The corpus contains identifiable
real people who did not consent to having their voices redistributed as cloning
prompts, and publishing cloned samples of them would compound that.

---

## ADR-013 {#adr-013}
### Train on AI4Bharat sources despite the accent finding — **OPEN**

**Status:** **Open** — a known risk with no decision recorded

**Context.** `scripts/filter_native.py` exists and documents a real finding: the
AI4Bharat-collected Nepali corpora are not Nepal-native speech. IndicVoices-R
Nepali is 100% West Bengal (Kalimpong, Darjeeling, Jalpaiguri, Alipurduar — all 908
speakers), and Rasa was collected by the same group; its "Srijana" voice was
rejected by ear as "a Hindi person speaking Nepali". Training on that teaches the
accent. The script keeps only
`ans_snr50, ans_snr40-50, mahadhwani, orpheus_snr50`.

**What actually happened.** The filter was never applied. Both training runs used
`manifests/train_v2_aligned.jsonl`, which still contains 168.1 h of
`indicvoices-r`, 74.0 h of `indicvoices-r-long` and 54.5 h of `rasa` — about 13% of
the corpus. No native-filtered manifest was ever written. The script is dead code
in the pipeline as it stands.

**Why this is Open rather than Accepted.** It is not clear the trade-off was ever
weighed. The 296 hours in question are the cleanest studio audio in the corpus and
the only source of emotional registers, so dropping them is not free. But half of
the 100-utterance evaluation set (18 `indicvoices-r`, 16 `indicvoices-r-long`, 16
`rasa`) comes from sources judged non-native by ear, which means the headline
numbers partly measure how well the model reproduces an accent the project decided
it did not want.

**What would settle it.** A blind listening comparison between the current model
and one trained on the native-filtered manifest, judged by a Nepal-native listener.
Until then, treat "trained on Nepal-native speech" as **false** for this checkpoint
and say so in any card or paper.

---

## ADR-014 {#adr-014}
### No emotion conditioning in this model

**Status:** Accepted (2026-09-08)

**Decision.** Ship no emotion or style conditioning. Prosody comes entirely from
the voice prompt.

**Why.** Pocket-TTS has no emotion conditioning input, and this project has no
valid Nepali emotion metric with which to build or verify one — see
[ADR-001](#adr-001) and [ADR-008](#adr-008). Adding an unverifiable feature would
manufacture exactly the situation that consumed the Parler-TTS effort.

**The one adjacent data point, with its caveat.** On a matched four-emotion SER
subset, Pocket-TTS via voice prompt scored 43.8% against Indic Parler base's 34.4%.
Both sit inside the noise of a metric that rates *real human speech* at 36.7%. That
is **not** a real difference and must not be cited as one.

**Revisit if** a validated Nepali emotion metric exists. The prerequisite is
demonstrating that it separates real human emotional Nepali from real human neutral
Nepali. Nothing else is worth building on.
