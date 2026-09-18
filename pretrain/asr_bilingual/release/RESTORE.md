# Nothing large is stored locally

Model weights, manifests, logs and a copy of the code all live in the **private**
repo `milanakdj/nemotron-asr-nepali-0.6b`. Rationale: `WHY_THIS_REPO.md` there.

```bash
export HF_TOKEN=<milanakdj write token>       # ambient HF_TOKEN is a DIFFERENT account
hf download milanakdj/nemotron-asr-nepali-0.6b --local-dir ./restored
# or just the weights:
hf download milanakdj/nemotron-asr-nepali-0.6b nemotron_ne_nepali_best.nemo --local-dir .
```

`nemotron_ne_nepali_best.nemo` md5 `f9bf472d72e821200ef4a7652d85c6ee`.
The training `.ckpt` files were deleted — the run cannot be resumed.
