import io, re, sys, numpy as np, soundfile as sf, torch, torchaudio
import pyarrow.parquet as pq, uroman as ur

dev = "cuda:0"
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).to(dev).eval()
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()
U = ur.Uroman()
print("MMS_FA sr:", bundle.sample_rate)

f = pq.ParquetFile("/root/tts/TTS_training/c9_data/raw/shard_0000.parquet")
row = next(f.iter_batches(batch_size=1)).to_pylist()[0]
y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
print("audio:", len(y)/sr, "s  sr", sr)

# Devanagari words, punctuation stripped, aligned 1:1 with their romanizations
raw_words = [w for w in re.split(r"\s+", row["text"].replace("\n", " ")) if w]
words, roms = [], []
for w in raw_words:
    core = re.sub(r"[।॥,.;:!?\"'()\[\]–—-]", "", w)
    if not core: continue
    r = re.sub(r"[^a-z']", "", U.romanize_string(core).lower())
    if not r: continue
    words.append(w); roms.append(r)
print("words:", len(words), " first:", words[:6], roms[:6])

with torch.inference_mode():
    wav = torch.from_numpy(y).unsqueeze(0).to(dev)
    emission, _ = model(wav)
spans = aligner(emission[0], tokenizer(roms))
ratio = wav.shape[1] / emission.shape[1] / sr
t = [(s[0].start*ratio, s[-1].end*ratio) for s in spans]
print("aligned spans:", len(t), " audio covered:", round(t[-1][1],2), "of", round(len(y)/sr,2))
print("\nfirst 8 word timings:")
for w,(a,b) in list(zip(words,t))[:8]: print(f"  {a:6.2f}-{b:6.2f}  {w}")
print("\nlargest inter-word gaps (pause candidates):")
gaps = sorted(((t[i+1][0]-t[i][1], i) for i in range(len(t)-1)), reverse=True)[:8]
for g,i in gaps: print(f"  gap {g:5.2f}s after word {i}: ...{words[max(0,i-2)]} {words[i]} | {words[i+1]}...")
