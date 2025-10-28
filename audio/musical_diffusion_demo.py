"""
Quick demo: did British bands copy US rock style in the '60s?

Using CLAP audio embeddings to measure musical similarity, then running an event study.
The idea: US innovators (e.g., Chuck Berry) define a "shock" style around 1956. Then we see
if UK bands started sounding more like them after the British Invasion kicked off in 1963.

TODO: ADD MP3 files-- maybe we'll need way more, tbd
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ----------- 1) Dataset ----------------------
# Toy dataset: US innovators vs UK followers
# PLACEHOLDERS FOR NOW-- will add songs later
# Keep dataset small so this runs fast in class
TRACKS = [
    # US innovators - these define the "shock" style
    {"path": "data/Chuck_Berry_RollOverBeethoven_1956.mp3", "country": "US", "year": 1956, "artist": "Chuck Berry", "title": "Roll Over Beethoven", "innovator": 1},
    {"path": "data/Little_Richard_LongTallSally_1956.mp3",  "country": "US", "year": 1956, "artist": "Little Richard", "title": "Long Tall Sally", "innovator": 1},

    # Some US songs from around the same time as controls
    {"path": "data/US_RnB_SongA_1962.mp3", "country": "US", "year": 1962, "artist": "US Artist A", "title": "US R&B A", "innovator": 0},
    {"path": "data/US_Rock_SongB_1964.mp3", "country": "US", "year": 1964, "artist": "US Artist B", "title": "US Rock B", "innovator": 0},

    # UK bands - these are the "treated" group (did they start copying US style?)
    {"path": "data/Beatles_RollOverBeethoven_1963.mp3", "country": "UK", "year": 1963, "artist": "The Beatles", "title": "Roll Over Beethoven", "innovator": 0},
    {"path": "data/Beatles_ItWontBeLong_1963.mp3",       "country": "UK", "year": 1963, "artist": "The Beatles", "title": "It Won't Be Long", "innovator": 0},
    {"path": "data/Stones_ComeOn_1963.mp3",               "country": "UK", "year": 1963, "artist": "The Rolling Stones", "title": "Come On", "innovator": 0},
    {"path": "data/Kinks_YouReallyGotMe_1964.mp3",        "country": "UK", "year": 1964, "artist": "The Kinks", "title": "You Really Got Me", "innovator": 0},

    # A few more controls
    {"path": "data/US_Pop_SongC_1960.mp3", "country": "US", "year": 1960, "artist": "US Artist C", "title": "US Pop C", "innovator": 0},
    {"path": "data/UK_Pop_SongD_1961.mp3", "country": "UK", "year": 1961, "artist": "UK Artist D", "title": "UK Pop D", "innovator": 0},
]

# The "event" is 1963 - when Beatlemania hit the US and the British Invasion really took off
EVENT_YEAR = 1963

# ----------- 2) Audio embeddings with CLAP ----------
def load_audio_mono(path, sr=48000, max_seconds=30):
    """Load audio and prep it for CLAP. Keep clips short to speed things up."""
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")
    if max_seconds is not None:
        target_len = int(sr * max_seconds)
        if len(y) > target_len:
            y = y[:target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
    return y, sr

def get_embedding(path):
    """Get audio embedding using CLAP. Downloads model weights on first run."""
    import torch
    from laion_clap import CLAP_Module
    
    y, sr = load_audio_mono(path, sr=48000, max_seconds=30)
    
    # For Macbook M serie GPU use (Olaf only)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = CLAP_Module(enable_fusion=False)
    model.eval().requires_grad_(False)
    model.to(device)

    # Need to convert numpy array to torch tensor for CLAP
    wav = torch.from_numpy(y).float().unsqueeze(0).to(device)
    emb = model.get_audio_embedding_from_data(x=wav, use_tensor=True)
    emb = emb.detach().cpu().numpy().squeeze()
    return emb

# ----------- 3) Process all the tracks -------------------
rows = []
for t in TRACKS:
    if not os.path.exists(t["path"]):
        print(f"[WARN] Missing file: {t['path']}")
        emb = np.full(512, np.nan)  # placeholder for now
    else:
        emb = get_embedding(t["path"])
    rows.append({**t, "embedding": emb})

df = pd.DataFrame(rows)

# Remove rows without valid embeddings
df = df[df["embedding"].apply(lambda v: isinstance(v, np.ndarray) and np.isfinite(v).all())].copy()
if df.empty:
    raise SystemExit("No valid audio files found. Need to add real MP3 files!")

# Normalize embeddings to enable cosine similarity (just dot product after normalizing)
def l2norm(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

df["emb_norm"] = df["embedding"].apply(l2norm)

# ----------- 4) Define the "shock" style vector -------------
# Average together the US innovator tracks to get the shock style
innovators = df[(df["country"] == "US") & (df["innovator"] == 1)]
if innovators.empty:
    raise SystemExit("Need at least one US innovator to define the shock style!")
shock_vec = l2norm(np.vstack(innovators["emb_norm"].to_list()).mean(axis=0))

# Now measure how similar each track is to this shock style
def cosim(u, v): 
    return float(np.dot(u, v))

df["sim_to_shock"] = df["emb_norm"].apply(lambda v: cosim(v, shock_vec))

# ----------- 5) Event study setup -----------------------
# UK is treated, US is control
df["treated_uk"] = (df["country"] == "UK").astype(int)
df["rel_year"] = df["year"] - EVENT_YEAR

# Bin years into event time (capped at ±5 years to keep it clean)
LOW, HIGH = -5, 5
df["event_bin"] = df["rel_year"].clip(LOW, HIGH)

# Setting up the regression: interactions of event time × UK treatment + FEs
ref_k = -1  # using -1 as the reference year
df["event_bin"] = df["event_bin"].astype(int)
d_event = pd.get_dummies(df["event_bin"], prefix="k")
d_country = pd.get_dummies(df["country"], prefix="cty", drop_first=True)
d_year = pd.get_dummies(df["year"], prefix="yr", drop_first=True)

# Build the interaction terms
Z_list = []
bin_cols = sorted([c for c in d_event.columns if c != f"k_{ref_k}"])
for c in bin_cols:
    Z_list.append(d_event[c].values[:, None] * df["treated_uk"].values[:, None])
Z = np.hstack(Z_list) if Z_list else np.empty((len(df), 0))

coef_names = bin_cols

# Put everything together: interactions + country FE + year FE + constant
X = np.column_stack([Z, d_country.values, d_year.values, np.ones(len(df))])
y = df["sim_to_shock"].values

model = sm.OLS(y, X).fit(cov_type="HC1")  # using robust standard errors

# Pull out just the interaction coefficients
b = model.params[:len(bin_cols)]
se = model.bse[:len(bin_cols)]

# Clean up names for plotting
ks = [int(name.split("_")[1]) for name in coef_names]
order = np.argsort(ks)
ks_ord = np.array(ks)[order]
b_ord = np.array(b)[order]
se_ord = np.array(se)[order]

# ----------- 6) Plot the results --------------------------------
plt.figure(figsize=(7,4.5))
plt.errorbar(ks_ord, b_ord, yerr=1.96*se_ord, fmt="o-", capsize=3)
plt.axvline(x=0, linestyle="--", label="Event")
plt.axhline(y=0, linestyle=":")
plt.title("Did UK bands copy US rock style after the British Invasion?")
plt.xlabel("Years relative to 1963")
plt.ylabel("Change in similarity (UK vs US)")
plt.legend()
plt.tight_layout()
plt.show()

# ----------- 7) Quick summary --------------------------------------------
print("\n=== Summary ===")
print(f"Processed {len(df)} tracks (US={sum(df.country=='US')}, UK={sum(df.country=='UK')})")
print(f"Using {len(innovators)} US innovator tracks to define the shock style")
print("Event bins:", sorted(df['event_bin'].unique()))
print("\nTop 5 most similar to US innovator style:")
print(df[["artist","title","country","year","sim_to_shock"]].sort_values("sim_to_shock", ascending=False).head(5).to_string(index=False))

