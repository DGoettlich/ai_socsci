"""
Simple visualization of songs in audio embedding space using MERT.

Maps ~10 songs including:
- US innovators (Chuck Berry, Little Richard)
- British Invasion bands (Beatles, Stones, Kinks)
- Modern control songs

Uses t-SNE and UMAP to reduce high-dimensional MERT embeddings to 2D for visualization.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel

# ----------- 1) Dataset ----------------------
TRACKS = [
    # US innovators
    {"path": "data/songs/chuck_berry_roll_over_beethoven.mp3", "artist": "Chuck Berry", "title": "Roll Over Beethoven", "year": 1956, "category": "US innovator"},
    
    # British Invasion
    {"path": "data/songs/beatles_roll_over_beethoven.mp3", "artist": "The Beatles", "title": "Roll Over Beethoven", "year": 1963, "category": "British Invasion"},
    {"path": "data/songs/beatles_yesterday.mp3", "artist": "The Beatles", "title": "Yesterday", "year": 1965, "category": "British Invasion"},
    {"path": "data/songs/rolling_stones_wild_horses.mp3", "artist": "The Rolling Stones", "title": "Wild Horses", "year": 1971, "category": "British Invasion"},
    
    # Modern pop
    {"path": "data/songs/billie_eilish_bad_guy.mp3", "artist": "Billie Eilish", "title": "bad guy", "year": 2019, "category": "Modern"},
    {"path": "data/songs/tate_mcrae_sports_car.mp3", "artist": "Tate McRae", "title": "Sports car", "year": 2024, "category": "Modern"},
    {"path": "data/songs/the_weeknd_blinding_lights.mp3", "artist": "The Weeknd", "title": "Blinding Lights", "year": 2019, "category": "Modern"},
    
    # Other eras
    {"path": "data/songs/david_bowie_life_on_mars.mp3", "artist": "David Bowie", "title": "Life On Mars", "year": 1971, "category": "70s/80s"},
    {"path": "data/songs/nena_99_luftballons.mp3", "artist": "Nena", "title": "99 Luftballons", "year": 1983, "category": "70s/80s"},
    {"path": "data/songs/pete_rodriguez_i_like_it_like_that.mp3", "artist": "Pete Rodriguez", "title": "I Like It Like That", "year": 1967, "category": "70s/80s"},
]

# ----------- 2) Audio embeddings with MERT ----------
def load_audio_mono(path, sr=24000, max_seconds=30):
    """Load audio and prep it for MERT. Keep clips short to speed things up."""
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

# Initialize MERT model once (downloads ~380MB on first run, cached afterwards)
print("Loading MERT model...")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
mert_model.eval().requires_grad_(False)
mert_model.to(device)
print("MERT model loaded successfully!")

def get_embedding(path):
    """Get audio embedding using MERT."""
    y, sr = load_audio_mono(path, sr=24000, max_seconds=30)
    
    # Process audio
    inputs = mert_processor(y, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings from all layers and average
    with torch.no_grad():
        outputs = mert_model(**inputs, output_hidden_states=True)
    
    # Average across all layers and time dimension
    all_hidden_states = torch.stack(outputs.hidden_states).squeeze()  # [layers, time, dim]
    emb = all_hidden_states.mean(dim=0).mean(dim=0).cpu().numpy()  # average across layers and time
    
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

# ----------- 4) Dimensionality reduction -------------------
embeddings = np.vstack(df["embedding"].to_list())

# Normalize embeddings for better similarity computation
def l2norm(v):
    """L2 normalize a vector."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

embeddings_normalized = np.array([l2norm(emb) for emb in embeddings])

# PCA
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings_normalized)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(df)-1))
embeddings_tsne = tsne.fit_transform(embeddings_normalized)

# UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_umap = umap_reducer.fit_transform(embeddings_normalized)

# Store in dataframe
df["pc1"] = embeddings_pca[:, 0]
df["pc2"] = embeddings_pca[:, 1]
df["tsne1"] = embeddings_tsne[:, 0]
df["tsne2"] = embeddings_tsne[:, 1]
df["umap1"] = embeddings_umap[:, 0]
df["umap2"] = embeddings_umap[:, 1]

# ----------- 5) Plot function -------------------
# Create figures directory
os.makedirs("figures", exist_ok=True)

def plot_embeddings(df, x_col, y_col, title, xlabel, ylabel, filename):
    """Helper function to plot embeddings and save to file."""
    plt.figure(figsize=(12, 9))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    categories = df["category"].unique()
    colors = {"US innovator": "#C41E3A", "British Invasion": "#00539B", "Modern": "#6A1B9A", "70s/80s": "#FF8C00"}
    markers = {"US innovator": "o", "British Invasion": "s", "Modern": "^", "70s/80s": "D"}
    
    for cat in categories:
        mask = df["category"] == cat
        plt.scatter(df[mask][x_col], df[mask][y_col], 
                    c=colors.get(cat, "black"), 
                    marker=markers.get(cat, "o"),
                    s=150, alpha=0.8, edgecolors='white', linewidths=1.5, label=cat)
    
    # Add labels with better positioning
    for _, row in df.iterrows():
        plt.annotate(f"{row['artist']} - {row['title']}", 
                    (row[x_col], row[y_col]),
                    fontsize=9, alpha=0.85, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
                    ha='center', va='bottom')
    
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=10, framealpha=0.9, loc='best')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: figures/{filename}")

# Plot t-SNE and UMAP
plot_embeddings(df, "tsne1", "tsne2",
                "Songs in MERT Embedding Space (t-SNE to 2D)",
                "t-SNE dimension 1", "t-SNE dimension 2",
                "songs_tsne.png")

plot_embeddings(df, "umap1", "umap2",
                "Songs in MERT Embedding Space (UMAP to 2D)",
                "UMAP dimension 1", "UMAP dimension 2",
                "songs_umap.png")

# ----------- 6) Summary -------------------
print("\n=== Summary ===")
print(f"Processed {len(df)} tracks")
print("\nTracks by category:")
print(df[["artist", "title", "category", "year"]].to_string(index=False))

# ----------- 7) Check similarity scores -------------------
print("\n=== Embedding Similarities ===")
from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity matrix (using normalized embeddings)
sim_matrix = cosine_similarity(embeddings_normalized)

# Find the two "Roll Over Beethoven" songs
chuck_idx = df[df['title'] == 'Roll Over Beethoven'].index[0]
beatles_idx = df[df['title'] == 'Roll Over Beethoven'].index[1]

chuck_artist = df.loc[chuck_idx, 'artist']
beatles_artist = df.loc[beatles_idx, 'artist']

similarity = sim_matrix[chuck_idx, beatles_idx]
print(f"\n{chuck_artist} vs {beatles_artist} (Roll Over Beethoven): {similarity:.3f}")

# Find most similar songs
print("\nMost similar pairs:")
for i in range(len(df)):
    for j in range(i+1, len(df)):
        sim = sim_matrix[i, j]
        print(f"{df.iloc[i]['artist']} - {df.iloc[i]['title']} vs {df.iloc[j]['artist']} - {df.iloc[j]['title']}: {sim:.3f}")
print("\n")
