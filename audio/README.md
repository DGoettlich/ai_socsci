# (Olaf) Audio Demo: Embedding Space Visualization

Simple visualization of songs in audio embedding space using MERT (Music Understanding Model).

Maps 10 songs including:

- US innovators (Chuck Berry)
- British Invasion bands (Beatles, Rolling Stones)
- Modern pop (Billie Eilish, Tate McRae, The Weeknd)
- 70s/80s (David Bowie, Nena, Pete Rodriguez)

Uses t-SNE and UMAP to reduce high-dimensional MERT embeddings to 2D for visualization.

## Setup

Core dependencies are already installed. MERT will auto-download pretrained weights (~380MB) on first run from Hugging Face.

## Usage

The songs are already in `data/songs/` folder. Just run:

```bash
uv run python audio/musical_diffusion_demo.py
```

Output: two PNG files in `figures/` folder showing songs in 2D embedding space (t-SNE and UMAP projections).
