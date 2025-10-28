# (Olaf) Audio Demo: Embedding Space Visualization

Simple visualization of songs in audio embedding space using MERT (Music Understanding Model).

Maps 11 songs including:

- US Innovators (Chuck Berry, Elvis Presley)
- British Invasion bands (The Beatles, The Kinks, Rolling Stones)
- 80s Pop (Madonna, Prince)
- Modern pop (Billie Eilish, Tate McRae, The Weeknd)

Uses t-SNE and UMAP to reduce high-dimensional MERT embeddings to 2D for visualization.

## Setup

Core dependencies are already installed. MERT will auto-download pretrained weights (~380MB) on first run from Hugging Face.

## Usage

The songs are already in `data/songs/` folder. Just run:

```bash
uv run python audio/musical_diffusion_demo.py
```

Output: two PNG files in `figures/` folder showing songs in 2D embedding space (t-SNE and UMAP projections).
