![Tests](https://github.com/DGoettlich/repo-template/actions/workflows/test.yaml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)

# Guide

## Audio Demo: Embedding Space Visualization

This is a simple demo that visualizes songs in audio embedding space using MERT (Music Understanding Model). It demonstrates how AI models can learn representations of music that capture acoustic and stylistic similarities.

The demo analyzes 11 songs spanning different eras and genres:

- **US Innovators** (1950s): Chuck Berry, Elvis Presley
- **British Invasion** (1960s-70s): The Beatles, The Kinks, Rolling Stones
- **80s Pop**: Madonna, Prince
- **Modern Pop** (2010s-2020s): Billie Eilish, Tate McRae, The Weeknd

The visualization uses t-SNE and UMAP to project high-dimensional MERT embeddings into 2D space.

### Running the Demo

Songs are already in the `data/songs/` folder. To generate the visualizations:

```bash
uv run python audio/musical_diffusion_demo.py
```

This will create two PNG files in the `figures/` folder showing songs in 2D embedding space (t-SNE and UMAP projections).

See `audio/README.md` for more details.
