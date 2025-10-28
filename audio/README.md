# (Olaf) Audio Demo: Musical Diffusion Analysis

Event study examining whether UK bands copied US rock style after the British Invasion (1963).

Uses CLAP audio embeddings to measure musical similarity, then tests for convergence post-event using standard event-study regression.

## Setup

```bash
uv sync --group clap
```

CLAP will auto-download pretrained weights on first run.

## Usage

1. Add audio files (MP3/WAV) to `data/` folder
2. Update `TRACKS` list in `musical_diffusion_demo.py` with your filenames
3. Run: `uv run python audio/musical_diffusion_demo.py`

Output: event-time coefficients plot showing UK vs US similarity around 1963.
