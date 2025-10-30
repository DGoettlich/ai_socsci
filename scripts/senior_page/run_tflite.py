"""Run the senior page classifier on every JPG image in the data folder.

Pipeline overview:
1. Discover the Teachable Machine assets (TFLite graph plus label dictionary).
2. Load each image from ``data/senior_page/`` and convert it into the exact tensor format the
   model expects (shape, dtype, quantization).
3. Invoke the interpreter to obtain raw quantized scores, persist those raw values,
   dequantize them into human-friendly confidences, and emit final predictions.
4. Save both the per-image predictions and all intermediate score tables under ``output/senior_page/``.

Adjust the path constants below when relocating the model, labels, inputs, or outputs.
"""

from __future__ import annotations

import contextlib
import csv
import logging
import os
import sys
import tempfile
import warnings
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image

ArrayLike = np.ndarray  # Clarify that our helper functions expect NumPy arrays.
BASE_ROOT = Path(r"C:\Users\ecompe\Documents\IMAGES\ai_socsci")

# --- Base configuration ----------------------------------------------------
# Update these paths when relocating the model, labels, inputs, or outputs.
MODEL_VERSION = "model-8999930383369764864"
MODEL_ROOT = BASE_ROOT / "vertex_model" / "senior_page" / MODEL_VERSION
MODEL_PATH: Optional[Union[str, Path]] = None
MODEL_DIR: Union[str, Path] = MODEL_ROOT / "tflite"
MODEL_PATTERN = "**/*.tflite"

LABELS_PATH: Optional[Union[str, Path]] = None
LABELS_DIR: Union[str, Path] = MODEL_ROOT / "tf-js"
LABELS_PATTERN = "**/*dict.txt"

DATA_DIR: Union[str, Path] = BASE_ROOT / "data" / "senior_page"
OUTPUT_DIR: Union[str, Path] = BASE_ROOT / "output" / "senior_page"
# ---------------------------------------------------------------------------


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TFLITE_LOG_LEVEL", "0")
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.lite\.Interpreter is deprecated.*",
    category=UserWarning,
)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass


@contextlib.contextmanager
def suppress_native_logs() -> None:
    """Silence C-level stdout/stderr noise emitted by TensorFlow Lite."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, ValueError, OSError):
        yield
        return

    with tempfile.TemporaryFile(mode="w+b") as tmp:
        saved_fds = []
        try:
            for fd in (stdout_fd, stderr_fd):
                duplicate = os.dup(fd)
                saved_fds.append(duplicate)
                os.dup2(tmp.fileno(), fd)
            yield
        finally:
            for fd, duplicate in zip((stdout_fd, stderr_fd), saved_fds):
                os.dup2(duplicate, fd)
                os.close(duplicate)


# Try the lightweight tflite-runtime first; fall back to TensorFlow if needed.
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Install either 'tflite-runtime' or 'tensorflow' to run this script."
        ) from exc

    Interpreter = tf.lite.Interpreter  # type: ignore[attr-defined]


def to_path(path: Union[str, Path]) -> Path:
    """Coerce string-like values into Path objects."""
    return path if isinstance(path, Path) else Path(path)


def normalize_path(path: Union[str, Path]) -> Path:
    """Expand user-relative paths and anchor relative paths to the project root."""
    candidate = to_path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (BASE_ROOT / candidate).resolve()


def pick_artifact(
    provided: Optional[Union[str, Path]],
    description: str,
    *,
    search_root: Union[str, Path],
    pattern: str,
) -> Path:
    """Return the user-specified artifact or find one via pattern search."""
    if provided is not None:
        candidate = normalize_path(provided)
        if not candidate.is_file():
            raise SystemExit(f"Provided {description} not found: {candidate}")
        return candidate

    search_root = normalize_path(search_root)
    if not search_root.exists():
        raise SystemExit(f"Search directory for {description} not found: {search_root}")

    matches = sorted(search_root.glob(pattern))
    if not matches:
        raise SystemExit(
            f"Could not find {description} matching pattern '{pattern}' under {search_root}"
        )
    return matches[-1].resolve()


def load_labels(label_path: Path) -> List[str]:
    """Read class labels from the Teachable Machine dictionary file."""
    return [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sanitize_label(label: str) -> str:
    """Lowercase label names and replace unsafe characters with underscores."""
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label.lower())


def preprocess(image_path: Path, input_details: dict) -> ArrayLike:
    """Prepare an image tensor that matches the model's expected dtype and shape."""
    height = int(input_details["shape"][1])
    width = int(input_details["shape"][2])

    # Resize to the spatial footprint the TFLite graph reports via tensor metadata.
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.BILINEAR)
    array = np.asarray(image)

    input_dtype = input_details["dtype"]
    if np.issubdtype(input_dtype, np.floating):
        # Floating-point graphs expect normalized pixels in the 0..1 range.
        array = array.astype(np.float32) / 255.0
    else:
        # Quantized graphs operate directly on uint8 bytes, so copy without scaling.
        array = array.astype(input_dtype, copy=False)

    return np.expand_dims(array, axis=0).astype(input_dtype, copy=False)


def main() -> None:
    """Locate assets, run inference, and print one line per prediction."""
    model_path = pick_artifact(
        MODEL_PATH,
        "model (.tflite)",
        search_root=MODEL_DIR,
        pattern=MODEL_PATTERN,
    )
    labels_path = pick_artifact(
        LABELS_PATH,
        "label dictionary",
        search_root=LABELS_DIR,
        pattern=LABELS_PATTERN,
    )

    # Step 2: gather evaluation images (flat list of JPGs from the data directory).
    data_dir = normalize_path(DATA_DIR)
    images = sorted(p for p in data_dir.glob("*.jpg") if p.is_file())
    if not images:
        raise SystemExit(f"No .jpg images found in '{data_dir}'.")

    output_dir = normalize_path(OUTPUT_DIR)

    labels = load_labels(labels_path)
    label_keys = [sanitize_label(label) for label in labels]

    # Step 3: boot the interpreter so we can query tensor metadata and run inference.
    with suppress_native_logs():
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Quantization parameters let us reverse the byte-level logits into floats.
    output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))

    # Step 4: pre-create the reporting folders (summary, intermediary floats, raw bytes).
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.txt"

    intermediary_dir = output_dir / "intermediary"
    intermediary_dir.mkdir(exist_ok=True)
    scores_path = intermediary_dir / "scores.csv"
    raw_dir = intermediary_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    legacy_csv = raw_dir / "scores_raw.csv"
    if legacy_csv.exists():
        legacy_csv.unlink()

    # Collect rows containing the raw uint8 logits for quick inspection.
    score_rows = []

    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in images:
            # Step 5a: feed input tensor and execute the forward pass.
            input_data = preprocess(image_path, input_details=input_details)
            interpreter.set_tensor(input_details["index"], input_data)
            with suppress_native_logs():
                interpreter.invoke()

            # Raw model output is quantized uint8 logits centered at zero_point.
            raw_scores = interpreter.get_tensor(output_details["index"])[0]
            row = {"filename": image_path.name}
            for key, raw_value in zip(label_keys, raw_scores):
                row[f"raw_{key}"] = int(raw_value)
            score_rows.append(row)

            # Persist the exact numpy output in .npy format (dtype/shape preserved).
            raw_file_path = raw_dir / f"{image_path.stem}.npy"
            np.save(raw_file_path, raw_scores, allow_pickle=False)

            # Convert raw logits into calibrated float confidences before ranking.
            scores = raw_scores
            if np.issubdtype(scores.dtype, np.integer) and output_scale:
                scores = (scores.astype(np.float32) - float(output_zero_point)) * float(
                    output_scale
                )
            else:
                scores = scores.astype(np.float32)

            # Step 5b: rank the calibrated confidences to pick the winning class.
            best_idx = int(np.argmax(scores))
            confidence = float(scores[best_idx])
            label = labels[best_idx] if best_idx < len(labels) else f"class_{best_idx}"

            line = f"{image_path.name}: {label} ({confidence:.3f})"
            print(line)
            handle.write(line + "\n")

    # Step 6: persist the raw uint8 logits for each image (direct model output).
    if score_rows:
        raw_fieldnames = ["filename"] + [f"raw_{key}" for key in label_keys]
        with scores_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=raw_fieldnames)
            writer.writeheader()
            writer.writerows(score_rows)


if __name__ == "__main__":
    main()
