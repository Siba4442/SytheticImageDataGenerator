# VisionSynth (Synthetic Image Data Generator)

VisionSynth is a Python project for generating synthetic text-in-image data for OCR and vision-language workflows. It is designed to ingest text from local CSV files and Hugging Face datasets, render text into images, apply background textures, and add visual degradations/effects.

This README gives an in-depth project walkthrough, including setup, architecture, and a full CLI argument reference from `src/visionsynth/cli.py`.

---

## 1) What this project is for

Use this project when you want to:

- Build synthetic datasets to train/evaluate OCR models.
- Create document-like text images with variable backgrounds and degradation.
- Bootstrap image-text pairs from Hugging Face datasets.
- Experiment with augmentation knobs such as blur, paper textures, shadows, and ink effects.

Typical output artifacts are:

- Rendered image files (e.g., `.jpg`, `.png`).
- A CSV file linking each image path to the source text and metadata.

---

## 2) Repository layout

```text
src/visionsynth/
  __init__.py                # package version
  __main__.py                # enables `python -m visionsynth`
  cli.py                     # CLI entrypoint (argparse + main())
  huggingface_processor.py   # Hugging Face dataset loader + Hub upload
  dataset_generator.py       # CSV loading, render pipeline, CSV/metadata writer
  text_renderer.py           # Text rendering utility
  background_generator.py    # Background generation styles
  effects_generator.py       # Post-processing visual effects
  distortion_generator.py    # Geometric warp helpers (not yet wired to the CLI)
tests/                       # pytest scaffold (test_cli.py, test_dataset_generator.py)
```

Other top-level files:

- `pyproject.toml`: project metadata (description, license, classifiers, keywords, URLs), dependencies, the `visionsynth` console-script entry point, `ruff`/`mypy`/`pytest` config, and the sdist file list (`[tool.hatch.build.targets.sdist]`).
- `LICENSE`: MIT.
- `uv.lock`: lockfile for reproducible installs with `uv`.

---

## 3) Installation and environment setup

### Requirements

- Python `>=3.12`

### Option A: with `uv` (recommended)

```bash
uv sync
```

This also installs the `dev` dependency group (`ruff`, `mypy`, `pytest`) into the project's local `.venv` — nothing is installed globally.

Then run commands with:

```bash
uv run visionsynth --help
# or, equivalently:
uv run python -m visionsynth --help
```

### Option B: with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

This installs the `visionsynth` package (and the `visionsynth` console script) from `src/visionsynth/` in editable mode, using the dependencies declared in `pyproject.toml`.

### Development tooling

```bash
uv run ruff check src tests   # lint
uv run ruff format src tests  # format
uv run mypy src                # type-check
uv run pytest                  # run the test suite (tests/ — a lean scaffold, not full coverage)
uv build                       # sanity-check the sdist/wheel actually build
```

Licensed under MIT (see `LICENSE`).

---

## 4) Data flow: from Hugging Face dataset to synthetic images

Use this conceptual pipeline:

1. **Choose source dataset**
   - Select a Hugging Face dataset ID (e.g., `imdb`, `ag_news`, `wikitext`).
   - Pick split (`train`, `test`, etc.) and optional config name.

2. **Load source text rows**
   - `huggingface_processor.py` calls `datasets.load_dataset(...)`.
   - Non-streaming mode materializes data as pandas DataFrame.

3. **Select text column**
   - Determine which field should be rendered (e.g., `text`, `sentence`, `question`).

4. **Render base text image**
   - Text renderer draws chosen text using the requested font.

5. **Apply background style**
   - Plain, Gaussian, or image backgrounds.
   - Optional lined/old/birch/parchment paper styles via `--background`.

6. **Apply post-processing effects**
   - Blur, fiber, fold creases, ink bleed, shadows, and strain.

7. **Write outputs**
   - Save images under `--output-dir`.
   - Save CSV metadata (`--output-csv`) describing image ↔ text mapping.

---

## 5) Quick start examples

### A) Generate using a Hugging Face dataset

```bash
uv run visionsynth \
  --dataset-name imdb \
  --split train \
  --output-dir output/ \
  --output-csv output.csv
```

### B) Add visual variety for OCR robustness

```bash
uv run visionsynth \
  --dataset-name ag_news \
  --split train \
  --output-dir output_aug/ \
  --background lined \
  --random-blur \
  --ink-bleed \
  --shadow
```

### C) Use CSV input

```bash
uv run visionsynth \
  --input-csv your_texts.csv \
  --output-dir output_csv/ \
  --output-csv labels.csv
```

---

## 6) Full CLI argument reference (`src/visionsynth/cli.py`)

Run `uv run visionsynth --help` for the authoritative, up-to-date listing (grouped exactly as below). All long flags use hyphens (e.g. `--output-dir`); only `--font`/`-f` and `--output-dir`/`-o` have short aliases.

### Input & output

- `--input-csv` *(str, default: `""`)* — path to a local input CSV file.
- `--dataset-name` *(str, default: `None`)* — Hugging Face dataset id, used instead of `--input-csv`.
- `--text-column` *(str, default: `"text"`)* — text column/field name.
- `-o, --output-dir` *(str, default: `output/`)* — directory for generated images and CSV.
- `--output-csv` *(str, default: `output.csv`)* — output CSV filename.
- `--extension` *(str, default: `.jpg`)* — output image file extension.
- `--max-samples` *(int, default: `None`)* — maximum number of rows to render (default: all).

### Text rendering

- `-f, --font` *(str, required)* — path to a `.ttf`/`.otf` font file.
- `--font-size` *(int, default: `28`)*
- `--width` *(int, default: `1000`)*
- `--height` *(int, default: `500`)*

### Background

- `--background` *(choice: `plain|gaussian|image|lined|old|birch|parchment`, default: `plain`)* — single flag selecting the background style.
- `--background-image-dir` *(str, default: `None`)* — source image directory, required when `--background image`.
- `--blur` *(int, default: `0`)*
- `--random-blur` *(flag)* — randomize blur between 0 and `--blur`.

### Birch paper (used with `--background birch`)

- `--birch-texture` *(float, default: `1.0`)* — spot-count multiplier.
- `--birch-spots` *(int, default: `150`)* — explicit spot count (overrides texture).
- `--birch-spot-radius` *('min,max' string, e.g. `12,30`, default: `10,25`)*
- `--birch-intensity` *(int, default: `10`)*
- `--birch-spot-sign` *(int, default: `-1`)* — 1 lighter, -1 darker, 0 random.
- `--birch-irregularity` *(float, default: `0.35`)*
- `--birch-blur` *(float, default: `1.2`)* — Gaussian blur sigma for spot softening.
- `--birch-max-spots` *(int, default: `2000`)* — safety cap on spot count.

### Parchment paper (used with `--background parchment`)

- `--parchment-texture` *(float, default: `1.0`)*
- `--parchment-spots` *(int, default: `400`)*
- `--parchment-spot-radius` *('min,max' string, e.g. `12,30`, default: `3,12`)*
- `--parchment-intensity` *(int, default: `7`)*
- `--parchment-spot-sign` *(int, default: `-1`)*
- `--parchment-irregularity` *(float, default: `0.35`)*
- `--parchment-blur` *(float, default: `1.2`)*
- `--parchment-max-spots` *(int, default: `2000`)*

### Effects

- `--fiber` *(flag)* / `--fiber-density` *(float, default: `0.2`)*
- `--fold-creases` *(flag)* / `--fold-creases-intensity` *(int, default: `20`)*
- `--ink-bleed` *(flag)* / `--ink-bleed-intensity` *(float, default: `0.3`)* / `--ink-bleed-radius` *(int, default: `3`)*
- `--shadow` *(flag)* / `--shadow-intensity` *(float, default: `0.4`)* / `--shadow-angle` *(float, default: `45`)*
- `--strain` *(flag)*

### Hugging Face dataset source

- `--config-name` *(str, default: `None`)*
- `--split` *(str, default: `None`)*
- `--streaming` *(flag)*

### Hugging Face Hub upload

- `--push-to-hub` *(flag)*
- `--hub-repo-id` *(str, default: `None`)*
- `--hub-private` *(flag)*

---

## 7) End-to-end usage strategy for Hugging Face datasets

To convert HF dataset text into synthetic images reliably:

1. Start with a small subset (`--max-samples` / split sample) to validate formatting.
2. Inspect the dataset schema and identify a safe text field.
3. Run a baseline render with minimal effects.
4. Gradually introduce blur/background/effects.
5. Validate output labels (`--output-csv`) against image files.
6. Scale up generation after quality checks.

Recommended progression:

- **Pass 1**: plain background, fixed blur 0, one font.
- **Pass 2**: add lined/old/parchment `--background` styles.
- **Pass 3**: add random blur + shadows + ink bleed.
- **Pass 4**: use multiple fonts and style mixes.

---

## 8) Practical tips for better synthetic variety

- Use multiple fonts to avoid overfitting to typography.
- Mix clean and degraded samples.
- Keep labels exact (no accidental text normalization unless intentional).
- Use train/test synthetic style mismatch carefully (can hurt generalization).
- Keep a reproducibility log (CLI command + git commit hash + output path).

---

## 9) Known caveats in current code state

- `lined_paper()` in `background_generator.py` takes only `height`/`width` — line spacing/width/intensity aren't configurable (`--background lined` always uses the function's internal randomized defaults). The CLI has no flags for this on purpose (they were previously present but silently ignored, so they were removed rather than kept as dead options).
- `distortion_generator.py`'s warp functions and a few `effects_generator.py` functions (`apply_perspective_distortion`, `apply_morphological_operations`, `simulate_scanner_artifacts`, `apply_lens_distortion`) exist but aren't exposed as CLI flags yet.
- `huggingface_processor.py: load_huggingface_dataset` and `distortion_generator.py`'s warp functions swallow exceptions and return `None`/the input image unchanged on error.

---

## 10) Troubleshooting

### `ModuleNotFoundError: datasets` or `pandas`
Run `uv sync` (or, with plain pip, `pip install -e .`) so all dependencies declared in `pyproject.toml` are installed.

### HF dataset fails to load
- Verify dataset ID and split exist.
- If dataset has configs, provide `--config-name`.
- Try without `--streaming` first for easier debugging.

### No output images created
- Verify `--output-dir` exists and is writable.
- Test with tiny sample before large runs (`--max-samples`).

---

## 11) Suggested next improvements

- Grow `tests/` beyond the current lean scaffold (e.g. an end-to-end test rendering a few images with a real font).
- Add deterministic seed control for reproducibility.
- Add a formal schema for `--output-csv`.
- Publish releases to PyPI once the API/CLI is considered stable (metadata and `uv build` are already in place).

---

## 12) License

MIT — see `LICENSE`.

