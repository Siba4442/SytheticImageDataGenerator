# VisionSynth (Synthetic Image Data Generator)

VisionSynth is a Python project for generating synthetic text-in-image data for OCR and vision-language workflows. It is designed to ingest text from local CSV files and Hugging Face datasets, render text into images, apply background textures, and add visual degradations/effects.

This README gives an in-depth project walkthrough, including setup, architecture, and a full CLI argument reference from `VisionSynth/run.py`.

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
VisionSynth/
  run.py                    # CLI entrypoint + argument parser (currently parser-focused)
  huggingface_processor.py  # Hugging Face dataset loader using `datasets`
  dataset_generator.py      # Dataset loading and CSV-saving helpers
  text_renderer.py          # Text rendering utility
  background_generator.py   # Background generation styles
  effects_generator.py      # Post-processing visual effects
  distrotion_generator.py   # Distortion helpers
```

Other top-level files:

- `pyproject.toml`: project metadata + core dependencies.
- `uv.lock`: lockfile for reproducible installs with `uv`.

---

## 3) Installation and environment setup

### Requirements

- Python `>=3.12`

### Option A: with `uv` (recommended)

```bash
uv sync
```

Then run commands with:

```bash
uv run python VisionSynth/run.py --help
```

### Option B: with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install noise numpy opencv-python pillow scipy pandas datasets
```

> Note: `pandas` and `datasets` are required by Hugging Face/CSV processing modules, even if not listed in core dependencies.

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
   - Optional lined/old/brownish/parchment paper transforms.

6. **Apply post-processing effects**
   - Blur, fiber, fold creases, ink bleed, shadows, and strain.

7. **Write outputs**
   - Save images in `output_dir`.
   - Save CSV metadata (`output_csv`) describing image ↔ text mapping.

---

## 5) Quick start examples

### A) Generate using a Hugging Face dataset

```bash
uv run python VisionSynth/run.py \
  --dataset_name imdb \
  --split train \
  --output_dir output/ \
  --output_csv output.csv
```

### B) Add visual variety for OCR robustness

```bash
uv run python VisionSynth/run.py \
  --dataset_name ag_news \
  --split train \
  --output_dir output_aug/ \
  --background gaussian \
  --lined_paper \
  --random_line_spacing \
  --random_blur_level \
  --effect_ink_bleed \
  --effect_shadow
```

### C) Use CSV input

```bash
uv run python VisionSynth/run.py \
  --input_csv your_texts.csv \
  --output_dir output_csv/ \
  --output_csv labels.csv
```

---

## 6) Full CLI argument reference (`VisionSynth/run.py`)

The following options are defined in the parser.

### Core I/O

- `--output_dir` *(str, default: `output/`)*
  - Output directory for generated images and CSV files.
- `-ic, --input_csv` *(str, default: `""`)*
  - Path to local input CSV file.
- `-e, --extension` *(str, default: `.jpg`)*
  - Output image extension.
- `--output_csv` *(str, default: `output.csv`)*
  - Name/path for output CSV file.
- `--csv_file` *(str, default: `output.csv`)*
  - CSV path used when accumulating.
- `-f, --font` *(str, default: `None`)*
  - Font file path to render text.

### Dataset source (Hugging Face)

- `-name, --dataset_name` *(str, default: `None`)*
  - Hugging Face dataset ID (e.g., `imdb`, `username/dataset`).
- `-conf, --config_name` *(str, default: `None`)*
  - Optional dataset config/subset.
- `-sp, --split` *(str, default: `None`)*
  - Dataset split to load.
- `-s, --streaming` *(flag, default: `False`)*
  - Enable streaming mode.

### Row/window control

- `-rs, --row_start` *(int, default: `0`)*
  - Start processing from this row index.
- `-r, --rows` *(str, default: `None`)*
  - Row count/checkpoint-related parameter (description in source indicates resume/checkpoint usage).
- `-ac --accumulate` *(flag, default: `False`)*
  - Accumulate into an existing CSV.

### Basic augmentation

- `-bl, --blur_level` *(int, default: `0`)*
  - Fixed blur level.
- `-rbl, --random_blur_level` *(flag, default: `False`)*
  - Random blur between 0 and `blur_level`.

### Background selection

- `-b, --background` *(choice: `plain|gaussian|image`, default: `plain`)*
  - Base background type.

### Lined paper options

- `-ln, --lined_paper` *(flag)*
- `-lns, --line_spacing` *(int, default: `15`)*
- `-rlns, --random_line_spacing` *(flag)*
- `-min_lns, --min_line_spacing` *(int, default: `15`)*
- `-max_lns, --max_line_spacing` *(int, default: `25`)*
- `-lni, --line_intensity` *(int, default: `100`)*
- `-lnw, --line_width` *(int, default: `1`)*
- `-rlnw, --random_line_width` *(flag)*
- `-max_lnw, --max_line_width` *(int, default: `2`)*

### Old paper options

- `-old, --old_paper` *(flag)*
- `-edw, --edge_width` *(float, default: `0.1`)*
- `-ai, --aging_intensity` *(int, default: `15`)*

### Brownish paper (named `brich_*` in CLI)

- `-bh, --brich_paper` *(flag)*
- `-bht, --brich_texture` *(float, default: `1.0`)*
- `-bhs, --brich_spots` *(int, default: `150`)*
- `-bhr, --brich_spot_radius` *(tuple, default: `(10, 25)`)*
- `-bhi, --brich_sport_intensity` *(int, default: `10`)*
- `-bhsi, --brich_spot_sign` *(int, default: `-1`)*
- `-bhir, --brich_irregularity` *(float, default: `0.35`)*
- `-bhbl, --brich_blur` *(float, default: `1.2`)*
- `-bhc, --brich_capspots` *(int, default: `2000`)*

### Parchment paper options

- `-ph, --parchment_paper` *(flag)*
- `-pht, --parchment_texture` *(float, default: `1.0`)*
- `-phs, --parchment_spots` *(int, default: `150`)*
- `-phr, --parchment_spot_radius` *(tuple, default: `(10, 25)`)*
- `-phi, --parchment_spot_intensity` *(int, default: `10`)*
- `-phsi, --parchment_spot_sign` *(int, default: `1`)*
- `-phir, --parchment_irregularity` *(float, default: `0.35`)*
- `-phbl, --parchment_blur` *(float, default: `1.2`)*
- `-phc, --parchment_capspots` *(int, default: `2000`)*

### Additional effects

- `-eff, --effect_fiber` *(flag)*
- `-effd, --effect_fiber_density` *(float, default: `0.2`)*
- `-efc, --effect_fold_creases` *(flag)*
- `-efci, --effect_fold_creases_intensity` *(int, default: `20`)*
- `-efink, --effect_ink_bleed` *(flag)*
- `-efinki, --effect_ink_bleed_intensity` *(float, default: `0.3`)*
- `-efinkr, --effect_ink_bleed_radius` *(int, default: `3`)*
- `-efsh, --effect_shadow` *(flag)*
- `-efshi, --effect_shadow_intensity` *(float, default: `0.4`)*
- `-efshr, --effect_shadow_radius` *(int, default: `5`)*
- `-efsa, --effect_shadows_angle` *(float, default: `45`)*
- `-efst, --effect_strain` *(flag)*

---

## 7) End-to-end usage strategy for Hugging Face datasets

To convert HF dataset text into synthetic images reliably:

1. Start with a small subset (`--rows` / split sample) to validate formatting.
2. Inspect the dataset schema and identify a safe text field.
3. Run a baseline render with minimal effects.
4. Gradually introduce blur/background/effects.
5. Validate output labels (`output_csv`) against image files.
6. Scale up generation after quality checks.

Recommended progression:

- **Pass 1**: plain background, fixed blur 0, one font.
- **Pass 2**: add lined/old/parchment toggles.
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

Based on current source files:

- `VisionSynth/run.py` is parser-heavy and appears truncated after output directory creation.
- `--effect_shadows_angle` is defined twice in parser.
- `-ac --accumulate` appears as a concatenated short/long string token in code (likely needs correction to `"-ac", "--accumulate"`).
- `VisionSynth/data_generator.py` is currently empty.

If you plan production use, review and complete the execution path in `run.py` and validate parser behavior with `--help` and smoke tests.

---

## 10) Troubleshooting

### `ModuleNotFoundError: datasets` or `pandas`
Install missing packages:

```bash
pip install datasets pandas
```

### HF dataset fails to load
- Verify dataset ID and split exist.
- If dataset has configs, provide `--config_name`.
- Try without `--streaming` first for easier debugging.

### No output images created
- Confirm `run.py` main generation flow is complete in your branch.
- Verify `output_dir` exists and is writable.
- Test with tiny sample before large runs.

---

## 11) Suggested next improvements

- Add a `--text_column` CLI argument explicitly.
- Add deterministic seed control for reproducibility.
- Add a formal schema for `output_csv`.
- Add unit tests for parser and HF dataset loading.
- Add integration test generating 5 images from a toy dataset.

---

## 12) License

Add your project license here (MIT/Apache-2.0/etc.) if you intend to distribute.

