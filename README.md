# Synthetic Text Generator

A modular package for generating synthetic text images with various realistic effects and transformations. Now includes support for processing HuggingFace datasets!

## Features

### 🎨 Text Generation
- **Multiple paper styles**: lined paper, old paper, birch bark, parchment
- **Advanced text rendering**: with baseline variations, color changes, and angle adjustments
- **Realistic paper textures**: using Perlin noise and fiber patterns

### 🔧 Advanced Effects
- **Geometric transformations**: cylindrical warping, washboard effects
- **Physical effects**: fold/crease simulation, ink bleeding, perspective distortion
- **Environmental effects**: shadow casting, lens distortion, scanner artifacts
- **Morphological operations**: erosion, dilation for text degradation

### 🤗 HuggingFace Dataset Support
- **Direct dataset processing**: Download and process CSV datasets from HuggingFace
- **Local CSV support**: Process local CSV files
- **Batch image generation**: Generate images for each text row
- **Automatic CSV creation**: Creates a new CSV with image paths and text

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the required font files in your font directory (default: `/content/static/`)

## Quick Start

### Basic Text Generation
```bash
# Generate single Sanskrit text samples
python main.py --mode single --output-dir ./output --base-images 5

# Generate comprehensive dataset with all effects
python main.py --mode comprehensive --output-dir ./comprehensive_output

# Generate ultra-realistic samples
python main.py --mode ultra-realistic --output-dir ./realistic_output
```

### HuggingFace Dataset Processing
```bash
# Process a HuggingFace dataset
python main.py --mode huggingface \
    --dataset-url "https://huggingface.co/datasets/username/dataset-name/raw/main/data.csv" \
    --text-column "text" \
    --output-dir ./hf_output \
    --max-samples 100

# Process a local CSV file
python main.py --mode huggingface \
    --csv-file ./my_dataset.csv \
    --text-column "sentence" \
    --output-dir ./csv_output \
    --max-samples 50
```

## Usage Examples

### Python API

```python
from synthetic_text_generator import (
    generate_enhanced_sanskrit_samples,
    HuggingFaceDatasetProcessor
)

# Generate text samples
images = generate_enhanced_sanskrit_samples(
    text="Your text here",
    output_dir="./output",
    params={'width': 600, 'height': 400}
)

# Process HuggingFace dataset
processor = HuggingFaceDatasetProcessor(output_dir="./hf_output")
success = processor.process_huggingface_dataset(
    dataset_url="https://huggingface.co/datasets/example/data/raw/main/dataset.csv",
    text_column="text",
    max_samples=100
)
```

### Simple Functions for Colab/Jupyter

```python
from main import (
    generate_colab_samples,
    process_huggingface_dataset_simple,
    process_local_csv_simple
)

# Generate samples in Colab
images = generate_colab_samples(base_images=5, width=600, height=400)

# Process HuggingFace dataset (simple)
process_huggingface_dataset_simple(
    dataset_url="https://huggingface.co/datasets/example/data.csv",
    text_column="text",
    max_samples=50
)

# Process local CSV (simple)
process_local_csv_simple(
    csv_path="./my_data.csv",
    text_column="sentence"
)
```

## Module Structure

```
synthetic_text_generator/
├── __init__.py              # Package initialization
├── config.py                # Configuration parameters
├── core.py                  # Main generation functions
├── effects.py               # Image effects and transformations
├── backgrounds.py           # Background generation
├── text_renderer.py         # Text rendering functionality
├── transformations.py       # Geometric transformations
└── huggingface_processor.py # HuggingFace dataset processing
```

## Configuration

The package uses `ENHANCED_DEFAULT_PARAMS` for configuration. You can override any parameter:

```python
custom_params = {
    'width': 800,
    'height': 600,
    'enable_advanced_effects': True,
    'fold_intensity': 0.4,
    'bleed_intensity': 0.3,
    # ... other parameters
}
```

## HuggingFace Dataset Format

Your dataset should be a CSV file with at least one text column. Example:

```csv
id,text,language
1,"Your text content here","en"
2,"Another text sample","en"
3,"More text data","en"
```

The processor will:
1. Download the dataset (if URL provided)
2. Load and validate the CSV
3. Generate an image for each text row
4. Create a new CSV with columns:
   - `image_path`: relative path to generated image
   - `text`: original text content
   - `style`: paper style used
   - `image_filename`: name of the image file
   - All other original columns

## Command Line Options

### Generation Modes
- `--mode single`: Generate images from built-in Sanskrit text
- `--mode comprehensive`: Generate systematic dataset with all effect combinations
- `--mode ultra-realistic`: Generate maximum realism samples
- `--mode huggingface`: Process HuggingFace datasets or CSV files

### HuggingFace Options
- `--dataset-url URL`: HuggingFace dataset URL (CSV format)
- `--text-column NAME`: Column name containing text (default: "text")
- `--max-samples N`: Maximum samples to process
- `--csv-file PATH`: Local CSV file to process

### Effect Parameters
- `--width`, `--height`: Image dimensions
- `--noise`, `--aging`, `--texture`: Effect intensities (0.0-1.0)
- `--enable-advanced-effects`: Enable advanced effects
- `--fold-intensity`, `--bleed-intensity`: Specific effect controls
- Many more options available (see `python main.py --help`)

## Output

### Regular Generation
- Individual image files (.png)
- Various transformations of each base image
- Organized by style and effect combination

### HuggingFace Processing
- `images/` directory with generated images
- `dataset.csv` with image paths and text
- `metadata.txt` with processing information

## Examples of Generated Effects

1. **Paper Textures**: Realistic paper fiber patterns using Perlin noise
2. **Aging Effects**: Edge darkening and aging patterns
3. **Physical Damage**: Fold lines, creases, and ink bleeding
4. **Scanner Artifacts**: Dust, compression artifacts, scanning lines
5. **Geometric Distortions**: Perspective changes, cylindrical warping
6. **Lighting Effects**: Shadows and lens distortions

## Font Requirements

The generator requires appropriate fonts for text rendering. Default configuration expects:
- Font directory: `/content/static/`
- Font file: `NotoSansOriya_ExtraCondensed-Regular.ttf`

You can specify custom fonts using `--font-dir` and `--font` parameters.

## Performance Tips

- Use `--max-samples` to limit processing for large datasets
- Disable advanced effects with `--no-advanced-effects` for faster generation
- Use multiprocessing with `--use-multiprocessing` for batch jobs
- Adjust image dimensions to balance quality and speed

## Error Handling

The package includes comprehensive error handling:
- Graceful fallbacks for missing dependencies
- Detailed logging for debugging
- Validation of input parameters
- Safe handling of malformed datasets

## Contributing

The modular structure makes it easy to extend:
- Add new effects in `effects.py`
- Implement new background styles in `backgrounds.py`
- Create custom transformations in `transformations.py`
- Extend dataset processing in `huggingface_processor.py`

## License

[Add your license information here]

---

**Note**: This is a complete rewrite of the original monolithic code into a modular, extensible package with added HuggingFace dataset processing capabilities.
