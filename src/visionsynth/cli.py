import argparse
import itertools
import os
import sys
from typing import cast

import pandas as pd

from . import dataset_generator, huggingface_processor

BACKGROUND_CHOICES = ["plain", "gaussian", "image", "lined", "old", "birch", "parchment"]


def _int_pair(value: str) -> tuple[int, int]:
    """argparse type for a 'min,max' pair, e.g. '10,25' -> (10, 25)."""
    try:
        low_str, high_str = value.split(",")
        return int(low_str), int(high_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected 'min,max' (e.g. '10,25'), got {value!r}"
        ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visionsynth",
        description="Generate synthetic text-in-image data for OCR and VLM training.",
    )

    io_group = parser.add_argument_group("input & output")
    io_group.add_argument("--input-csv", default="", help="Path to a local input CSV file.")
    io_group.add_argument(
        "--dataset-name", help="Hugging Face dataset id to use instead of --input-csv."
    )
    io_group.add_argument("--text-column", default="text", help="Text column/field name.")
    io_group.add_argument(
        "-o", "--output-dir", default="output/", help="Directory for generated images and CSV."
    )
    io_group.add_argument("--output-csv", default="output.csv", help="Output CSV filename.")
    io_group.add_argument("--extension", default=".jpg", help="Output image file extension.")
    io_group.add_argument(
        "--max-samples", type=int, help="Maximum number of rows to render (default: all)."
    )

    text_group = parser.add_argument_group("text rendering")
    text_group.add_argument(
        "-f", "--font", help="Path to a .ttf/.otf font file to render text with. Required."
    )
    text_group.add_argument("--font-size", type=int, default=28, help="Font size in points.")
    text_group.add_argument("--width", type=int, default=1000, help="Output image width, px.")
    text_group.add_argument("--height", type=int, default=500, help="Output image height, px.")

    bg_group = parser.add_argument_group("background")
    bg_group.add_argument(
        "--background",
        choices=BACKGROUND_CHOICES,
        default="plain",
        help="Background style to render text on.",
    )
    bg_group.add_argument(
        "--background-image-dir", help="Directory of source images for --background image."
    )
    bg_group.add_argument("--blur", type=int, default=0, help="Blur radius to apply.")
    bg_group.add_argument(
        "--random-blur", action="store_true", help="Randomize blur between 0 and --blur."
    )

    birch_group = parser.add_argument_group("birch paper (used with --background birch)")
    birch_group.add_argument(
        "--birch-texture", type=float, default=1.0, help="Multiplier for spot count."
    )
    birch_group.add_argument(
        "--birch-spots", type=int, default=150, help="Explicit spot count (overrides texture)."
    )
    birch_group.add_argument(
        "--birch-spot-radius",
        type=_int_pair,
        default=(10, 25),
        help="Spot radius range as 'min,max'.",
    )
    birch_group.add_argument(
        "--birch-intensity", type=int, default=10, help="Spot intensity magnitude."
    )
    birch_group.add_argument(
        "--birch-spot-sign",
        type=int,
        default=-1,
        help="1 for lighter spots, -1 for darker, 0 for random.",
    )
    birch_group.add_argument(
        "--birch-irregularity", type=float, default=0.35, help="Spot shape irregularity, 0-1."
    )
    birch_group.add_argument(
        "--birch-blur", type=float, default=1.2, help="Gaussian blur sigma for spot softening."
    )
    birch_group.add_argument(
        "--birch-max-spots", type=int, default=2000, help="Safety cap on spot count."
    )

    parchment_group = parser.add_argument_group(
        "parchment paper (used with --background parchment)"
    )
    parchment_group.add_argument(
        "--parchment-texture", type=float, default=1.0, help="Multiplier for spot count."
    )
    parchment_group.add_argument(
        "--parchment-spots", type=int, default=400, help="Explicit spot count (overrides texture)."
    )
    parchment_group.add_argument(
        "--parchment-spot-radius",
        type=_int_pair,
        default=(3, 12),
        help="Spot radius range as 'min,max'.",
    )
    parchment_group.add_argument(
        "--parchment-intensity", type=int, default=7, help="Spot intensity magnitude."
    )
    parchment_group.add_argument(
        "--parchment-spot-sign",
        type=int,
        default=-1,
        help="1 for lighter spots, -1 for darker, 0 for random.",
    )
    parchment_group.add_argument(
        "--parchment-irregularity", type=float, default=0.35, help="Spot shape irregularity, 0-1."
    )
    parchment_group.add_argument(
        "--parchment-blur", type=float, default=1.2, help="Gaussian blur sigma for spot softening."
    )
    parchment_group.add_argument(
        "--parchment-max-spots", type=int, default=2000, help="Safety cap on spot count."
    )

    effects_group = parser.add_argument_group("effects")
    effects_group.add_argument("--fiber", action="store_true", help="Apply paper fiber texture.")
    effects_group.add_argument(
        "--fiber-density", type=float, default=0.2, help="Paper fiber texture density."
    )
    effects_group.add_argument(
        "--fold-creases", action="store_true", help="Apply paper fold crease effect."
    )
    effects_group.add_argument(
        "--fold-creases-intensity", type=int, default=20, help="Fold crease intensity."
    )
    effects_group.add_argument("--ink-bleed", action="store_true", help="Apply ink bleed effect.")
    effects_group.add_argument(
        "--ink-bleed-intensity", type=float, default=0.3, help="Ink bleed effect intensity."
    )
    effects_group.add_argument(
        "--ink-bleed-radius", type=int, default=3, help="Ink bleed spread radius."
    )
    effects_group.add_argument(
        "--shadow", action="store_true", help="Apply a directional shadow effect."
    )
    effects_group.add_argument(
        "--shadow-intensity", type=float, default=0.4, help="Shadow effect intensity."
    )
    effects_group.add_argument(
        "--shadow-angle", type=float, default=45, help="Shadow direction angle, in degrees."
    )
    effects_group.add_argument(
        "--strain", action="store_true", help="Apply a strain-mark discoloration overlay."
    )

    hf_group = parser.add_argument_group("hugging face dataset source")
    hf_group.add_argument("--config-name", help="Dataset config/subset name.")
    hf_group.add_argument("--split", help="Dataset split to load, e.g. 'train'.")
    hf_group.add_argument(
        "--streaming", action="store_true", help="Load the dataset in streaming mode."
    )

    hub_group = parser.add_argument_group("hugging face hub upload")
    hub_group.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the generated output directory to the Hub as a dataset repo.",
    )
    hub_group.add_argument(
        "--hub-repo-id", help="Target dataset repo id, e.g. 'username/my-synthetic-dataset'."
    )
    hub_group.add_argument(
        "--hub-private", action="store_true", help="Create the Hub dataset repo as private."
    )

    return parser.parse_args()


def _load_source_dataframe(args: argparse.Namespace) -> pd.DataFrame | None:
    if args.input_csv:
        df = dataset_generator.load_dataset(args.input_csv, args.text_column)
        if df is None:
            print(f"Failed to load CSV '{args.input_csv}' (check the file path and --text-column).")
            return None
        return df

    if args.dataset_name:
        dataset = huggingface_processor.load_huggingface_dataset(
            args.dataset_name,
            config_name=args.config_name,
            split=args.split,
            streaming=args.streaming,
        )
        if dataset is None:
            print(f"Failed to load Hugging Face dataset '{args.dataset_name}'.")
            return None

        if args.streaming:
            limit = args.max_samples or 1000
            rows = list(itertools.islice(dataset, limit))
            return pd.DataFrame(rows)

        return cast(pd.DataFrame, dataset)

    print("Provide either --input-csv or --dataset-name as the text source.")
    return None


def main() -> None:
    args = parse_arguments()

    if not args.font:
        print("--font is required: pass the path to a .ttf/.otf file to render text with.")
        sys.exit(1)

    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    df = _load_source_dataframe(args)
    if df is None:
        sys.exit(1)

    if args.text_column not in df.columns:
        print(f"Column '{args.text_column}' not found. Available columns: {list(df.columns)}")
        sys.exit(1)

    params = {
        "width": args.width,
        "height": args.height,
        "font": args.font,
        "font_size": args.font_size,
        "extension": args.extension,
        "background": args.background,
        "background_image_dir": args.background_image_dir,
        "birch_texture": args.birch_texture,
        "birch_spots": args.birch_spots,
        "birch_min_radius": args.birch_spot_radius[0],
        "birch_max_radius": args.birch_spot_radius[1],
        "birch_intensity": args.birch_intensity,
        "birch_spot_sign": args.birch_spot_sign,
        "birch_irregularity": args.birch_irregularity,
        "birch_blur": args.birch_blur,
        "birch_cap_spots": args.birch_max_spots,
        "parchment_texture": args.parchment_texture,
        "parchment_spots": args.parchment_spots,
        "parchment_min_radius": args.parchment_spot_radius[0],
        "parchment_max_radius": args.parchment_spot_radius[1],
        "parchment_intensity": args.parchment_intensity,
        "parchment_spot_sign": args.parchment_spot_sign,
        "parchment_irregularity": args.parchment_irregularity,
        "parchment_blur": args.parchment_blur,
        "parchment_cap_spots": args.parchment_max_spots,
        "blur_level": args.blur,
        "random_blur_level": args.random_blur,
        "effect_fiber": args.fiber,
        "effect_fiber_density": args.fiber_density,
        "effect_fold_creases": args.fold_creases,
        "effect_fold_creases_intensity": args.fold_creases_intensity,
        "effect_ink_bleed": args.ink_bleed,
        "effect_ink_bleed_intensity": args.ink_bleed_intensity,
        "effect_ink_bleed_radius": args.ink_bleed_radius,
        "effect_shadow": args.shadow,
        "effect_shadow_intensity": args.shadow_intensity,
        "effect_shadows_angle": args.shadow_angle,
        "effect_strain": args.strain,
    }

    results = dataset_generator.generate_dataset(
        df, args.text_column, params, args.output_dir, image_dir, max_samples=args.max_samples
    )

    if not results:
        print("No images were generated (empty source or all rows failed to render).")
        sys.exit(1)

    csv_path = os.path.join(args.output_dir, args.output_csv)
    additional_info = {
        "source": args.input_csv or args.dataset_name,
        "text_column": args.text_column,
        "rows_processed": len(results),
    }
    dataset_generator.save_results_csv(
        results, additional_info, csv_path=csv_path, output_dir=args.output_dir
    )

    print(f"Generated {len(results)} images in '{image_dir}', metadata at '{csv_path}'.")

    if args.push_to_hub:
        if not args.hub_repo_id:
            print("--push-to-hub requires --hub-repo-id (e.g. 'username/my-synthetic-dataset').")
            sys.exit(1)
        huggingface_processor.push_output_to_hub(
            args.output_dir, args.hub_repo_id, private=args.hub_private
        )
        print(f"Uploaded '{args.output_dir}' to https://huggingface.co/datasets/{args.hub_repo_id}")


if __name__ == "__main__":
    main()
