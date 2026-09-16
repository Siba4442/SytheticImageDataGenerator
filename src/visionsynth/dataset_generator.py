import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import background_generator, effects_generator, text_renderer


def load_dataset(file_path: str, text_column: str) -> pd.DataFrame | None:
    """Load dataset from CSV file and validate text column"""
    try:
        encodings = ["utf-8", "iso-8859-1", "windows-1252", "utf-16"]
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise Exception("Could not load dataset with any supported encoding")

        if text_column not in df.columns:
            raise Exception(
                f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
            )

        df = df.dropna(subset=[text_column])
        df = df[df[text_column].astype(str).str.strip() != ""]

        return df

    except Exception as e:
        print(f"Failed to load '{file_path}': {e}")
        return None


def _build_background(params: dict) -> tuple[Image.Image, str]:
    width, height = params["width"], params["height"]
    background = params.get("background", "plain")

    if background == "old":
        return background_generator.old_paper(height, width), "old"

    if background == "birch":
        birch_params = {
            "texture": params.get("birch_texture", 1.0),
            "spots": params.get("birch_spots"),
            "min_radius": params.get("birch_min_radius", 10),
            "max_radius": params.get("birch_max_radius", 25),
            "intensity": params.get("birch_intensity", 10),
            "spot_sign": params.get("birch_spot_sign", -1),
            "irregularity": params.get("birch_irregularity", 0.35),
            "blur": params.get("birch_blur", 1.2),
            "cap_spots": params.get("birch_cap_spots", 2000),
        }
        return background_generator.birch(height, width, birch_params), "birch"

    if background == "parchment":
        parchment_params = {
            "texture": params.get("parchment_texture", 1.0),
            "spots": params.get("parchment_spots"),
            "min_radius": params.get("parchment_min_radius", 3),
            "max_radius": params.get("parchment_max_radius", 12),
            "intensity": params.get("parchment_intensity", 7),
            "spot_sign": params.get("parchment_spot_sign", -1),
            "irregularity": params.get("parchment_irregularity", 0.35),
            "blur_sigma": params.get("parchment_blur", 1.2),
            "cap_spots": params.get("parchment_cap_spots", 2000),
        }
        return background_generator.parchment(height, width, parchment_params), "parchment"

    if background == "lined":
        return background_generator.lined_paper(height, width), "lined"

    if background == "gaussian":
        return background_generator.gaussian_noise(height, width), "gaussian"

    if background == "image":
        image_dir = params.get("background_image_dir")
        if not image_dir:
            raise ValueError("background=image requires background_image_dir to be set")
        return background_generator.image(height, width, image_dir), "image"

    return background_generator.plain_background(height, width), "plain"


def render_and_save_image(text: str, params: dict) -> tuple[Image.Image | None, str]:
    """Render one synthetic document image for `text` following `params`.

    Returns (PIL.Image, style_name).
    """
    bg, style = _build_background(params)

    img = text_renderer.render_text(
        text,
        font_path=params["font"],
        width=params["width"],
        height=params["height"],
        font_size=params.get("font_size", 28),
        img=bg.convert("RGB"),
    )
    if img is None:
        return None, style

    arr = np.array(img.convert("RGB"))

    blur_level = params.get("blur_level", 0)
    if params.get("random_blur_level") and blur_level > 0:
        blur_level = int(np.random.randint(0, blur_level + 1))
    if blur_level > 0:
        k = blur_level * 2 + 1
        arr = cv2.GaussianBlur(arr, (k, k), 0)

    if params.get("effect_fiber"):
        fiber = effects_generator.simulate_paper_fiber_texture(
            params["width"], params["height"], params.get("effect_fiber_density", 0.2)
        )
        arr = np.clip(arr.astype(np.float32) - fiber.astype(np.float32) * 0.5, 0, 255).astype(
            np.uint8
        )

    if params.get("effect_fold_creases"):
        fold_lines = effects_generator.generate_random_fold_lines(
            (params["width"], params["height"])
        )
        fold_intensity = params.get("effect_fold_creases_intensity", 20) / 100.0
        arr = effects_generator.simulate_fold_crease(arr, fold_lines, fold_intensity)

    if params.get("effect_ink_bleed"):
        arr = effects_generator.simulate_ink_bleed(
            arr,
            params.get("effect_ink_bleed_intensity", 0.3),
            params.get("effect_ink_bleed_radius", 3),
        )

    if params.get("effect_shadow"):
        arr = effects_generator.apply_shadow_effects(
            arr,
            shadow_angle=params.get("effect_shadows_angle", 45),
            shadow_intensity=params.get("effect_shadow_intensity", 0.4),
        )

    if params.get("effect_strain"):
        strain_arr = np.array(
            background_generator.strain(params["height"], params["width"]).convert("RGB")
        )
        arr = np.clip(
            arr.astype(np.float32) * (strain_arr.astype(np.float32) / 255.0), 0, 255
        ).astype(np.uint8)

    return Image.fromarray(arr), style


def generate_dataset(
    dataset_df: pd.DataFrame,
    text_column: str,
    params: dict,
    output_dir: str,
    image_dir: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Render an image for each row's text and save it under image_dir.

    Returns row metadata for the CSV.
    """
    results = []

    if max_samples and max_samples < len(dataset_df):
        dataset_df = dataset_df.head(max_samples)

    extension = params.get("extension", ".jpg")

    for idx, row in dataset_df.iterrows():
        text = str(row[text_column]).strip()
        if not text:
            continue

        try:
            img, style = render_and_save_image(text, params)
        except Exception as e:
            print(f"Row {idx}: render failed: {e}")
            continue

        if img is None:
            continue

        image_filename = f"text_image_{idx:06d}{extension}"
        image_path = os.path.join(image_dir, image_filename)
        img.save(image_path)

        result = {
            "image_path": os.path.relpath(image_path, output_dir),
            "text": text,
            "style": style,
            "image_filename": image_filename,
        }
        for col in dataset_df.columns:
            if col != text_column:
                result[col] = row[col]

        results.append(result)

    return results


def save_results_csv(
    results: list[dict],
    additional_info: dict | None = None,
    csv_path: str = "dataset.csv",
    output_dir: str = "output",
) -> None:
    """Save results to CSV file, plus a metadata.txt sidecar if additional_info is given"""
    if not results:
        return

    df = pd.DataFrame(results)

    important_cols = [
        c for c in ["image_path", "text", "style", "image_filename"] if c in df.columns
    ]
    other_cols = [c for c in df.columns if c not in important_cols]
    df = df[important_cols + other_cols]

    df.to_csv(csv_path, index=False, encoding="utf-8")

    if additional_info:
        metadata_path = os.path.join(output_dir, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("Dataset Processing Metadata\n")
            f.write("=" * 30 + "\n")
            for key, value in additional_info.items():
                f.write(f"{key}: {value}\n")
