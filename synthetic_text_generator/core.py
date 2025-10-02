"""
Core module containing main generation functions
"""

import os
import random
import logging
from typing import Dict, List, Optional
from PIL import Image
from .config import ENHANCED_DEFAULT_PARAMS
from .text_renderer import render_enhanced_sanskrit
from .transformations import (apply_enhanced_postprocessing, create_comprehensive_effect_combinations, 
                             apply_systematic_postprocessing)

logger = logging.getLogger(__name__)


def generate_enhanced_sanskrit_samples(text: str, font_path: str = None,
                                     output_dir: str = None, params: Dict = None) -> Optional[List[Image.Image]]:
    """Enhanced main generation function with all advanced features"""
    # Use enhanced default params if none provided
    if params is None:
        params = ENHANCED_DEFAULT_PARAMS.copy()
    else:
        params = {**ENHANCED_DEFAULT_PARAMS, **params}

    # Set default font path if not provided
    if font_path is None:
        font_path = os.path.join(params['font_dir'], params['font'])

    if not os.path.exists(font_path):
        logger.error(f"Font not found at {font_path}")
        return [] if output_dir is None else None

    styles = ["lined_paper", "old_paper", "birch", "parchment"]

    ink_colors = {
        "lined_paper": (60, 30, 10),
        "old_paper": (20, 20, 20),
        "birch": (50, 20, 10),
        "parchment": (10, 10, 10)
    }

    width, height = params['width'], params['height']
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Randomly sample styles for the total number of base images
    sampled_styles = random.choices(styles, k=params['base_images'])
    style_counts = {style: sampled_styles.count(style) for style in styles}
    logger.info(f"Randomly selected styles: {style_counts}")

    base_images = []
    image_counter = 0

    # Generate randomly sampled base images
    for style, count in style_counts.items():
        for i in range(count):
            image_counter += 1

            # Randomly select a font size between 12 and 18
            font_size = random.randint(12, 18)
            logger.info(f"Using font size {font_size} for {style}_{i+1}")

            # If output_dir is provided, save to file, otherwise just render
            output_path = os.path.join(output_dir, f"enhanced_sanskrit_{style}_{i+1}.png") if output_dir else None

            img = render_enhanced_sanskrit(
                text=text,
                font_path=font_path,
                output_path=output_path,
                width=width,
                height=height,
                font_size=font_size,
                style=style,
                ink_color=ink_colors[style],
                params=params
            )

            if img:
                base_images.append(img)

                if params['apply_transforms'] and output_dir:
                    base_filename = f"enhanced_sanskrit_{style}_{i+1}"
                    transformed_images = apply_enhanced_postprocessing(img, output_dir, base_filename, params)
                    base_images.extend(transformed_images[1:])  # Skip the original which is already added

    return base_images if output_dir is None else None


def generate_comprehensive_dataset(text: str, font_path: str = None,
                                 output_dir: str = None, params: Dict = None) -> List[Image.Image]:
    """Generate a comprehensive dataset using all effects systematically"""

    if params is None:
        params = ENHANCED_DEFAULT_PARAMS.copy()
    else:
        params = {**ENHANCED_DEFAULT_PARAMS, **params}

    if font_path is None:
        font_path = os.path.join(params['font_dir'], params['font'])

    if not os.path.exists(font_path):
        logger.error(f"Font not found at {font_path}")
        return []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate effect combinations
    effect_combinations = create_comprehensive_effect_combinations()

    styles = ["lined_paper", "old_paper", "birch", "parchment"]
    ink_colors = {
        "lined_paper": (60, 30, 10),
        "old_paper": (20, 20, 20),
        "birch": (50, 20, 10),
        "parchment": (10, 10, 10)
    }

    width, height = params['width'], params['height']
    all_generated_images = []

    logger.info(f"Generating comprehensive dataset with {len(effect_combinations)} effect combinations")

    # Generate base images for each style
    for style_idx, style in enumerate(styles):
        logger.info(f"Processing style: {style}")

        # Generate base image
        font_size = random.randint(14, 18)
        output_path = os.path.join(output_dir, f"base_{style}.png") if output_dir else None

        base_image = render_enhanced_sanskrit(
            text=text,
            font_path=font_path,
            output_path=output_path,
            width=width,
            height=height,
            font_size=font_size,
            style=style,
            ink_color=ink_colors[style],
            params=params
        )

        if base_image:
            all_generated_images.append(base_image)

            # Apply each effect combination to this base image
            for combo_idx, effect_combo in enumerate(effect_combinations):
                base_filename = f"comprehensive_{style}_{combo_idx:03d}"

                enhanced_images = apply_systematic_postprocessing(
                    base_image, output_dir, base_filename, params, effect_combo
                )

                # Add only the enhanced images (skip the original)
                all_generated_images.extend(enhanced_images[1:])

                logger.info(f"Generated {len(enhanced_images)} images for {style} with effects: {effect_combo}")

    logger.info(f"Total images generated: {len(all_generated_images)}")
    return all_generated_images


def generate_ultra_realistic_samples(text: str, output_dir: str = None,
                                   style_focus: str = None, params: Dict = None) -> List[Image.Image]:
    """Generate ultra-realistic samples with maximum effect application"""

    if params is None:
        params = ENHANCED_DEFAULT_PARAMS.copy()

    # Override parameters for maximum realism
    ultra_realistic_params = {
        **params,
        'fold_intensity': 0.4,
        'bleed_intensity': 0.35,
        'shadow_intensity': 0.5,
        'lens_distortion_strength': 0.15,
        'aging_intensity': 0.7,
        'fiber_density': 0.6,
        'texture': 0.8,
        'noise': 0.6,
        'stains': 0.7,
        'stain_intensity': 0.6
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define ultra-realistic effect combinations
    ultra_combinations = [
        # Historical document simulation
        ["fold_crease", "ink_bleed", "shadow_cast", "scanner_artifacts"],

        # Aged manuscript simulation
        ["perspective", "morphological", "lens_distortion", "washboard"],

        # Scanner/photography simulation
        ["cylinder", "scanner_artifacts", "lens_distortion", "shadow_cast"],

        # Weather-damaged document
        ["fold_crease", "ink_bleed", "morphological", "perspective"],

        # Complete realism (all effects)
        ["fold_crease", "ink_bleed", "perspective", "shadow_cast", "morphological", "scanner_artifacts", "lens_distortion"],

        # Photographic realism
        ["perspective", "lens_distortion", "shadow_cast", "cylinder"],

        # Manuscript preservation
        ["washboard", "ink_bleed", "morphological", "fold_crease"]
    ]

    font_path = os.path.join(ultra_realistic_params['font_dir'], ultra_realistic_params['font'])

    styles = ["lined_paper", "old_paper", "birch", "parchment"] if not style_focus else [style_focus]
    ink_colors = {
        "lined_paper": (60, 30, 10),
        "old_paper": (20, 20, 20),
        "birch": (50, 20, 10),
        "parchment": (10, 10, 10)
    }

    all_images = []

    logger.info(f"Generating ultra-realistic samples with {len(ultra_combinations)} combinations")

    for style in styles:
        base_image = render_enhanced_sanskrit(
            text=text,
            font_path=font_path,
            output_path=None,
            width=ultra_realistic_params['width'],
            height=ultra_realistic_params['height'],
            font_size=random.randint(14, 18),
            style=style,
            ink_color=ink_colors[style],
            params=ultra_realistic_params
        )

        if base_image:
            for combo_idx, effect_combo in enumerate(ultra_combinations):
                base_filename = f"ultra_realistic_{style}_{combo_idx:02d}"

                enhanced_images = apply_systematic_postprocessing(
                    base_image, output_dir, base_filename, ultra_realistic_params, effect_combo
                )

                all_images.extend(enhanced_images[1:])  # Skip original

                logger.info(f"Generated ultra-realistic sample: {base_filename}")

    return all_images
