"""
Transformations module for geometric transformations and post-processing effects
"""

import cv2
import os
import random
import logging
import itertools
from math import pi
from typing import List, Dict, Tuple, Callable
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from .effects import AdvancedImageEffects, generate_random_fold_lines, safe_apply_effect

logger = logging.getLogger(__name__)


def cylindrical_edge_warp(pil_img: Image.Image, side: str = "left",
                         strength: float = 0.6, warp_portion: float = 0.45) -> Image.Image:
    """Enhanced cylindrical edge warp with better error handling"""
    try:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # width of the curved strip
        W = int(warp_portion * w)
        # fake focal length = radius of cylinder
        R = W / strength if strength != 0 else 1e9

        # Build meshgrid of pixel coords
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = X.astype(np.float32).copy()
        map_y = Y.astype(np.float32).copy()

        if side == "left":
            strip = X < W
            dx = W - X[strip]
        else:  # right
            strip = X > (w - W)
            dx = X[strip] - (w - W)

        # angle on cylinder surface for those pixels
        theta = dx / R
        # horizontal mapping
        displacement = R * np.sin(theta) - dx
        map_x[strip] += displacement

        # vertical scaling
        scale_y = np.cos(theta)
        map_y[strip] = (Y[strip] - h/2) / scale_y + h/2

        warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logger.error(f"Error in cylindrical warp: {e}")
        return pil_img


def washboard_warp(pil_img: Image.Image, amplitude: float = 8, wavelength: float = 120,
                  phase: float = 0.0, decay_from_top: bool = True) -> Image.Image:
    """Enhanced washboard warp with better error handling"""
    try:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # build a vector of vertical offsets
        x = np.arange(w, dtype=np.float32)
        dy = amplitude * np.sin(2*pi*x / wavelength + phase)

        if decay_from_top:
            atten = np.linspace(1, 0.2, h, dtype=np.float32)[:, None]
        else:
            atten = 1.0

        # broadcast to full map
        map_x, map_y = np.meshgrid(x, np.arange(h, dtype=np.float32))
        map_y += dy * atten

        warped = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logger.error(f"Error in washboard warp: {e}")
        return pil_img


def apply_enhanced_postprocessing(original_image: Image.Image, output_dir: str,
                                base_filename: str, params: Dict) -> List[Image.Image]:
    """Enhanced post-processing with all advanced effects"""
    all_images = [original_image]
    transforms = []

    def rotate_image(img, angle):
        bg_color = tuple(np.array(img).mean(axis=(0, 1)).astype(int))
        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg_color)
        return rotated

    def adjust_brightness(img, factor):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def adjust_contrast(img, factor):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def add_noise(img, intensity):
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    def blur_image(img, radius):
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Original transforms
    transforms.append(("rotate", lambda img: rotate_image(img,
                                    random.uniform(-params["rotation_max"], params["rotation_max"]))))
    transforms.append(("brightness", lambda img: adjust_brightness(img,
                                    random.uniform(1.0-params["brightness_var"], 1.0+params["brightness_var"]))))
    transforms.append(("contrast", lambda img: adjust_contrast(img,
                                    random.uniform(1.0-params["contrast_var"], 1.0+params["contrast_var"]))))
    transforms.append(("noise", lambda img: add_noise(img,
                                    random.uniform(params["noise_min"], params["noise_max"]))))
    transforms.append(("blur", lambda img: blur_image(img,
                                    random.uniform(params["blur_min"], params["blur_max"]))))

    # Existing geometric transforms
    transforms.append(("washboard", lambda img: washboard_warp(
        img,
        amplitude=random.uniform(6, 12),
        wavelength=random.uniform(90, 150),
        phase=random.uniform(0, 2*pi),
        decay_from_top=random.choice([True, False]))))

    transforms.append(("cylinder", lambda img: cylindrical_edge_warp(
        img,
        side=random.choice(["left", "right"]),
        strength=random.uniform(0.4, 0.8) * random.choice([1, -1]),
        warp_portion=random.uniform(0.35, 0.5))))

    # New advanced transforms
    if params.get('enable_advanced_effects', True):
        # Fold and crease effects
        if random.random() < params.get('fold_probability', 0.4):
            transforms.append(("fold_crease", lambda img: Image.fromarray(
                AdvancedImageEffects.simulate_fold_crease(
                    np.array(img),
                    generate_random_fold_lines(img.size),
                    params.get("fold_intensity", 0.3)
                )
            )))

        # Ink bleed effects
        if random.random() < params.get('advanced_effect_probability', 0.7):
            transforms.append(("ink_bleed", lambda img: Image.fromarray(
                AdvancedImageEffects.simulate_ink_bleed(
                    np.array(img),
                    params.get("bleed_intensity", 0.3),
                    params.get("bleed_radius", 3)
                )
            )))

        # Perspective distortion
        if random.random() < params.get('perspective_probability', 0.5):
            transforms.append(("perspective", lambda img: Image.fromarray(
                AdvancedImageEffects.apply_perspective_distortion(
                    np.array(img),
                    params.get("corner_displacement", 20)
                )
            )))

        # Shadow effects
        if random.random() < params.get('shadow_probability', 0.6):
            transforms.append(("shadow_cast", lambda img: Image.fromarray(
                AdvancedImageEffects.apply_shadow_effects(
                    np.array(img),
                    params.get("shadow_angle", 45),
                    params.get("shadow_intensity", 0.4)
                )
            )))

        # Morphological operations
        if random.random() < params.get('advanced_effect_probability', 0.7):
            transforms.append(("morphological", lambda img: Image.fromarray(
                AdvancedImageEffects.apply_morphological_operations(
                    np.array(img),
                    params.get("morph_operation", "mixed"),
                    params.get("morph_kernel_size", 3)
                )
            )))

        # Scanner artifacts
        if params.get('scanner_artifacts', True) and random.random() < 0.3:
            transforms.append(("scanner_artifacts", lambda img: Image.fromarray(
                AdvancedImageEffects.simulate_scanner_artifacts(
                    np.array(img),
                    params.get("compression_quality", 85)
                )
            )))

        # Lens distortion
        if random.random() < 0.3:
            transforms.append(("lens_distortion", lambda img: Image.fromarray(
                AdvancedImageEffects.apply_lens_distortion(
                    np.array(img),
                    params.get("lens_distortion_strength", 0.2)
                )
            )))

    # Select transforms to apply
    if params["all_transforms"]:
        selected_transforms = transforms
    else:
        n_transforms = random.randint(1, min(5, len(transforms)))
        selected_transforms = random.sample(transforms, n_transforms)

    # Apply transforms
    for transform_name, transform_func in selected_transforms:
        try:
            transformed_img = safe_apply_effect(transform_func, original_image, transform_name)
            if output_dir:
                transformed_filename = f"{base_filename}_{transform_name}.png"
                transformed_path = os.path.join(output_dir, transformed_filename)
                transformed_img.save(transformed_path)
                logger.info(f"Saved transformed image to {transformed_path}")
            all_images.append(transformed_img)
        except Exception as e:
            logger.error(f"Error applying transform {transform_name}: {e}")

    # Combined transformations
    if len(selected_transforms) > 1:
        try:
            combined_img = original_image.copy()
            for _, transform_func in selected_transforms:
                combined_img = safe_apply_effect(transform_func, combined_img, "combined")

            if output_dir:
                combined_filename = f"{base_filename}_combined.png"
                combined_path = os.path.join(output_dir, combined_filename)
                combined_img.save(combined_path)
                logger.info(f"Saved combined transformation to {combined_path}")
            all_images.append(combined_img)
        except Exception as e:
            logger.error(f"Error creating combined transformation: {e}")

    return all_images


def create_comprehensive_effect_combinations():
    """Create all possible effect combinations for maximum realism"""

    # Base effects (always applied)
    base_effects = [
        "rotate", "brightness", "contrast", "noise", "blur"
    ]

    # Geometric effects
    geometric_effects = [
        "washboard", "cylinder"
    ]

    # Advanced effects
    advanced_effects = [
        "fold_crease", "ink_bleed", "perspective", "shadow_cast",
        "morphological", "scanner_artifacts", "lens_distortion"
    ]

    # Create systematic combinations
    effect_combinations = []

    # 1. Individual effects (each effect alone)
    for effect in base_effects + geometric_effects + advanced_effects:
        effect_combinations.append([effect])

    # 2. Pairs of effects
    for combo in itertools.combinations(advanced_effects, 2):
        effect_combinations.append(list(combo))

    # 3. Geometric + Advanced combinations
    for geo in geometric_effects:
        for adv in advanced_effects:
            effect_combinations.append([geo, adv])

    # 4. Triple combinations (most realistic)
    for combo in itertools.combinations(advanced_effects, 3):
        effect_combinations.append(list(combo))

    # 5. Full combination sets (maximum realism)
    effect_combinations.append(advanced_effects[:4])  # First 4 advanced effects
    effect_combinations.append(advanced_effects[4:])  # Last 3 advanced effects
    effect_combinations.append(advanced_effects)      # All advanced effects

    return effect_combinations


def apply_systematic_postprocessing(original_image: Image.Image, output_dir: str,
                                  base_filename: str, params: Dict,
                                  effect_combination: List[str] = None) -> List[Image.Image]:
    """Apply effects systematically based on specified combination"""

    all_images = [original_image]

    # Define all possible transforms
    def rotate_image(img, angle):
        bg_color = tuple(np.array(img).mean(axis=(0, 1)).astype(int))
        return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg_color)

    def adjust_brightness(img, factor):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def adjust_contrast(img, factor):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def add_noise(img, intensity):
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    def blur_image(img, radius):
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Complete transform dictionary
    transforms = {
        # Base effects
        "rotate": lambda img: rotate_image(img, random.uniform(-params["rotation_max"], params["rotation_max"])),
        "brightness": lambda img: adjust_brightness(img, random.uniform(1.0-params["brightness_var"], 1.0+params["brightness_var"])),
        "contrast": lambda img: adjust_contrast(img, random.uniform(1.0-params["contrast_var"], 1.0+params["contrast_var"])),
        "noise": lambda img: add_noise(img, random.uniform(params["noise_min"], params["noise_max"])),
        "blur": lambda img: blur_image(img, random.uniform(params["blur_min"], params["blur_max"])),

        # Geometric effects
        "washboard": lambda img: washboard_warp(
            img, amplitude=random.uniform(6, 12), wavelength=random.uniform(90, 150),
            phase=random.uniform(0, 2*pi), decay_from_top=random.choice([True, False])
        ),
        "cylinder": lambda img: cylindrical_edge_warp(
            img, side=random.choice(["left", "right"]), strength=random.uniform(0.4, 0.8) * random.choice([1, -1]),
            warp_portion=random.uniform(0.35, 0.5)
        ),

        # Advanced effects
        "fold_crease": lambda img: Image.fromarray(
            AdvancedImageEffects.simulate_fold_crease(
                np.array(img), generate_random_fold_lines(img.size), params.get("fold_intensity", 0.3)
            )
        ),
        "ink_bleed": lambda img: Image.fromarray(
            AdvancedImageEffects.simulate_ink_bleed(
                np.array(img), params.get("bleed_intensity", 0.3), params.get("bleed_radius", 3)
            )
        ),
        "perspective": lambda img: Image.fromarray(
            AdvancedImageEffects.apply_perspective_distortion(
                np.array(img), params.get("corner_displacement", 20)
            )
        ),
        "shadow_cast": lambda img: Image.fromarray(
            AdvancedImageEffects.apply_shadow_effects(
                np.array(img), params.get("shadow_angle", 45), params.get("shadow_intensity", 0.4)
            )
        ),
        "morphological": lambda img: Image.fromarray(
            AdvancedImageEffects.apply_morphological_operations(
                np.array(img), params.get("morph_operation", "mixed"), params.get("morph_kernel_size", 3)
            )
        ),
        "scanner_artifacts": lambda img: Image.fromarray(
            AdvancedImageEffects.simulate_scanner_artifacts(
                np.array(img), params.get("compression_quality", 85)
            )
        ),
        "lens_distortion": lambda img: Image.fromarray(
            AdvancedImageEffects.apply_lens_distortion(
                np.array(img), params.get("lens_distortion_strength", 0.2)
            )
        )
    }

    # Apply base effects first (always applied for realism)
    current_image = original_image
    base_effects = ["rotate", "brightness", "contrast", "noise", "blur"]

    for effect_name in base_effects:
        if effect_name in transforms:
            current_image = safe_apply_effect(transforms[effect_name], current_image, effect_name)

    # Apply specified effect combination
    if effect_combination:
        for effect_name in effect_combination:
            if effect_name in transforms:
                current_image = safe_apply_effect(transforms[effect_name], current_image, effect_name)

        # Save the combined result
        if output_dir:
            combo_name = "_".join(effect_combination)
            filename = f"{base_filename}_{combo_name}.png"
            filepath = os.path.join(output_dir, filename)
            current_image.save(filepath)
            logger.info(f"Saved combination image: {filepath}")

    all_images.append(current_image)
    return all_images
