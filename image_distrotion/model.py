#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Synthetic Document Image Generator

This script generates highly realistic document images with text, applying a
wide variety of advanced augmentations to simulate age, wear, material types,
and physical distortions. It is designed to be a flexible command-line tool
for creating rich datasets for OCR and document analysis.

Key Features:
- Four procedural background styles: Lined, Old Paper, Birch, Parchment.
- Option to use real images as backgrounds.
- Advanced effects: Paper folds, ink bleed, perspective warp, shadows, and more.
- Geometric distortions: Cylindrical and washboard page warping.
- Fine-grained control over which effects to apply via command line.
"""

import os
import cv2
import math
import random
import argparse
import logging
from math import pi
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

# Attempt to import the 'noise' library for Perlin noise effects.
try:
    from noise import pnoise2
    NOISE_LIB_AVAILABLE = True
except ImportError:
    NOISE_LIB_AVAILABLE = False
    print("Warning: 'noise' library not found. Paper fiber texture will be disabled.")
    print("Install it with: pip install noise")


# --- Configuration ---

# Configure logging for clear feedback during the generation process.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A consolidated dictionary of all default parameters, combining both script versions.
DEFAULT_PARAMS = {
    # Basic options
    'width': 400,
    'height': 320,
    'base_images': 1,

    # Font and file path options
    'font_dir': '.',  # Default to the current directory for portability
    'font': 'NotoSansOriya_ExtraCondensed-Regular.ttf',
    'image_dir': '', # Optional: Directory to sample real background images from

    # Generation-level augmentations
    'noise': 0.7,
    'aging': 0.6,
    'texture': 0.7,
    'stains': 0.6,
    'stain_intensity': 0.5,
    'fiber_density': 0.5, # Controls intensity of Perlin noise paper fibers

    # Word-level options
    'word_position': 0.6,
    'ink_color': 0.5,
    'line_spacing': 0.4,
    'baseline': 0.3,
    'word_angle': 0.0,

    # Post-processing toggles and parameters
    'apply_transforms': True,
    'rotation_max': 5.0,
    'brightness_var': 0.2,
    'contrast_var': 0.2,
    'noise_min': 0.01,
    'noise_max': 0.05,
    'blur_min': 0.5,
    'blur_max': 1.0,

    # Advanced effect parameters
    'fold_intensity': 0.3,
    'bleed_intensity': 0.3,
    'bleed_radius': 3,
    'corner_displacement': 20, # For perspective warp
    'shadow_angle': 45,
    'shadow_intensity': 0.4,

    # Debugging
    'debug_mode': False,
}

# --- Advanced Effect Implementation ---

class AdvancedImageEffects:
    """A collection of static methods for simulating realistic document effects."""

    @staticmethod
    def simulate_paper_fiber_texture(width: int, height: int, fiber_density: float = 0.5) -> np.ndarray:
        """Creates a realistic paper fiber texture using multi-layered Perlin noise."""
        if not NOISE_LIB_AVAILABLE:
            # Fallback to simple random noise if the 'noise' library isn't installed.
            return np.random.randint(0, int(15 * fiber_density), (height, width, 3), dtype=np.uint8)
        try:
            # Generate two layers of Perlin noise for a more complex texture.
            base_texture = np.zeros((height, width))
            fine_texture = np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    base_texture[i][j] = pnoise2(i * 0.02, j * 0.02, octaves=4)
                    fine_texture[i][j] = pnoise2(i * 0.1, j * 0.1, octaves=2)
            
            # Combine the noise layers and scale by the fiber density.
            combined = base_texture * 0.7 + fine_texture * 0.3
            combined = ((combined + 1) / 2) * fiber_density * 20 # Normalize and scale
            texture = np.stack([combined] * 3, axis=2) # Convert to 3-channel image
            return texture.astype(np.uint8)
        except Exception as e:
            logger.warning(f"Failed to generate Perlin noise texture, falling back to simple noise: {e}")
            return np.random.randint(0, int(20 * fiber_density), (height, width, 3), dtype=np.uint8)

    @staticmethod
    def simulate_fold_crease(image: np.ndarray, fold_intensity: float = 0.5) -> np.ndarray:
        """Simulates paper folds and creases with realistic shadowing."""
        try:
            height, width = image.shape[:2]
            result = image.copy().astype(np.float32)
            
            # Create a random line to represent the fold
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            
            y_coords, x_coords = np.ogrid[:height, :width]
            line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
            if line_length_sq == 0: return image
            
            # Calculate the distance of each pixel from the fold line
            distances = np.abs((y2 - y1) * x_coords - (x2 - x1) * y_coords + x2 * y1 - y2 * x1) / np.sqrt(line_length_sq)
            fold_width = min(width, height) * 0.1 # The crease affects a certain width
            fold_profile = np.exp(-0.5 * (distances / fold_width)**2) # Gaussian falloff
            fold_effect = fold_profile * fold_intensity * 40
            
            # Add a subtle shadow on one side of the fold for realism
            shadow_mask = (y_coords - y1) * (x2 - x1) - (x_coords - x1) * (y2 - y1) > 0
            shadow_effect = fold_profile * shadow_mask * fold_intensity * 20
            
            for c in range(3):
                result[:, :, c] -= (fold_effect + shadow_effect)
            
            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in fold/crease simulation: {e}")
            return image

    @staticmethod
    def simulate_ink_bleed(image: np.ndarray, bleed_intensity: float = 0.3, bleed_radius: int = 3) -> np.ndarray:
        """Simulates ink bleeding into the paper using morphological dilation."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Invert the threshold to get a mask of the dark text areas
            _, text_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            kernel_size = max(1, int(bleed_radius * 2 + 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            # Dilate the text mask to simulate the ink spreading
            bleeding_mask = cv2.dilate(text_mask, kernel, iterations=1)
            
            # Blur the mask to create a soft falloff effect
            bleed_effect = cv2.GaussianBlur(bleeding_mask.astype(np.float32), (kernel_size, kernel_size), 0)
            bleed_effect = (bleed_effect / 255.0) * bleed_intensity
            
            result = image.copy().astype(np.float32)
            for c in range(3):
                # Blend the original image with a darkened version based on the bleed mask
                result[:, :, c] = result[:, :, c] * (1 - bleed_effect) + (result[:, :, c] * 0.7) * bleed_effect
            
            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in ink bleed simulation: {e}")
            return image

    @staticmethod
    def apply_perspective_distortion(image: np.ndarray, corner_displacement: int = 20) -> np.ndarray:
        """Applies a random perspective warp to simulate a non-flat view."""
        try:
            height, width = image.shape[:2]
            src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            
            # Create destination points by randomly displacing the corners
            dst_points = src_points.copy()
            for i in range(4):
                dst_points[i][0] += random.randint(-corner_displacement, corner_displacement)
                dst_points[i][1] += random.randint(-corner_displacement, corner_displacement)
            
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            logger.error(f"Error in perspective distortion: {e}")
            return image

# --- Page Warping Helpers ---

def cylindrical_edge_warp(pil_img: Image.Image, strength=0.6, warp_portion=0.45) -> Image.Image:
    """Applies a cylindrical bend to one side of the page."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    side = random.choice(["left", "right"])
    
    W = int(warp_portion * w) # width of the curved strip
    R = W / strength if strength != 0 else 1e9 # radius of cylinder

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    map_x, map_y = X.astype(np.float32), Y.astype(np.float32)

    if side == "left":
        strip = X < W
        dx = W - X[strip]
    else: # right
        strip = X > (w - W)
        dx = X[strip] - (w - W)

    theta = dx / R
    map_x[strip] += (R * np.sin(theta) - dx)
    map_y[strip] = (Y[strip] - h/2) / np.cos(theta) + h/2

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

def washboard_warp(pil_img: Image.Image, amplitude=8, wavelength=120) -> Image.Image:
    """Creates vertical sine ripples that run horizontally across the page."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    x = np.arange(w, dtype=np.float32)
    dy = amplitude * np.sin(2 * pi * x / wavelength + random.uniform(0, 2*pi))
    
    # Attenuate the effect towards the bottom of the page for realism
    atten = np.linspace(1, 0.2, h, dtype=np.float32)[:, None]
    
    map_x, map_y = np.meshgrid(x, np.arange(h, dtype=np.float32))
    map_y += dy * atten

    warped = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


# --- Core Generation Logic ---

def create_background(width: int, height: int, style: str, params: Dict) -> Image.Image:
    """Generates a textured background image based on the chosen style."""
    if params.get("image_dir") and os.path.isdir(params["image_dir"]):
        image_files = [f for f in os.listdir(params["image_dir"]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            img_path = os.path.join(params["image_dir"], random.choice(image_files))
            try:
                bg_img = Image.open(img_path).convert('RGB').resize((width, height), Image.LANCZOS)
                logger.info(f"Using background image: {img_path}")
                return bg_img
            except Exception as e:
                logger.error(f"Could not load background image {img_path}, falling back to procedural. Error: {e}")

    # Generate a base procedural background
    fiber_texture = AdvancedImageEffects.simulate_paper_fiber_texture(width, height, params['fiber_density'])
    
    color_map = {"old_paper": [236, 222, 181], "birch": [235, 225, 215], "parchment": [230, 215, 185], "lined_paper": [210, 180, 140]}
    background = np.full((height, width, 3), color_map.get(style, [230, 215, 185]), dtype=np.uint8)
    
    # Blend the fiber texture with the base color
    background = np.clip(background.astype(np.float32) + fiber_texture, 0, 255).astype(np.uint8)

    # Add style-specific features (e.g., lines for lined paper)
    if style == "lined_paper":
        for y in range(0, height, random.randint(15, 25)):
            darkness = random.randint(6, 20) * params["texture"]
            background[y:y+2, :, :] = np.clip(background[y:y+2, :, :] - darkness, 0, 255)

    # Apply general aging and noise
    noise_val = np.random.randint(0, int(12 * params["noise"]), (height, width, 3), dtype=np.uint8)
    background = np.clip(background.astype(np.int16) - noise_val, 0, 255).astype(np.uint8)

    return Image.fromarray(background)

def render_text(text: str, font_path: str, width: int, height: int, style: str, params: Dict) -> Optional[Image.Image]:
    """Renders text onto a generated background with various word-level augmentations."""
    try:
        img = create_background(width, height, style, params)
        draw = ImageDraw.Draw(img)
        font_size = random.randint(12, 18)
        font = ImageFont.truetype(font_path, font_size)

        ink_color_map = {"old_paper": (20, 20, 20), "birch": (50, 20, 10), "parchment": (10, 10, 10), "lined_paper": (60, 30, 10)}
        base_ink_color = ink_color_map.get(style)

        words = text.strip().replace('\n', ' ').split()
        y_pos, margin = 25, 25
        available_width = width - 2 * margin
        space_width = draw.textlength(" ", font=font)

        # Simple word wrapping
        lines, current_line, current_width = [], [], 0
        for word in words:
            word_width = draw.textlength(word, font=font)
            if current_line and current_width + space_width + word_width > available_width:
                lines.append(" ".join(current_line))
                current_line, current_width = [word], word_width
            else:
                current_line.append(word)
                current_width += word_width + space_width
        if current_line: lines.append(" ".join(current_line))

        # Render each line with augmentations
        for line in lines:
            if y_pos + font_size > height - margin: break
            
            line_width = draw.textlength(line, font=font)
            x_pos = (width - line_width) // 2
            
            # Apply baseline wobble and color variation
            y_line_pos = y_pos + (random.randint(-2, 2) * params["baseline"])
            color_var = int(random.randint(-3, 3) * params["ink_color"])
            ink_color = tuple(np.clip(c + color_var, 0, 255) for c in base_ink_color)
            
            draw.text((x_pos, y_line_pos), line, font=font, fill=ink_color)
            
            y_pos += int(font_size * 1.2 * (1.0 + random.uniform(-0.1, 0.1) * params["line_spacing"]))
        
        return img

    except FileNotFoundError:
        logger.error(f"Font file not found at '{font_path}'. Please check the path.")
        return None
    except Exception as e:
        logger.error(f"An error occurred during text rendering: {e}", exc_info=params['debug_mode'])
        return None

def apply_postprocessing(image: Image.Image, effects_to_apply: List[str], params: Dict) -> Image.Image:
    """Applies a specific list of post-processing effects to an image."""
    if not image: return None
    
    # A dictionary mapping effect names to their corresponding functions
    all_transforms = {
        "rotate": lambda img: img.rotate(random.uniform(-params["rotation_max"], params["rotation_max"]), resample=Image.BICUBIC, expand=False),
        "brightness": lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(1.0 - params["brightness_var"], 1.0 + params["brightness_var"])),
        "contrast": lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(1.0 - params["contrast_var"], 1.0 + params["contrast_var"])),
        "blur": lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(params["blur_min"], params["blur_max"]))),
        "fold": lambda img: Image.fromarray(AdvancedImageEffects.simulate_fold_crease(np.array(img), params['fold_intensity'])),
        "bleed": lambda img: Image.fromarray(AdvancedImageEffects.simulate_ink_bleed(np.array(img), params['bleed_intensity'], params['bleed_radius'])),
        "perspective": lambda img: Image.fromarray(AdvancedImageEffects.apply_perspective_distortion(np.array(img), params['corner_displacement'])),
        "washboard": lambda img: washboard_warp(img),
        "cylinder": lambda img: cylindrical_edge_warp(img),
    }

    current_image = image.copy()
    logger.info(f"Applying effects: {', '.join(effects_to_apply)}")
    
    for effect_name in effects_to_apply:
        if effect_name in all_transforms:
            try:
                current_image = all_transforms[effect_name](current_image)
            except Exception as e:
                logger.error(f"Failed to apply '{effect_name}' effect: {e}", exc_info=params['debug_mode'])
        else:
            logger.warning(f"Effect '{effect_name}' not recognized and will be skipped.")
            
    return current_image

# --- Main Execution ---

def main():
    """Parses command-line arguments and runs the image generation process."""
    
    # Define all possible effects for the help message and choices
    available_effects = ["rotate", "brightness", "contrast", "blur", "fold", "bleed", "perspective", "washboard", "cylinder"]
    
    parser = argparse.ArgumentParser(
        description='Comprehensive Synthetic Document Image Generator.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Add arguments dynamically from the DEFAULT_PARAMS dictionary ---
    for key, value in DEFAULT_PARAMS.items():
        arg_name = f'--{key.replace("_", "-")}'
        if isinstance(value, bool):
            parser.add_argument(arg_name, action='store_true', default=value, help=f'Enable {key}.')
        else:
            parser.add_argument(arg_name, type=type(value), default=value, help=f'Set {key}.')
    
    # --- Add specific control arguments ---
    parser.add_argument('--output-dir', type=str, default='output_images', help='Directory to save generated images.')
    parser.add_argument('--text-file', type=str, default=None, help='Path to a UTF-8 text file to use as input.')
    parser.add_argument('--effects', nargs='*', choices=available_effects, default=["rotate", "blur", "perspective"],
                        help='A space-separated list of effects to apply. Choose from: ' + ', '.join(available_effects))

    args = parser.parse_args()
    params = vars(args)

    if params.get('debug_mode'):
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Load Text ---
    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                sanskrit_text = f.read()
        except FileNotFoundError:
            logger.error(f"Text file not found: {args.text_file}. Exiting.")
            return
    else:
        sanskrit_text = """
        କବି ସମ୍ରାଟ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ (୧୬୭୦ - ୧୭୪୦, ଅନ୍ୟମତେ ୧୭୨୦) ସପ୍ତଦଶ ଶତାବ୍ଦୀର ପ୍ରମୁଖ ଓଡ଼ିଶୀ ସଙ୍ଗୀତକାର କବି ଥିଲେ । ସେ ସପ୍ତଦଶ ଶତାବ୍ଦୀର ଶେଷ ପର୍ଯ୍ୟାୟରେ ପୁରାତନ ରାଜଶାସିତ ଘୁମୁସରର ରାଜବଂଶରେ କୁଲାଡ଼ଗଡଜନ୍ମଲାଭ କରିଥିଲେ।
        """
        logger.info("Using default sample text as no --text-file was provided.")

    # --- Validate Font Path ---
    font_path = os.path.join(params['font_dir'], params['font'])
    if not os.path.exists(font_path):
        logger.error(f"Font not found at '{font_path}'. Please provide the correct path using --font-dir and --font.")
        logger.error("You can download 'NotoSansOriya' from Google Fonts or other sources.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Starting generation of {params['base_images']} image(s)...")
    
    # --- Generation Loop ---
    for i in range(params['base_images']):
        style = random.choice(["lined_paper", "old_paper", "birch", "parchment"])
        logger.info(f"--- Image {i+1}/{params['base_images']} | Style: {style} ---")
        
        # 1. Render the base image with text
        base_img = render_text(sanskrit_text, font_path, params['width'], params['height'], style, params)
        
        if not base_img:
            logger.error("Failed to render base image. Skipping...")
            continue
            
        # 2. Apply the selected post-processing effects
        if params['apply_transforms']:
            final_img = apply_postprocessing(base_img, args.effects, params)
        else:
            final_img = base_img
            logger.info("Post-processing transforms are disabled.")

        # 3. Save the final image
        try:
            effects_str = '_'.join(args.effects) if params['apply_transforms'] else 'base'
            filename = f"doc_{style}_{i+1}_{effects_str}.png"
            output_path = os.path.join(args.output_dir, filename)
            final_img.save(output_path)
            logger.info(f"Successfully saved final image to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save the final image: {e}", exc_info=params['debug_mode'])

    logger.info("Generation process complete.")

if __name__ == "__main__":
    main()
