"""
Background generation module for creating realistic paper textures and backgrounds
"""

import os
import random
import numpy as np
from PIL import Image
from typing import Dict
import logging
from .effects import AdvancedImageEffects

logger = logging.getLogger(__name__)


def create_enhanced_background(width: int, height: int, style: str, params: Dict) -> Image.Image:
    """Enhanced background generation with advanced paper textures"""
    # Check if image directory is provided and use image background if available
    if params.get("image_dir") and os.path.exists(params["image_dir"]):
        image_files = [f for f in os.listdir(params["image_dir"])
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            img_path = os.path.join(params["image_dir"], random.choice(image_files))
            try:
                bg_img = Image.open(img_path).convert('RGB')
                bg_img = bg_img.resize((width, height), Image.LANCZOS)
                return bg_img
            except Exception as e:
                logger.error(f"Error loading background image {img_path}: {e}")

    # Generate paper fiber texture
    if params.get('fiber_density', 0) > 0:
        fiber_texture = AdvancedImageEffects.simulate_paper_fiber_texture(
            width, height, params['fiber_density']
        )
    else:
        fiber_texture = np.zeros((height, width, 3), dtype=np.uint8)

    if style == "lined_paper":
        background = np.ones((height, width, 3), dtype=np.uint8) * [210, 180, 140]

        # Add fiber texture
        background = np.clip(background.astype(np.float32) - fiber_texture, 0, 255).astype(np.uint8)

        # Enhanced line generation
        line_spacing = random.randint(15, 25)
        for y in range(0, height, line_spacing):
            line_width = random.randint(1, 2)
            darkness = random.randint(6, 20) * params["texture"]
            if y + line_width < height:
                background[y:y+line_width, :, :] = np.clip(background[y:y+line_width, :, :] - darkness, 0, 255)

        # Enhanced noise
        noise = np.random.randint(0, int(15 * params["noise"]), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)

        # Enhanced stains with realistic shapes
        stain_count = int(random.randint(2, 4) * params["stains"])
        for _ in range(stain_count):
            x = random.randint(0, width-100)
            y = random.randint(0, height-100)
            size = random.randint(20, 60)
            darkness = random.randint(8, 25) * params["stain_intensity"]

            # Create more realistic stain shapes
            stain_mask = np.zeros((size, size), dtype=np.float32)
            center = size // 2
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - center)**2 + (j - center)**2)
                    if dist < center:
                        stain_mask[i, j] = (1 - dist / center) * np.random.uniform(0.4, 1.0)

            # Apply stain
            end_y = min(y + size, height)
            end_x = min(x + size, width)
            actual_size_y = end_y - y
            actual_size_x = end_x - x

            if actual_size_y > 0 and actual_size_x > 0:
                stain_region = stain_mask[:actual_size_y, :actual_size_x]
                for c in range(3):
                    background[y:end_y, x:end_x, c] = np.clip(
                        background[y:end_y, x:end_x, c] - darkness * stain_region * params["stain_intensity"], 0, 255
                    )

    elif style == "old_paper":
        background = np.ones((height, width, 3), dtype=np.uint8) * [236, 222, 181]

        # Add fiber texture
        background = np.clip(background.astype(np.float32) - fiber_texture, 0, 255).astype(np.uint8)

        # Enhanced aging with more realistic patterns
        noise = np.random.randint(0, int(12 * params["noise"]), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)

        # More realistic edge aging
        edge_width = width // 10
        for i in range(edge_width):
            factor = (edge_width - i) / edge_width * 15 * params["aging"]
            # Apply non-uniform aging
            aging_noise = np.random.uniform(0.5, 1.5, (height, width))

            # Top and bottom edges
            if i < height:
                background[i, :, 2] = np.clip(background[i, :, 2] - factor * aging_noise[i, :], 0, 255)
            if height - i - 1 >= 0:
                background[height-i-1, :, 2] = np.clip(background[height-i-1, :, 2] - factor * aging_noise[height-i-1, :], 0, 255)

            # Left and right edges
            if i < width:
                background[:, i, 2] = np.clip(background[:, i, 2] - factor * aging_noise[:, i], 0, 255)
            if width - i - 1 >= 0:
                background[:, width-i-1, 2] = np.clip(background[:, width-i-1, 2] - factor * aging_noise[:, width-i-1], 0, 255)

    elif style == "birch":
        background = np.ones((height, width, 3), dtype=np.uint8) * [235, 225, 215]

        # Add fiber texture
        background = np.clip(background.astype(np.float32) - fiber_texture, 0, 255).astype(np.uint8)

        # Enhanced birch bark texture
        noise = np.random.randint(0, int(10 * params["noise"]), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)

        # More realistic bark patterns
        variation_count = int(150 * params["texture"])
        for _ in range(variation_count):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(10, 25)
            variation = random.randint(-6, 6) * params["texture"]

            # Create more organic shapes
            for i in range(-size, size):
                for j in range(-size, size):
                    dist = np.sqrt(i*i + j*j)
                    if dist <= size:
                        # Add some randomness to the shape
                        shape_factor = np.random.uniform(0.7, 1.3)
                        if dist <= size * shape_factor:
                            yi, xi = y + i, x + j
                            if 0 <= yi < height and 0 <= xi < width:
                                background[yi, xi, :] = np.clip(
                                    background[yi, xi, :] + variation, 0, 255
                                )

    else:  # "parchment"
        background = np.ones((height, width, 3), dtype=np.uint8) * [230, 215, 185]

        # Add fiber texture
        background = np.clip(background.astype(np.float32) - fiber_texture, 0, 255).astype(np.uint8)

        # Enhanced parchment texture
        variation_count = int(400 * params["texture"])
        for _ in range(variation_count):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(5, 12)
            variation = random.randint(-7, 7) * params["texture"]

            # More realistic parchment grain
            for i in range(-size, size):
                for j in range(-size, size):
                    dist = np.sqrt(i*i + j*j)
                    if dist <= size:
                        # Add grain direction
                        grain_factor = 1 + 0.3 * np.sin(j * 0.5) * np.cos(i * 0.3)
                        if dist <= size * grain_factor:
                            yi, xi = y + i, x + j
                            if 0 <= yi < height and 0 <= xi < width:
                                background[yi, xi, :] = np.clip(
                                    background[yi, xi, :] + variation, 0, 255
                                )

        # Enhanced noise
        noise = np.random.randint(0, int(8 * params["noise"]), (height, width, 3), dtype=np.uint8)
        background = np.clip(background - noise, 0, 255).astype(np.uint8)

    return Image.fromarray(background)
