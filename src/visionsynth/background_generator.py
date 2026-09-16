"""
VisionSynth Background Generator

This module provides a comprehensive collection of background generation functions
for creating various paper textures, vintage effects, and document backgrounds.
All functions return PIL Images in RGBA format for consistent usage.

Author: VisionSynth Team
"""

import os

import cv2
import numpy as np
from PIL import Image

# Module-level constants for consistent background generation
DEFAULT_BASE_COLOR_OLD_PAPER = [236, 222, 181]
DEFAULT_BASE_COLOR_BIRCH = [235, 225, 215]
DEFAULT_BASE_COLOR_PARCHMENT = [245, 245, 220]
DEFAULT_GAUSSIAN_NOISE_MEAN = 235
DEFAULT_GAUSSIAN_NOISE_STD = 10
DEFAULT_LINE_SPACING_MIN = 15
DEFAULT_LINE_SPACING_MAX = 25
MAX_PIXELS_SAFETY_LIMIT = 80_000_000


def plain_background(height: int, width: int) -> Image.Image:
    """
    Create a plain white background suitable for document generation.

    Generates a simple white background that can serve as a base layer
    for document synthesis or text overlay applications.

    Args:
        height (int): Height of the background in pixels
        width (int): Width of the background in pixels

    Returns:
        PIL.Image: RGBA image with plain white background
    """
    # Create an all-white image array
    image = np.ones((height, width)) * 255

    return Image.fromarray(image).convert("RGBA")


def gaussian_noise(height: int, width: int) -> Image.Image:
    """
    Create a background with Gaussian noise to simulate natural paper texture.

    Generates a white background with subtle Gaussian noise to mimic the
    natural variations and texture found in real paper.

    Args:
        height (int): Height of the background in pixels
        width (int): Width of the background in pixels

    Returns:
        PIL.Image: RGBA image with Gaussian noise texture
    """
    # Create base white image
    image = np.ones((height, width)) * 255

    # Add Gaussian noise with mean=235, std=10 for subtle paper texture
    # cv2's stubs only declare the ndarray-mean/stddev overload; scalars work fine at runtime.
    cv2.randn(image, DEFAULT_GAUSSIAN_NOISE_MEAN, DEFAULT_GAUSSIAN_NOISE_STD)  # type: ignore[call-overload]

    return Image.fromarray(image).convert("RGBA")


def lined_paper(height: int, width: int) -> Image.Image:
    """
    Create a realistic lined paper background with horizontal ruling lines.

    Generates a white background with evenly spaced horizontal lines
    similar to notebook paper, including subtle noise for realism.

    Args:
        height (int): Height of the background in pixels
        width (int): Width of the background in pixels

    Returns:
        PIL.Image: RGBA image with lined paper effect
    """
    # Create base white RGB image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Generate horizontal ruling lines with random spacing
    line_spacing = np.random.randint(DEFAULT_LINE_SPACING_MIN, DEFAULT_LINE_SPACING_MAX)

    for y in range(0, height, line_spacing):
        # Random line width and darkness for natural variation
        line_width = np.random.randint(1, 2)
        line_darkness = 200  # Darkness value for line color

        # Draw line if it fits within image bounds
        if y + line_width < height:
            image[y : y + line_width, :, :] = np.clip(
                image[y : y + line_width, :, :] - line_darkness, 0, 255
            )

    # Add subtle noise for realistic paper texture
    noise_intensity = 2 * 100  # Noise intensity factor
    noise = np.random.randint(0, int(noise_intensity), (height, width, 3), dtype=np.uint8)
    image = np.clip(image - noise, 0, 255).astype(np.uint8)

    return Image.fromarray(image).convert("RGBA")


def strain(height: int, width: int) -> Image.Image:
    """
    Create a background with realistic strain marks and discoloration spots.

    Generates circular strain marks with gradient falloff to simulate
    water damage, aging spots, or other natural paper deterioration.

    Args:
        height (int): Height of the background in pixels
        width (int): Width of the background in pixels

    Returns:
        PIL.Image: RGBA image with strain mark effects
    """
    # Create base white RGB image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Generate random strain marks
    strain_count = int(np.random.randint(2, 4))

    for _ in range(strain_count):
        # Random position with boundary buffer
        x = np.random.randint(0, width - 100)
        y = np.random.randint(0, height - 100)

        # Random size and darkness for natural variation
        size = np.random.randint(20, 60)
        darkness = np.random.randint(8, 25) * 5

        # Create circular gradient mask for strain effect
        center = size // 2
        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
        norm = dist / center

        # Create radial falloff mask (1 at center, 0 at edge)
        strain_mask = np.clip(1 - norm, 0, 1)

        # Add random variation for irregular shape
        strain_mask *= np.random.uniform(0.4, 1.0, (size, size))

        # Apply Gaussian blur for soft edges
        strain_mask = cv2.GaussianBlur(strain_mask, (5, 5), 0)

        # Calculate actual bounds to avoid array overflow
        end_y = min(y + size, height)
        end_x = min(x + size, width)
        actual_size_y = end_y - y
        actual_size_x = end_x - x

        # Apply strain effect if valid region exists
        if actual_size_y > 0 and actual_size_x > 0:
            strain_region = strain_mask[:actual_size_y, :actual_size_x]
            image[y:end_y, x:end_x, :] = np.clip(
                image[y:end_y, x:end_x, :] - (darkness * strain_region[..., None]), 0, 255
            )

    return Image.fromarray(image).convert("RGBA")


def old_paper(height: int, width: int) -> Image.Image:
    """
    Create a vintage old paper background with aged edges and warm coloring.

    Generates a paper texture with aged yellow-brown coloring and darkened
    edges to simulate natural aging and wear patterns.

    Args:
        height (int): Height of the background in pixels
        width (int): Width of the background in pixels

    Returns:
        PIL.Image: RGBA image with old paper aging effects
    """
    # Define warm aged paper base color (beige/cream)
    base_color = np.array(DEFAULT_BASE_COLOR_OLD_PAPER, dtype=np.float32)
    background = np.ones((height, width, 3), dtype=np.float32) * base_color

    # Calculate edge darkening width (10% of smallest dimension)
    edge_width = max(1, int(min(width, height) * 0.1))

    # Create coordinate grids for edge effects
    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]

    # Calculate edge proximity masks for all four edges
    top_edge = np.clip((edge_width - yy) / edge_width, 0, 1)
    bottom_edge = np.clip((edge_width - (height - 1 - yy)) / edge_width, 0, 1)
    left_edge = np.clip((edge_width - xx) / edge_width, 0, 1)
    right_edge = np.clip((edge_width - (width - 1 - xx)) / edge_width, 0, 1)

    # Combine all edge effects
    edge_mask = top_edge + bottom_edge + left_edge + right_edge

    # Add random aging noise for natural variation
    aging_noise = np.random.uniform(0.5, 1.5, (height, width))

    # Apply aging effect primarily to blue channel for yellow-brown tint
    aging_intensity = 15
    background[:, :, 2] -= edge_mask * aging_noise * aging_intensity

    # Ensure values stay within valid range
    background = np.clip(background, 0, 255).astype(np.uint8)

    return Image.fromarray(background).convert("RGBA")


def birch(height: int, width: int, params: dict | None = None) -> Image.Image:
    """
    Generate a textured 'birch' background by stamping many small irregular circular spots.

    Creates a textured background using numerous small circular spots with irregular
    shapes and random positioning to simulate natural paper texture variations.

    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels
        params (Dict, optional): Control parameters with the following options:
            - "texture" (float): Multiplier for spot count (default 1.0)
            - "spots" (int): Explicit number of spots (overrides texture if present)
            - "min_radius" (int): Minimum spot radius in pixels (default 10)
            - "max_radius" (int): Maximum spot radius in pixels (default 25)
            - "intensity" (float): Maximum per-spot intensity/absolute pixel change (default 10)
            - "spot_sign" (int): -1 for dark spots (stains), +1 for bright spots,
                                0 for random (default -1)
            - "irregularity" (float): 0-1, how irregular spots are (default 0.35)
            - "blur" (float): Gaussian blur sigma for soft edges (default 1.2)
            - "cap_spots" (int): Safety cap on maximum spots (default 2000)

    Returns:
        PIL.Image: RGBA textured background image

    Raises:
        ValueError: If dimensions are invalid or image too large
    """
    if params is None:
        params = {}

    # Input validation
    if not (isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0):
        raise ValueError("height and width must be positive integers")

    if height * width > MAX_PIXELS_SAFETY_LIMIT:
        raise ValueError("Requested image too large.")

    # Extract parameters with defaults
    texture = float(params.get("texture", 1.0))
    spots = params.get("spots")
    if spots is None:
        spots = max(1, int(150 * texture))
    spots = min(int(spots), int(params.get("cap_spots", 2000)))

    min_radius = int(params.get("min_radius", 10))
    max_radius = int(params.get("max_radius", 25))
    intensity = float(params.get("intensity", 10.0))
    spot_sign = params.get("spot_sign", -1)
    irregularity = float(params.get("irregularity", 0.35))
    blur_sigma = float(params.get("blur", 1.2))

    # Initialize canvas with warm tone base color
    base_color = np.array(DEFAULT_BASE_COLOR_BIRCH, dtype=np.float32)
    canvas = np.ones((height, width, 3), dtype=np.float32) * base_color[None, None, :]

    # Accumulator for delta to avoid repeated writes
    delta_total = np.zeros_like(canvas, dtype=np.float32)

    # Generate spots
    for _ in range(spots):
        # Random center position and radius
        cx = int(np.random.randint(0, width))
        cy = int(np.random.randint(0, height))
        r = int(np.random.randint(min_radius, max_radius + 1))

        # Calculate bounding box (clamped to image bounds)
        y0 = max(0, cy - r)
        y1 = min(height, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(width, cx + r + 1)

        patch_height = y1 - y0
        patch_width = x1 - x0

        if patch_height <= 0 or patch_width <= 0:
            continue

        # Create local coordinate grid relative to spot center
        yy = np.arange(y0, y1)[:, None] - cy  # Shape (patch_height, 1)
        xx = np.arange(x0, x1)[None, :] - cx  # Shape (1, patch_width)
        dist = np.sqrt(yy.astype(np.float32) ** 2 + xx.astype(np.float32) ** 2)

        # Create base radial mask (1 at center -> 0 at radius)
        radial_mask = np.clip(1 - dist / (r + 1e-9), 0, 1)

        # Add irregularity through random scaling and per-pixel noise
        scale = 1.0 + (np.random.uniform(-0.6, 0.6) * irregularity)
        pixel_noise = np.random.uniform(
            1.0 - irregularity, 1.0 + irregularity, size=(patch_height, patch_width)
        ).astype(np.float32)

        mask = radial_mask * scale * pixel_noise
        mask = np.clip(mask, 0, 1)

        # Apply Gaussian blur for soft edges
        if blur_sigma > 0:
            kernel_size = int(2 * np.ceil(3 * blur_sigma) + 1)
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), blur_sigma)

        # Create irregular rim by thresholding with random factor
        rim_threshold = np.random.uniform(0.85, 1.05)
        mask = np.where(dist <= r * rim_threshold, mask, 0.0).astype(np.float32)

        # Determine intensity sign (darken/brighten)
        if spot_sign == 0:
            sign = int(np.random.choice([-1, 1]))
        else:
            sign = int(np.sign(spot_sign))
            if sign == 0:
                sign = -1

        # Calculate spot intensity with random variation
        spot_intensity = sign * intensity * np.random.uniform(0.6, 1.4)

        # Add delta for this patch to accumulator
        delta_patch = (mask[..., None]) * spot_intensity
        delta_total[y0:y1, x0:x1, :] += delta_patch

    # Apply all accumulated changes
    canvas += delta_total
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    return Image.fromarray(canvas, mode="RGB").convert("RGBA")


def parchment(height: int, width: int, params: dict | None = None) -> Image.Image:
    """
    Generate a realistic parchment-like texture background.

    Creates a parchment texture using numerous small spots with grain effects
    and subtle color variations to simulate aged paper or vellum.

    Args:
        height (int): Output image height in pixels
        width (int): Output image width in pixels
        params (Dict, optional): Control parameters with the following options:
            - "texture" (float): Multiplication factor for number of spots (default 1.0)
            - "spots" (int): Explicit number of spots (overrides texture if provided)
            - "min_radius" (int): Minimum spot radius in pixels (default 3)
            - "max_radius" (int): Maximum spot radius in pixels (default 12)
            - "intensity" (float): Spot intensity magnitude (default 7.0).
                                  Positive values brighten, negative values darken.
            - "spot_sign" (int): -1 for dark stains, +1 for bright spots,
                                0 for random per-spot (default -1)
            - "irregularity" (float): 0-1, how irregular spots are (default 0.35)
            - "blur_sigma" (float): Gaussian blur sigma for spot softening (default 1.2)
            - "base_color" (tuple): RGB base paper color (default (245,245,220))
            - "cap_spots" (int): Safety cap on maximum spots (default 2000)

    Returns:
        PIL.Image: RGBA parchment texture image

    Raises:
        ValueError: If dimensions are invalid or image too large
    """
    if params is None:
        params = {}

    # Input validation
    if not (isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0):
        raise ValueError("height and width must be positive integers")

    if height * width > MAX_PIXELS_SAFETY_LIMIT:
        raise ValueError("Requested image too large; reduce dimensions.")

    # Extract parameters with defaults (fixed typos in parameter names)
    texture = float(params.get("texture", 1.0))
    spots = params.get("spots")
    base_spots = int(400 * texture)
    if spots is None:
        spots = max(1, base_spots)
    spots = min(spots, int(params.get("cap_spots", 2000)))

    min_radius = int(params.get("min_radius", 3))
    max_radius = int(params.get("max_radius", 12))
    intensity_base = float(params.get("intensity", 7.0))
    spot_sign = int(params.get("spot_sign", -1))  # Fixed typo: was "spots_sign"
    irregularity = float(params.get("irregularity", 0.35))  # Fixed typo: was "irregualarity"
    blur_sigma = float(params.get("blur_sigma", 1.2))
    base_color = np.array(params.get("base_color", DEFAULT_BASE_COLOR_PARCHMENT), dtype=np.float32)
    cap_spots = int(params.get("cap_spots", 2000))

    # Ensure spots count is within valid range
    spots = max(0, min(spots, cap_spots))

    # Initialize canvas with base parchment color
    canvas = np.ones((height, width, 3), dtype=np.float32) * base_color[None, None, :]

    # Accumulator for all spot effects
    delta_total = np.zeros_like(canvas, dtype=np.float32)

    # Generate parchment spots
    for _ in range(spots):
        # Random spot position and size
        cx = int(np.random.randint(0, width))
        cy = int(np.random.randint(0, height))
        r = int(np.random.randint(min_radius, max_radius + 1))

        # Calculate bounding box
        y0 = max(0, cy - r)
        y1 = min(height, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(width, cx + r + 1)

        patch_height = y1 - y0
        patch_width = x1 - x0
        if patch_height <= 0 or patch_width <= 0:
            continue

        # Create local coordinates relative to spot center
        yy = np.arange(y0, y1)[:, None] - cy
        xx = np.arange(x0, x1)[None, :] - cx
        dist = np.sqrt(yy.astype(np.float32) ** 2 + xx.astype(np.float32) ** 2)

        # Create base radial mask with grain effect
        grain_factor = 1 + 0.3 * np.sin(xx * 0.5) * np.cos(yy * 0.3)
        mask = np.clip(1.0 - (dist / (r * grain_factor + 1e-9)), 0.0, 1.0)

        # Add irregularity through pixel noise and scaling
        pixel_noise = np.random.uniform(
            1.0 - irregularity, 1.0 + irregularity, size=(patch_height, patch_width)
        ).astype(np.float32)
        scale = 1.0 + irregularity * (np.random.uniform(-0.4, 0.4))
        mask *= pixel_noise * scale
        mask = np.clip(mask, 0.0, 1.0)

        # Apply Gaussian blur for soft edges
        if blur_sigma > 0:
            kernel_size = max(3, int(blur_sigma * 3) | 1)  # Ensure odd kernel size
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), blur_sigma)

        # Determine spot intensity direction
        if spot_sign == 0:
            sign = int(np.random.choice([-1, 1]))
        else:
            sign = np.sign(spot_sign) if spot_sign != 0 else -1
            sign = int(sign)

        # Calculate final spot intensity with random variation
        spot_intensity = sign * intensity_base * float(np.random.uniform(0.8, 1.2))

        # Add spot effect to accumulator
        delta_patch = (mask[..., None]) * spot_intensity
        delta_total[y0:y1, x0:x1, :] += delta_patch

    # Apply all accumulated spot effects
    canvas += delta_total

    # Ensure values are within valid range and convert to uint8
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    return Image.fromarray(canvas, mode="RGB").convert("RGBA")


def image(height: int, width: int, image_dir: str) -> Image.Image:
    """
    Create a background using a randomly selected image from a directory.

    Loads a random image from the specified directory, resizes it to fit
    the required dimensions, and crops it to the exact size needed.

    Args:
        height (int): Required background height in pixels
        width (int): Required background width in pixels
        image_dir (str): Path to directory containing background images

    Returns:
        PIL.Image: RGBA background image cropped to specified dimensions

    Raises:
        Exception: If no images are found in the specified directory
    """
    # Get list of all files in the image directory
    images = os.listdir(image_dir)

    if len(images) > 0:
        # Select a random image from the directory
        selected_image = images[np.random.randint(0, len(images))]
        pic: Image.Image = Image.open(os.path.join(image_dir, selected_image))

        # Resize image to ensure it covers the required dimensions

        # If image width is smaller than required, scale up based on width
        if pic.size[0] < width:
            scale_factor = width / pic.size[0]
            new_width = width
            new_height = int(pic.size[1] * scale_factor)
            pic = pic.resize([new_width, new_height], Image.Resampling.LANCZOS)

        # If image height is still smaller than required, scale up based on height
        if pic.size[1] < height:
            scale_factor = height / pic.size[1]
            new_width = int(pic.size[0] * scale_factor)
            new_height = height
            pic = pic.resize([new_width, new_height], Image.Resampling.LANCZOS)

        # Calculate crop position (random if image is larger than needed)
        crop_x = 0 if pic.size[0] == width else np.random.randint(0, pic.size[0] - width)
        crop_y = 0 if pic.size[1] == height else np.random.randint(0, pic.size[1] - height)

        # Crop to exact required dimensions
        return pic.crop((crop_x, crop_y, crop_x + width, crop_y + height))
    else:
        raise Exception("No images were found in the images folder!")
