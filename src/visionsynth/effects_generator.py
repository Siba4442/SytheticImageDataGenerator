"""
VisionSynth Effects Generator

This module provides a comprehensive collection of image processing effects
for simulating paper textures, document artifacts, and various visual distortions.
All functions are optimized for performance using vectorized NumPy operations.

Author: VisionSynth Team
"""

import cv2
import numpy as np
from noise import pnoise2
from PIL import Image, ImageFilter

# Module-level constants for consistent effect parameters
DEFAULT_FIBER_DENSITY = 0.2
DEFAULT_FOLD_INTENSITY = 0.5
DEFAULT_BLEED_INTENSITY = 0.3
DEFAULT_BLEED_RADIUS = 3
DEFAULT_CORNER_DISPLACEMENT = 20
DEFAULT_SHADOW_ANGLE = 45
DEFAULT_SHADOW_INTENSITY = 0.4
DEFAULT_KERNEL_SIZE = 3
DEFAULT_COMPRESSION_QUALITY = 85
DEFAULT_LENS_DISTORTION = 0.2


def generate_perlin_noise(
    width: int, height: int, scale: float = 0.1, octaves: int = 4
) -> np.ndarray:
    """
    Generate Perlin noise using vectorized operations for improved performance.

    Args:
        width (int): Width of the noise map in pixels
        height (int): Height of the noise map in pixels
        scale (float, optional): Scale factor for noise frequency.
                               Smaller values create larger features. Defaults to 0.1.
        octaves (int, optional): Number of noise octaves to combine.
                               More octaves add detail. Defaults to 4.

    Returns:
        np.ndarray: 2D array of noise values in range [-1, 1]
    """
    y, x = np.mgrid[0:height, 0:width]

    # Use vectorized Perlin noise generation for better performance
    noise_map = np.vectorize(lambda i, j: pnoise2(i * scale, j * scale, octaves=octaves))(y, x)

    return noise_map


def simulate_paper_fiber_texture(width: int, height: int, fiber_density: float = 0.2) -> np.ndarray:
    """
    Generate realistic paper fiber texture using layered Perlin noise.

    Creates a paper-like texture by combining coarse and fine noise patterns
    to simulate the natural fiber structure of paper.

    Args:
        width (int): Width of the texture in pixels
        height (int): Height of the texture in pixels
        fiber_density (float, optional): Intensity of fiber visibility (0.0-1.0).
                                       Higher values create more prominent fibers.
                                       Defaults to 0.2.

    Returns:
        np.ndarray: RGB texture array with shape (height, width, 3)
    """
    try:
        # Generate base coarse texture for paper structure
        base_texture = generate_perlin_noise(width, height, scale=0.02, octaves=4)

        # Generate fine texture for detailed fiber patterns
        fine_texture = generate_perlin_noise(width, height, scale=0.1, octaves=2)

        # Combine textures with weighted blend (70% base, 30% fine)
        combined = base_texture * 0.7 + fine_texture * 0.3

        # Normalize from [-1,1] to [0,1] and apply fiber density
        combined = ((combined + 1) / 2) * fiber_density * 255

        # Convert to RGB by stacking the same pattern across all channels
        texture = np.stack([combined] * 3, axis=2)
        return np.clip(texture, 0, 255).astype(np.uint8)

    except Exception:
        # Fallback to random noise if Perlin noise generation fails
        return np.random.randint(0, int(255 * fiber_density), (height, width, 3), dtype=np.uint8)


def simulate_fold_crease(
    image: np.ndarray, fold_lines: list[tuple[int, int, int, int]], fold_intensity: float = 0.5
) -> np.ndarray:
    """
    Apply realistic paper fold creases with shadows and highlights.

    Simulates the visual effect of paper folds by creating dark creases with
    asymmetric shadows and highlights on either side of fold lines.

    Args:
        image (np.ndarray): Input image as RGB array
        fold_lines (List[Tuple[int, int, int, int]]): List of fold lines as
                    (x1, y1, x2, y2) coordinates
        fold_intensity (float, optional): Strength of fold effects (0.0-1.0).
                                        Defaults to 0.5.

    Returns:
        np.ndarray: Image with fold effects applied
    """
    height, width = image.shape[:2]
    result = image.astype(np.float32)

    # Constants for fold effect intensities
    CREASE_INTENSITY = 50
    SHADOW_INTENSITY = 25
    HIGHLIGHT_INTENSITY = 10
    FOLD_WIDTH_RATIO = 0.05  # Fold width as ratio of image size

    for x1, y1, x2, y2 in fold_lines:
        y_coords, x_coords = np.ogrid[:height, :width]

        # Calculate perpendicular distance from each pixel to the fold line
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_length == 0:
            continue  # Skip zero-length lines

        distances = (
            np.abs((y2 - y1) * x_coords - (x2 - x1) * y_coords + x2 * y1 - y2 * x1) / line_length
        )

        # Create Gaussian profile for smooth fold transition
        fold_width = min(width, height) * FOLD_WIDTH_RATIO
        fold_profile = np.exp(-0.5 * (distances / fold_width) ** 2)

        # Create dark crease along the fold line
        fold_effect = fold_profile * fold_intensity * CREASE_INTENSITY

        # Determine which side of the line each pixel is on for asymmetric effects
        shadow_mask = (y_coords - y1) * (x2 - x1) - (x_coords - x1) * (y2 - y1) > 0

        # Apply shadow on one side
        shadow_effect = fold_profile * shadow_mask * fold_intensity * SHADOW_INTENSITY

        # Apply subtle highlight on the other side
        highlight_effect = fold_profile * (~shadow_mask) * fold_intensity * HIGHLIGHT_INTENSITY

        # Apply all effects to the image
        result -= fold_effect[..., np.newaxis] + shadow_effect[..., np.newaxis]
        result += highlight_effect[..., np.newaxis]

    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_ink_bleed(
    image: np.ndarray, bleed_intensity: float = 0.3, bleed_radius: int = 3
) -> np.ndarray:
    """
    Simulate ink bleeding effect using morphological operations and Gaussian blur.

    Creates a realistic ink bleeding effect by identifying dark text regions
    and applying controlled expansion with blur to simulate ink spreading
    into paper fibers.

    Args:
        image (np.ndarray): Input RGB image
        bleed_intensity (float, optional): Strength of bleeding effect (0.0-1.0).
                                         Defaults to 0.3.
        bleed_radius (int, optional): Radius of bleeding spread in pixels.
                                    Defaults to 3.

    Returns:
        np.ndarray: Image with ink bleeding effect applied
    """
    # Convert to grayscale for text detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold to identify text/ink regions (assuming dark text on light background)
    TEXT_THRESHOLD = 200
    _, text_mask = cv2.threshold(gray, TEXT_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Create morphological kernel for bleeding effect
    kernel_size = max(1, int(bleed_radius * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Dilate text regions to simulate ink spreading
    bleeding_mask = cv2.dilate(text_mask, kernel, iterations=1)

    # Apply Gaussian blur for smooth bleeding transition
    bleed_effect = cv2.GaussianBlur(bleeding_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    bleed_effect = bleed_effect * bleed_intensity / 255.0

    # Apply bleeding effect with darkening factor
    DARKENING_FACTOR = 0.7
    result = image.astype(np.float32)
    result = (
        result * (1 - bleed_effect[..., None])
        + (result * DARKENING_FACTOR) * bleed_effect[..., None]
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_perspective_distortion(image: np.ndarray, corner_displacement: int = 20) -> np.ndarray:
    """
    Apply random perspective distortion to simulate document scanning angles.

    Randomly displaces image corners to create realistic perspective effects
    that might occur when photographing or scanning documents at angles.

    Args:
        image (np.ndarray): Input RGB image
        corner_displacement (int, optional): Maximum pixel displacement for corners.
                                           Defaults to 20.

    Returns:
        np.ndarray: Perspective-distorted image
    """
    h, w = image.shape[:2]

    # Define original corner positions
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Apply random displacement to corners with boundary constraints
    displacement = np.random.randint(-corner_displacement, corner_displacement + 1, size=(4, 2))
    dst = (src + displacement).astype(np.float32)

    # Ensure displaced corners stay within reasonable bounds
    BOUNDARY_TOLERANCE = 0.1  # 10% beyond image boundaries
    dst[:, 0] = np.clip(dst[:, 0], -w * BOUNDARY_TOLERANCE, w * (1 + BOUNDARY_TOLERANCE))
    dst[:, 1] = np.clip(dst[:, 1], -h * BOUNDARY_TOLERANCE, h * (1 + BOUNDARY_TOLERANCE))

    # Calculate and apply perspective transformation
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_shadow_effects(
    image: np.ndarray,
    shadow_angle: float = 45,
    shadow_intensity: float = 0.4,
    power: float = 1.0,
    blur_sigma: float = 5.0,
) -> np.ndarray:
    """
    Apply directional shadow effects to simulate lighting conditions.

    Creates realistic shadows by generating a directional gradient based on
    the specified angle and applying it as a multiplicative mask.

    Args:
        image (np.ndarray): Input RGB image
        shadow_angle (float, optional): Shadow direction angle in degrees.
                                      Defaults to 45.
        shadow_intensity (float, optional): Strength of shadow effect (0.0-1.0).
                                          Defaults to 0.4.
        power (float, optional): Shadow gradient power for non-linear falloff.
                               Defaults to 1.0 (linear).
        blur_sigma (float, optional): Gaussian blur radius for shadow softening.
                                     Set to 0 for sharp shadows. Defaults to 5.0.

    Returns:
        np.ndarray: Image with shadow effects applied
    """
    h, w = image.shape[:2]
    img = image.copy().astype(np.float32) / 255.0  # Normalize to [0,1]

    # Convert angle to radians and create coordinate grid
    angle_rad = np.radians(shadow_angle)
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

    # Create directional projection for shadow gradient
    projection = np.cos(angle_rad) * x + np.sin(angle_rad) * y

    # Normalize projection to [0, 1] range
    shadow = projection - np.min(projection)
    if np.max(shadow) > 0:
        shadow = shadow / np.max(shadow)

    # Apply power transformation for non-linear shadow falloff
    shadow = shadow**power

    # Ensure proper normalization after power transformation
    if np.max(shadow) > 0:
        shadow = np.clip(shadow / np.max(shadow), 0, 1)

    # Apply Gaussian blur for soft shadow edges
    if blur_sigma > 0:
        shadow_img = Image.fromarray((shadow * 255).astype(np.uint8))
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        shadow = np.array(shadow_img).astype(np.float32) / 255.0

        # Re-normalize after blur
        if np.max(shadow) > 0:
            shadow = np.clip(shadow / np.max(shadow), 0, 1)

    # Create shadow mask and apply to image
    mask = 1 - shadow * shadow_intensity
    img *= mask[..., None]

    return np.clip(img * 255, 0, 255).astype(np.uint8)


def apply_morphological_operations(
    image: np.ndarray, operation: str = "mixed", kernel_size: int = 3
) -> np.ndarray:
    """
    Apply morphological operations to modify text and shape characteristics.

    Performs morphological operations (erosion, dilation, opening, closing)
    on the image to simulate effects like ink spreading, paper absorption,
    or printing artifacts.

    Args:
        image (np.ndarray): Input RGB image
        operation (str, optional): Type of operation. Options are:
                                 - 'erosion': Shrinks bright regions
                                 - 'dilation': Expands bright regions
                                 - 'opening': Erosion followed by dilation
                                 - 'closing': Dilation followed by erosion
                                 - 'mixed': Randomly selects operation
                                 Defaults to 'mixed'.
        kernel_size (int, optional): Size of morphological kernel (odd numbers).
                                    Defaults to 3.

    Returns:
        np.ndarray: Image with morphological operations applied
    """
    # Convert to grayscale for morphological operations
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create elliptical kernel for natural-looking effects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Randomly select operation if 'mixed' is specified
    if operation == "mixed":
        operation = np.random.choice(["erosion", "dilation", "opening", "closing"])

    # Apply the specified morphological operation
    if operation == "erosion":
        processed = cv2.erode(gray, kernel, iterations=1)
    elif operation == "dilation":
        processed = cv2.dilate(gray, kernel, iterations=1)
    elif operation == "opening":
        processed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation == "closing":
        processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    else:
        processed = gray  # Fallback to original if operation not recognized

    # Convert back to RGB format
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)


def simulate_scanner_artifacts(image: np.ndarray, compression_quality: int = 85) -> np.ndarray:
    """
    Simulate common scanner artifacts including scan lines, dust, and compression.

    Adds realistic scanner imperfections such as horizontal scan lines,
    dust spots, and JPEG compression artifacts to mimic real scanner output.

    Args:
        image (np.ndarray): Input RGB image
        compression_quality (int, optional): JPEG compression quality (1-100).
                                           Lower values increase compression artifacts.
                                           Defaults to 85.

    Returns:
        np.ndarray: Image with scanner artifacts applied
    """
    result = image.copy().astype(np.float32)
    height, width = result.shape[:2]

    # Add horizontal scan lines (vectorized for performance)
    scan_line_spacing = np.random.randint(8, 15)
    ys = np.arange(0, height, scan_line_spacing)

    # Random intensities for each scan line
    line_intensities = np.random.randint(5, 15, size=len(ys))
    result[ys, :, :] = np.clip(result[ys, :, :] - line_intensities[:, None, None], 0, 255)

    # Add random dust spots
    dust_count = np.random.randint(3, 8)
    for _ in range(dust_count):
        # Random dust spot parameters
        dust_x = np.random.randint(0, width - 5)
        dust_y = np.random.randint(0, height - 5)
        dust_size = np.random.randint(2, 5)
        dust_intensity = np.random.randint(20, 40)

        # Apply dust spot as darkened area
        result[dust_y : dust_y + dust_size, dust_x : dust_x + dust_size, :] = np.clip(
            result[dust_y : dust_y + dust_size, dust_x : dust_x + dust_size, :] - dust_intensity,
            0,
            255,
        )

    # Apply JPEG compression artifacts
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]
    _, encoded_img = cv2.imencode(
        ".jpg", cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR), encode_param
    )

    # Decode to simulate compression loss
    decoded = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def apply_lens_distortion(image: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """
    Apply barrel/pincushion lens distortion to simulate camera lens effects.

    Creates realistic lens distortion by applying radial displacement
    that varies with distance from the image center.

    Args:
        image (np.ndarray): Input RGB image
        strength (float, optional): Distortion strength. Positive values create
                                  barrel distortion, negative values create
                                  pincushion distortion. Defaults to 0.2.

    Returns:
        np.ndarray: Image with lens distortion applied
    """
    h, w = image.shape[:2]

    # Calculate image center
    cx, cy = w / 2, h / 2

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    dx, dy = x - cx, y - cy

    # Calculate radial distance from center
    r = np.sqrt(dx**2 + dy**2)
    r_norm = r / r.max()  # Normalize to [0, 1]

    # Apply quadratic distortion model
    distortion_factor = 1 + strength * r_norm**2

    # Calculate corrected coordinates
    map_x = (dx / distortion_factor + cx).astype(np.float32)
    map_y = (dy / distortion_factor + cy).astype(np.float32)

    # Apply distortion using bilinear interpolation
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def generate_random_fold_lines(
    image_size: tuple[int, int], num_folds: int | None = None
) -> list[tuple[int, int, int, int]]:
    """
    Generate random fold lines for creating realistic paper crease effects.

    Creates random line segments across the image that can be used with
    the simulate_fold_crease function to add paper fold effects.

    Args:
        image_size (Tuple[int, int]): Image dimensions as (width, height)
        num_folds (Optional[int], optional): Number of fold lines to generate.
                                           If None, randomly selects 1-3 folds.
                                           Defaults to None.

    Returns:
        List[Tuple[int, int, int, int]]: List of fold lines as (x1, y1, x2, y2)
                                       coordinate tuples
    """
    width, height = image_size

    # Set random number of folds if not specified
    if num_folds is None:
        num_folds = np.random.randint(1, 3)

    fold_lines = []
    for _ in range(num_folds):
        # Generate random fold line endpoints
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(0, width)
        y2 = np.random.randint(0, height)

        fold_lines.append((x1, y1, x2, y2))

    return fold_lines
