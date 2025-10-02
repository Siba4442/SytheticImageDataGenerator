import os
import random as rnd
from PIL import Image
import numpy as np
import cv2  
import noise
from typing import Optional, Dict

def brich1(height: int, width:int, params: dict= {}) -> Image.Image:

    base_color = np.array([235, 225, 215], dtype=np.float32)
    background = np.ones((height, width, 3), dtype=np.uint8) * base_color
    
    variation = int(150 * params.get("texture", 1.0))
    
    for _ in range(variation):
        x = rnd.randint(0, width-1)
        y = rnd.randint(0, height-1)
        
        size = rnd.randint(10, 25)
        
        variation = rnd.randint(-6, 6) * params.get("texture", 1.0)
        
        for i in range(-size, size):
            for j in range(-size, size):
                dist = np.sqrt(i**2 + j**2)
                if dist <= size:
                    shape_factor = np.random.uniform(0.7, 1.3)
                    if dist <= size * shape_factor:
                        yi, xi = y + i, x + j
                        if 0 <= yi < height and 0 <= xi < width:
                            background[yi, xi, :] = np.clip(
                                background[yi, xi, :] + variation, 0, 255
                            )

    return Image.fromarray(background.astype(np.uint8)).convert("RGBA")


def parchment1(height: int, width: int, params: dict = {}):
    
    base_color = np.array([245, 245, 220], dtype=np.float32)

    background = np.ones((height, width, 3), dtype=np.float32) * base_color

    variation_count = int(400 * params.get("texture", 1.0))
    for _ in range(variation_count):
        x = rnd.randint(0, width-1)
        y = rnd.randint(0, height-1)
        
        size = rnd.randint(5, 12)
        variation = rnd.randint(-7, 7) * params.get("texture", 1.0)

        for i in range(-size, size):
            for j in range(-size, size):
                dist = np.sqrt(i**2 + j**2)
                
                if dist <= size:
                    grain_factor = rnd.randint(-7, 7) * params.get("texture", 1.0)
                    if dist <= size * grain_factor:
                        yi, xi = y + i, x + j
                        if 0 <= yi < height and 0 <= xi < width:
                            background[yi, xi, :] = np.clip(background[yi, xi, :] + variation, 0, 255)
    # Add some noise
    noise = np.random.normal(0, 25, (height, width, 3)).astype(np.float32)
    
    background = np.clip(background, 0, 255)

    return Image.fromarray(background.astype(np.uint8)).convert("RGBA")


def parchment(height: int, width: int, params: Optional[Dict] = None) -> Image.Image:
    """
    Generate a parchment-like texture image.
    
    Args:
        height (int), width (int): output size in pixels.
        params (dict, optional): control parameters:
            - texture (float): multiplication factor for number of spots (default 1.0)
            - spots (int): explicit number of spots (overrides texture if provided)
            - min_radius (int): min spot radius in pixels (default 3)
            - max_radius (int): max spot radius in pixels (default 12)
            - intensity (float): spot intensity magnitude (default 7.0). Positive -> brighten, Negative -> darken.
            - spot_sign (int): -1 (dark stains), +1 (bright), 0 random per-spot (default -1)
            - irregularity (float): 0..1, how irregular spots are (default 0.35)
            - blur_sigma (float): gaussian blur sigma for spot softening (default 1.2)
            - noise_sigma (float): global additive noise sigma (default 6.0)
            - base_color (tuple): RGB base paper color (default (245,245,220))
            - cap_spots (int): safety cap on spots (default 2000)

    Returns:
        PIL.Image (RGBA)
    """
    if params is None:
        params = {}
        
    # Validate sizes
    if not (isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0):
        raise ValueError("height and width must be positve integers")
    max_pixels = 80_000_000
    if height * width > max_pixels:
        raise ValueError("Requested image too large; reduce dimensions.")
    
    # params with defaults
    texture = float(params.get("texture", 1.0))
    spots = params.get("spots", None)
    base_spots = int(400 * texture)
    if spots is None:
        spots = max(1, base_spots)
    spots = min(spots, int(params.get("cap_spots", 2000)))
    
    min_radius = int(params.get("min_radius", 3))
    max_radius = int(params.get("max_radius", 12))
    intensity_base = float(params.get("intensity", 7.0)) # negative default = dark stains
    spot_sign = int(params.get("spots_sign", -1)) # -1 darken, +1 brighten, 0 random
    irregularity = float(params.get("irregualarity", 0.35))
    blur_sigma = float(params.get("blur_sigma", 1.2))
    noise_sigma = float(params.get("noise_sigma", 6.0))
    base_color = np.array(params.get("base_color", (245,245,220)), dtype=np.float32)
    cap_spots = int(params.get("cap_spots", 2000))
    
    spots = max(0, min(spots, cap_spots))
    
    # Canvas (float32)
    canvas = np.ones((height, width, 3), dtype=np.float32) * base_color[None, None, :]
    
    # Acculmulator for deltas
    delta_total = np.zeros_like(canvas, dtype=np.float32)
    
    for _ in range(spots):
        cx = int(np.random.randint(0, width))
        cy = int(np.random.randint(0, height))
        r = int(np.random.randint(min_radius, max_radius + 1))
        
        # bounding box
        y0 = max(0, cy - r)
        y1 = min(height, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(width, cx + r + 1)
        
        ph = y1 - y0
        pw = x1 - x0
        if ph <= 0 or pw <= 0:
            continue
        
        # local coords relative to center
        yy = np.arange(y0, y1)[:, None] - cy
        xx = np.arange(x0, x1)[None, :] - cx
        dist = np.sqrt(yy.astype(np.float32)**2 + xx.astype(np.float32)**2)
        
        # base radial mask (1 at center, 0 outside r)
        grain_factor = 1 + 0.3 * np.sin(xx * 0.5) * np.cos(yy * 0.3)
        mask = np.clip(1.0 - (dist / (r * grain_factor) + 1e-9), 0.0, 1.0)
        
        pixel_noise = np.random.uniform(1.0 - irregularity, 1.0 + irregularity, size=(ph, pw)).astype(np.float32)
        scale = 1.0 + irregularity * (np.random.uniform(-0.4, 0.4))
        mask *= (pixel_noise * scale)
        mask = np.clip(mask, 0.0, 1.0)
        
        # optionally blur local mask for softer edges
        if blur_sigma > 0:
            k = max(3, int(blur_sigma * 3) | 1)
            mask = cv2.GaussianBlur(mask, (k, k), blur_sigma)
            
        if spot_sign == 0:
            sign = int(np.random.choice([-1, 1]))
        else:
            sign = np.sign(spot_sign) if spot_sign != 0 else -1
            sign = int(sign)

        spot_intensity = sign * intensity_base * float(np.random.uniform(0.8, 1.2))
        
        delta_path = (mask[..., None]) * spot_intensity
        delta_total[y0:y1, x0:x1, :] += delta_path
        
    # apply accumulated deltas    
    canvas += delta_total
    
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas, mode="RGB").convert("RGBA")

img = parchment(512, 512, params = {"blur_sigma":0})
img1 = parchment1(512, 512)


img.show()
# img1.show()