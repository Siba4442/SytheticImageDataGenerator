from math import pi

import cv2
import numpy as np
from PIL import Image


def cylindrical_edge_warp(
    pil_img: Image.Image, side: str = "left", strength: float = 0.6, warp_portion: float = 0.45
) -> Image.Image:
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
            strip = (w - W) < X
            dx = X[strip] - (w - W)

        # angle on cylinder surface for those pixels
        theta = dx / R
        # horizontal mapping
        displacement = R * np.sin(theta) - dx
        map_x[strip] += displacement

        # vertical scaling
        scale_y = np.cos(theta)
        map_y[strip] = (Y[strip] - h / 2) / scale_y + h / 2

        warped = cv2.remap(
            img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Error in cylindrical_edge_warp: {e}")
        return pil_img


def washboard_warp(
    pil_img: Image.Image,
    amplitude: float = 8,
    wavelength: float = 120,
    phase: float = 0.0,
    decay_from_top: bool = True,
) -> Image.Image:
    """Enhanced washboard warp with better error handling"""
    try:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # build a vector of vertical offsets
        x = np.arange(w, dtype=np.float32)
        dy = amplitude * np.sin(2 * pi * x / wavelength + phase)

        atten = np.linspace(1, 0.2, h, dtype=np.float32)[:, None] if decay_from_top else 1.0

        # broadcast to full map
        map_x, map_y = np.meshgrid(x, np.arange(h, dtype=np.float32))
        map_y += dy * atten

        warped = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Error in washboard_warp: {e}")
        return pil_img
