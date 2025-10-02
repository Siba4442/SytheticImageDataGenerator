"""
Advanced image effects for synthetic text generation
"""

import cv2
import numpy as np
import random
import logging
from typing import List, Tuple
from noise import pnoise2

logger = logging.getLogger(__name__)


class EffectPlugin:
    """Base class for all image effects"""
    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params
        self.validate_params()

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the effect to the image"""
        raise NotImplementedError

    def validate_params(self):
        """Validate effect parameters"""
        pass


class AdvancedImageEffects:
    """Collection of advanced image processing effects"""

    @staticmethod
    def generate_perlin_noise(width: int, height: int, scale: float = 0.1, octaves: int = 4) -> np.ndarray:
        """Generate Perlin noise for realistic textures"""
        noise_map = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                noise_map[i][j] = pnoise2(i * scale, j * scale, octaves=octaves)
        return noise_map

    @staticmethod
    def simulate_paper_fiber_texture(width: int, height: int, fiber_density: float = 0.5) -> np.ndarray:
        """Generate realistic paper fiber texture using multi-octave Perlin noise"""
        try:
            # Base texture
            base_texture = AdvancedImageEffects.generate_perlin_noise(width, height, 0.02, 4)

            # Fine fiber details
            fine_texture = AdvancedImageEffects.generate_perlin_noise(width, height, 0.1, 2)

            # Combine textures
            combined = base_texture * 0.7 + fine_texture * 0.3
            combined = ((combined + 1) / 2) * fiber_density * 20

            # Convert to 3-channel
            texture = np.stack([combined, combined, combined], axis=2)
            return texture.astype(np.uint8)
        except Exception as e:
            logger.warning(f"Failed to generate Perlin noise texture: {e}")
            # Fallback to simple noise
            return np.random.randint(0, int(20 * fiber_density), (height, width, 3), dtype=np.uint8)

    @staticmethod
    def simulate_fold_crease(image: np.ndarray, fold_lines: List[Tuple], fold_intensity: float = 0.5) -> np.ndarray:
        """Create realistic paper fold effects with shadow casting"""
        try:
            height, width = image.shape[:2]
            result = image.copy()

            for fold_line in fold_lines:
                # Create distance map from fold line
                y_coords, x_coords = np.ogrid[:height, :width]

                # Calculate distance from fold line
                x1, y1, x2, y2 = fold_line
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                if line_length > 0:
                    # Distance from point to line
                    distances = np.abs((y2 - y1) * x_coords - (x2 - x1) * y_coords + x2 * y1 - y2 * x1) / line_length

                    # Create fold profile with Gaussian falloff
                    fold_width = min(width, height) * 0.1
                    fold_profile = np.exp(-0.5 * (distances / fold_width)**2)

                    # Apply fold effect
                    fold_effect = fold_profile * fold_intensity * 40

                    # Create shadow on one side
                    shadow_mask = (y_coords - y1) * (x2 - x1) - (x_coords - x1) * (y2 - y1) > 0
                    shadow_effect = fold_profile * shadow_mask * fold_intensity * 20

                    # Apply effects
                    result = result.astype(np.float32)
                    result[:, :, 0] -= fold_effect + shadow_effect
                    result[:, :, 1] -= fold_effect + shadow_effect
                    result[:, :, 2] -= fold_effect + shadow_effect
                    result = np.clip(result, 0, 255).astype(np.uint8)

            return result
        except Exception as e:
            logger.error(f"Error in fold/crease simulation: {e}")
            return image

    @staticmethod
    def simulate_ink_bleed(image: np.ndarray, bleed_intensity: float = 0.3, bleed_radius: int = 3) -> np.ndarray:
        """Simulate ink bleeding using morphological operations"""
        try:
            # Convert to grayscale to detect text
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Threshold to find text regions
            _, text_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            # Create bleeding kernel
            kernel_size = max(1, int(bleed_radius * 2 + 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # Apply morphological dilation for bleeding effect
            bleeding_mask = cv2.dilate(text_mask, kernel, iterations=1)

            # Create bleed effect
            bleed_effect = cv2.GaussianBlur(bleeding_mask.astype(np.float32), (kernel_size, kernel_size), 0)
            bleed_effect = bleed_effect * bleed_intensity / 255.0

            # Apply bleeding to image
            result = image.copy().astype(np.float32)
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - bleed_effect) + (result[:, :, c] * 0.7) * bleed_effect

            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in ink bleed simulation: {e}")
            return image

    @staticmethod
    def apply_perspective_distortion(image: np.ndarray, corner_displacement: int = 20) -> np.ndarray:
        """Apply realistic perspective distortion to simulate camera angles"""
        try:
            height, width = image.shape[:2]

            # Define source points (corners of the image)
            src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

            # Add random displacement to corners
            dst_points = src_points.copy()
            for i in range(4):
                dst_points[i][0] += random.randint(-corner_displacement, corner_displacement)
                dst_points[i][1] += random.randint(-corner_displacement, corner_displacement)

            # Ensure points are within reasonable bounds
            dst_points[:, 0] = np.clip(dst_points[:, 0], -width*0.1, width*1.1)
            dst_points[:, 1] = np.clip(dst_points[:, 1], -height*0.1, height*1.1)

            # Calculate perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # Apply transformation
            result = cv2.warpPerspective(image, matrix, (width, height),
                                       borderMode=cv2.BORDER_REPLICATE)

            return result
        except Exception as e:
            logger.error(f"Error in perspective distortion: {e}")
            return image

    @staticmethod
    def apply_shadow_effects(image: np.ndarray, shadow_angle: float = 45, shadow_intensity: float = 0.4) -> np.ndarray:
        """Apply realistic shadow effects"""
        try:
            height, width = image.shape[:2]
            result = image.copy().astype(np.float32)

            # Create shadow gradient
            angle_rad = np.radians(shadow_angle)
            x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

            # Calculate shadow based on angle
            shadow_factor = (np.cos(angle_rad) * x_coords / width +
                           np.sin(angle_rad) * y_coords / height)
            shadow_factor = np.clip(shadow_factor, 0, 1)

            # Apply shadow
            shadow_effect = 1 - shadow_factor * shadow_intensity
            for c in range(3):
                result[:, :, c] *= shadow_effect

            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in shadow effects: {e}")
            return image

    @staticmethod
    def apply_morphological_operations(image: np.ndarray, operation: str = 'mixed', kernel_size: int = 3) -> np.ndarray:
        """Apply morphological operations for text degradation"""
        try:
            # Convert to grayscale for morphological operations
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Create kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # Apply operations based on type
            if operation == 'erosion':
                processed = cv2.erode(gray, kernel, iterations=1)
            elif operation == 'dilation':
                processed = cv2.dilate(gray, kernel, iterations=1)
            elif operation == 'opening':
                processed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            elif operation == 'closing':
                processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            else:  # mixed
                # Randomly choose operation
                ops = ['erosion', 'dilation', 'opening', 'closing']
                chosen_op = random.choice(ops)
                return AdvancedImageEffects.apply_morphological_operations(image, chosen_op, kernel_size)

            # Convert back to color
            result = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            return result
        except Exception as e:
            logger.error(f"Error in morphological operations: {e}")
            return image

    @staticmethod
    def simulate_scanner_artifacts(image: np.ndarray, compression_quality: int = 85) -> np.ndarray:
        """Simulate scanner artifacts and compression effects"""
        try:
            # Add horizontal scanning lines
            height, width = image.shape[:2]
            result = image.copy()

            # Add periodic horizontal lines (scanning artifacts)
            for y in range(0, height, random.randint(8, 15)):
                intensity = random.randint(5, 15)
                if y < height:
                    result[y, :, :] = np.clip(result[y, :, :] - intensity, 0, 255)

            # Add dust spots
            dust_count = random.randint(3, 8)
            for _ in range(dust_count):
                x = random.randint(0, width - 5)
                y = random.randint(0, height - 5)
                size = random.randint(2, 5)
                dust_intensity = random.randint(20, 40)
                result[y:y+size, x:x+size, :] = np.clip(result[y:y+size, x:x+size, :] - dust_intensity, 0, 255)

            # Simulate JPEG compression artifacts
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]
            _, encimg = cv2.imencode('.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR), encode_param)
            result = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            return result
        except Exception as e:
            logger.error(f"Error in scanner artifacts: {e}")
            return image

    @staticmethod
    def apply_lens_distortion(image: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """Apply lens distortion effects"""
        try:
            height, width = image.shape[:2]

            # Create distortion map
            center_x, center_y = width // 2, height // 2
            y_coords, x_coords = np.ogrid[:height, :width]

            # Calculate distance from center
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)

            # Normalize distances
            normalized_distances = distances / max_distance

            # Apply barrel distortion
            distortion_factor = 1 + strength * normalized_distances**2

            # Create mapping
            map_x = ((x_coords - center_x) / distortion_factor + center_x).astype(np.float32)
            map_y = ((y_coords - center_y) / distortion_factor + center_y).astype(np.float32)

            # Apply distortion
            result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

            return result
        except Exception as e:
            logger.error(f"Error in lens distortion: {e}")
            return image


def generate_random_fold_lines(image_size: Tuple[int, int], num_folds: int = None) -> List[Tuple]:
    """Generate random fold lines for crease effects"""
    width, height = image_size

    if num_folds is None:
        num_folds = random.randint(1, 3)

    fold_lines = []
    for _ in range(num_folds):
        # Random fold line coordinates
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        fold_lines.append((x1, y1, x2, y2))

    return fold_lines


def safe_apply_effect(effect_func, image: np.ndarray, effect_name: str) -> np.ndarray:
    """Safely apply an effect with error handling"""
    try:
        return effect_func(image)
    except Exception as e:
        logger.error(f"Error applying {effect_name}: {e}")
        return image
