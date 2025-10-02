import os
import argparse
import logging
from typing import Dict
from synthetic_text_generator import (
    ENHANCED_DEFAULT_PARAMS,
    generate_enhanced_sanskrit_samples,
    generate_comprehensive_dataset, 
    generate_ultra_realistic_samples,
    HuggingFaceDatasetProcessor
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced global parameters with all advanced features
ENHANCED_DEFAULT_PARAMS = {
    # Basic options
    'width': 400,
    'height': 320,
    'base_images': 1,

    # Font options
    'font_dir': '/content/static',
    'font': 'NotoSansOriya_ExtraCondensed-Regular.ttf',

    # Generation-level augmentations
    'noise': 0.7,
    'aging': 0.6,
    'texture': 0.7,
    'stains': 0.6,
    'stain_intensity': 0.5,

    # Word-level options
    'word_position': 0.6,
    'ink_color': 0.5,
    'line_spacing': 0.4,
    'baseline': 0.3,
    'word_angle': 0.0,

    # Post-processing options
    'apply_transforms': True,
    'all_transforms': False,
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
    'corner_displacement': 20,
    'morph_operation': 'mixed',
    'morph_kernel_size': 3,
    'aging_intensity': 0.5,
    'fiber_density': 0.5,
    'enable_advanced_effects': True,
    'advanced_effect_probability': 0.7,
    'shadow_angle': 45,
    'shadow_intensity': 0.4,
    'lens_distortion_strength': 0.2,
    'scanner_artifacts': True,
    'compression_quality': 85,
    'fold_probability': 0.4,
    'crease_probability': 0.3,
    'perspective_probability': 0.5,
    'shadow_probability': 0.6,

    # Performance parameters
    'use_multiprocessing': False,
    'num_processes': 4,
    'enable_caching': True,
    'debug_mode': False,
    'image_dir': ''
}

class EffectPlugin:
    """Base class for all image effects"""
    def __init__(self, name: str, params: Dict):
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

def safe_apply_effect(effect_func: Callable, image: np.ndarray, effect_name: str) -> np.ndarray:
    """Safely apply an effect with error handling"""
    try:
        return effect_func(image)
    except Exception as e:
        logger.error(f"Error applying {effect_name}: {e}")
        return image

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

def render_enhanced_sanskrit(text: str, font_path: str, output_path: str,
                           width: int, height: int, font_size: int,
                           style: str, ink_color: Tuple[int, int, int],
                           params: Dict) -> Optional[Image.Image]:
    """Enhanced Sanskrit text rendering with advanced features"""
    img = create_enhanced_background(width, height, style, params)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size)

        # Remove newlines and treat all text as one block
        words = text.strip().replace('\n', ' ').split()

        y_position = random.randint(25, 75)
        margin = 25

        # Available width for text
        available_width = width - 2 * margin
        space_width = draw.textlength(" ", font=font)

        current_line = []
        current_line_width = 0

        # Collect all lines first
        all_lines = []
        for word in words:
            word_width = draw.textlength(word, font=font)

            # Check if adding this word would exceed available width
            if current_line and current_line_width + space_width + word_width > available_width:
                all_lines.append(current_line)
                current_line = [word]
                current_line_width = word_width
            else:
                if current_line:
                    current_line_width += space_width + word_width
                else:
                    current_line_width = word_width
                current_line.append(word)

        # Add the last line if there's anything left
        if current_line:
            all_lines.append(current_line)

        # Render all lines with enhanced effects
        for line in all_lines:
            # Center the line horizontally
            line_text = " ".join(line)
            line_width = draw.textlength(line_text, font=font)
            x_position = (width - line_width) // 2

            baseline_offset = random.randint(-2, 2) * params["baseline"]
            y_line_position = y_position + baseline_offset

            # Check if we've reached the bottom of the image
            if y_line_position + font_size > height - margin:
                break

            # Render each word in the line with enhanced effects
            x_word_position = x_position
            for word in line:
                word_x_offset = int(random.uniform(-1.5, 1.5) * params["word_position"])
                word_y_offset = int(random.uniform(-1, 1) * params["word_position"])

                # Enhanced color variation
                color_variation = int(random.randint(-3, 3) * params["ink_color"])
                word_color = (
                    np.clip(ink_color[0] + color_variation, 0, 255),
                    np.clip(ink_color[1] + color_variation, 0, 255),
                    np.clip(ink_color[2] + color_variation, 0, 255)
                )

                word_width = draw.textlength(word, font=font)
                word_height = font_size * 1.2

                if params["word_angle"] > 0:
                    # Apply rotation to individual word
                    word_angle = random.uniform(-2, 2) * params["word_angle"]

                    diagonal = math.sqrt(word_width**2 + word_height**2)
                    padding = int(diagonal * 0.5)

                    temp_width = int(diagonal + 2 * padding)
                    temp_height = int(diagonal + 2 * padding)
                    txt_img = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
                    txt_d = ImageDraw.Draw(txt_img)

                    center_x = temp_width // 2 - word_width // 2
                    center_y = temp_height // 2 - word_height // 2
                    txt_d.text((center_x, center_y), word, font=font, fill=word_color + (255,))

                    rotated = txt_img.rotate(word_angle, resample=Image.BICUBIC, expand=0,
                                           center=(temp_width//2, temp_height//2))

                    paste_x = int(x_word_position + word_x_offset - padding)
                    paste_y = int(y_line_position + word_y_offset - padding)

                    img.paste(rotated, (paste_x, paste_y), rotated)
                else:
                    draw.text(
                        (x_word_position + word_x_offset, y_line_position + word_y_offset),
                        word, fill=word_color, font=font
                    )

                x_word_position += word_width + space_width

            # Move to next line
            line_spacing_factor = 1.0 + (random.uniform(-0.1, 0.1) * params["line_spacing"])
            y_position += int(font_size * 1.2 * line_spacing_factor)

        if output_path is not None:
            img.save(output_path)
            logger.info(f"Saved rendered Sanskrit to {output_path}")

        return img

    except Exception as e:
        logger.error(f"Error rendering text with font {font_path}: {e}")
        return None

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

def main():
    """Enhanced main function with Colab-compatible argument parsing"""
    sanskrit_text = """
    କବି ସମ୍ରାଟ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ (୧୬୭୦ - ୧୭୪୦, ଅନ୍ୟମତେ ୧୭୨୦) ସପ୍ତଦଶ ଶତାବ୍ଦୀର ପ୍ରମୁଖ ଓଡ଼ିଶୀ ସଙ୍ଗୀତକାର କବି ଥିଲେ । ସେ ସପ୍ତଦଶ ଶତାବ୍ଦୀର ଶେଷ ପର୍ଯ୍ୟାୟରେ ପୁରାତନ ରାଜଶାସିତ ଘୁମୁସରର ରାଜବଂଶରେ କୁଲାଡ଼ଗଡଜନ୍ମଲାଭ କରିଥିଲେ। ରୀତିଯୁଗର ତାଙ୍କର ପାଣ୍ଡିତ୍ୟପୂର୍ଣ୍ଣ ସାହିତ୍ୟ କୃତି ପାଇଁ ତାଙ୍କୁ କବି ସମ୍ରାଟ ଉପାଧିରେ ଭୂଷିତ କରାଯାଇଅଛି । ଓଡ଼ିଆ ସାହିତ୍ୟରେ ଥିବା ଅସଂଖ୍ୟ କବିଙ୍କ ମଧ୍ୟରେ ସେ ଅସାଧାରଣ ପ୍ରସିଦ୍ଧି ଲାଭ କରିଅଛନ୍ତି । ଏକ ରାଜ ପରିବାରରେ ଜନ୍ମିତ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ ରାଜପଦଠାରୁ ଦୂରରେ ରହି ଓଡ଼ିଆ ସାହିତ୍ୟରେ ମନୋନିବେଶ କରିଥିଲେ ।
    """

    parser = argparse.ArgumentParser(description='Enhanced Sanskrit text generation with advanced image effects')

    # Basic options
    basic = parser.add_argument_group('Basic Options')
    basic.add_argument('--output-dir', type=str, default='data/enhanced_synthetic/images',
                      help='Output directory for generated images')
    basic.add_argument('--width', type=int, default=ENHANCED_DEFAULT_PARAMS['width'],
                      help='Width of output images')
    basic.add_argument('--height', type=int, default=ENHANCED_DEFAULT_PARAMS['height'],
                      help='Height of output images')
    basic.add_argument('--base-images', type=int, default=ENHANCED_DEFAULT_PARAMS['base_images'],
                      help='Total number of base images to generate')

    # Font options
    font = parser.add_argument_group('Font Options')
    font.add_argument('--font-dir', type=str, default=ENHANCED_DEFAULT_PARAMS['font_dir'],
                     help='Directory containing font files')
    font.add_argument('--font', type=str, default=ENHANCED_DEFAULT_PARAMS['font'],
                     help='Font filename within the font directory')

    # Generation-level augmentations
    gen = parser.add_argument_group('Generation-Level Augmentations')
    gen.add_argument('--noise', type=float, default=ENHANCED_DEFAULT_PARAMS['noise'],
                    help='Background noise intensity (0.0-1.0)')
    gen.add_argument('--aging', type=float, default=ENHANCED_DEFAULT_PARAMS['aging'],
                    help='Edge aging effect (0.0-1.0)')
    gen.add_argument('--texture', type=float, default=ENHANCED_DEFAULT_PARAMS['texture'],
                    help='Texture variation (0.0-1.0)')
    gen.add_argument('--stains', type=float, default=ENHANCED_DEFAULT_PARAMS['stains'],
                    help='Number of stains (0.0-1.0)')
    gen.add_argument('--stain-intensity', type=float, default=ENHANCED_DEFAULT_PARAMS['stain_intensity'],
                    help='Intensity of stain effects (0.0-1.0)')
    gen.add_argument('--fiber-density', type=float, default=ENHANCED_DEFAULT_PARAMS['fiber_density'],
                    help='Paper fiber texture density (0.0-1.0)')
    gen.add_argument('--image-dir', type=str, default=ENHANCED_DEFAULT_PARAMS['image_dir'],
                    help='Directory to randomly sample background images from')

    # Word-level options
    word = parser.add_argument_group('Word-Level Options')
    word.add_argument('--word-position', type=float, default=ENHANCED_DEFAULT_PARAMS['word_position'],
                     help='Random word position variation (0.0-1.0)')
    word.add_argument('--ink-color', type=float, default=ENHANCED_DEFAULT_PARAMS['ink_color'],
                     help='Ink color variation (0.0-1.0)')
    word.add_argument('--line-spacing', type=float, default=ENHANCED_DEFAULT_PARAMS['line_spacing'],
                     help='Random line spacing (0.0-1.0)')
    word.add_argument('--baseline', type=float, default=ENHANCED_DEFAULT_PARAMS['baseline'],
                     help='Baseline wobble effect (0.0-1.0)')
    word.add_argument('--word-angle', type=float, default=ENHANCED_DEFAULT_PARAMS['word_angle'],
                     help='Random word angle (0.0-1.0)')

    # Post-processing options
    post = parser.add_argument_group('Post-Processing Augmentations')
    post.add_argument('--no-transforms', dest='apply_transforms', action='store_false',
                     help='Disable post-processing transforms')
    post.add_argument('--all-transforms', action='store_true',
                     help='Apply all transforms instead of random subset')
    post.add_argument('--rotation-max', type=float, default=ENHANCED_DEFAULT_PARAMS['rotation_max'],
                     help='Maximum rotation angle in degrees')
    post.add_argument('--brightness-var', type=float, default=ENHANCED_DEFAULT_PARAMS['brightness_var'],
                     help='Brightness variation factor (0.0-1.0)')
    post.add_argument('--contrast-var', type=float, default=ENHANCED_DEFAULT_PARAMS['contrast_var'],
                     help='Contrast variation factor (0.0-1.0)')
    post.add_argument('--noise-min', type=float, default=ENHANCED_DEFAULT_PARAMS['noise_min'],
                     help='Minimum noise intensity for transforms')
    post.add_argument('--noise-max', type=float, default=ENHANCED_DEFAULT_PARAMS['noise_max'],
                     help='Maximum noise intensity for transforms')
    post.add_argument('--blur-min', type=float, default=ENHANCED_DEFAULT_PARAMS['blur_min'],
                     help='Minimum blur radius')
    post.add_argument('--blur-max', type=float, default=ENHANCED_DEFAULT_PARAMS['blur_max'],
                     help='Maximum blur radius')

    # Advanced effects options
    advanced = parser.add_argument_group('Advanced Effects')
    advanced.add_argument('--no-advanced-effects', dest='enable_advanced_effects', action='store_false',
                         help='Disable advanced effects')
    advanced.add_argument('--fold-intensity', type=float, default=ENHANCED_DEFAULT_PARAMS['fold_intensity'],
                         help='Intensity of fold/crease effects (0.0-1.0)')
    advanced.add_argument('--bleed-intensity', type=float, default=ENHANCED_DEFAULT_PARAMS['bleed_intensity'],
                         help='Intensity of ink bleeding effects (0.0-1.0)')
    advanced.add_argument('--bleed-radius', type=int, default=ENHANCED_DEFAULT_PARAMS['bleed_radius'],
                         help='Radius of ink bleeding effect')
    advanced.add_argument('--corner-displacement', type=int, default=ENHANCED_DEFAULT_PARAMS['corner_displacement'],
                         help='Maximum corner displacement for perspective distortion')
    advanced.add_argument('--shadow-intensity', type=float, default=ENHANCED_DEFAULT_PARAMS['shadow_intensity'],
                         help='Intensity of shadow effects (0.0-1.0)')
    advanced.add_argument('--shadow-angle', type=float, default=ENHANCED_DEFAULT_PARAMS['shadow_angle'],
                         help='Angle of shadow effects in degrees')
    advanced.add_argument('--lens-distortion-strength', type=float, default=ENHANCED_DEFAULT_PARAMS['lens_distortion_strength'],
                         help='Strength of lens distortion effects (0.0-1.0)')
    advanced.add_argument('--no-scanner-artifacts', dest='scanner_artifacts', action='store_false',
                         help='Disable scanner artifact simulation')
    advanced.add_argument('--compression-quality', type=int, default=ENHANCED_DEFAULT_PARAMS['compression_quality'],
                         help='JPEG compression quality (1-100)')

    # Probability controls
    prob = parser.add_argument_group('Effect Probabilities')
    prob.add_argument('--advanced-effect-probability', type=float, default=ENHANCED_DEFAULT_PARAMS['advanced_effect_probability'],
                     help='Probability of applying advanced effects (0.0-1.0)')
    prob.add_argument('--fold-probability', type=float, default=ENHANCED_DEFAULT_PARAMS['fold_probability'],
                     help='Probability of applying fold effects (0.0-1.0)')
    prob.add_argument('--perspective-probability', type=float, default=ENHANCED_DEFAULT_PARAMS['perspective_probability'],
                     help='Probability of applying perspective distortion (0.0-1.0)')
    prob.add_argument('--shadow-probability', type=float, default=ENHANCED_DEFAULT_PARAMS['shadow_probability'],
                     help='Probability of applying shadow effects (0.0-1.0)')

    # Performance options
    perf = parser.add_argument_group('Performance Options')
    perf.add_argument('--use-multiprocessing', action='store_true',
                     help='Enable multiprocessing for batch generation')
    perf.add_argument('--num-processes', type=int, default=ENHANCED_DEFAULT_PARAMS['num_processes'],
                     help='Number of processes for multiprocessing')
    perf.add_argument('--debug-mode', action='store_true',
                     help='Enable debug mode with verbose logging')

    # Set defaults
    parser.set_defaults(
        apply_transforms=ENHANCED_DEFAULT_PARAMS['apply_transforms'],
        all_transforms=ENHANCED_DEFAULT_PARAMS['all_transforms'],
        enable_advanced_effects=ENHANCED_DEFAULT_PARAMS['enable_advanced_effects'],
        scanner_artifacts=ENHANCED_DEFAULT_PARAMS['scanner_artifacts']
    )

    # FIX FOR GOOGLE COLAB: Handle Jupyter kernel arguments
    import sys

    # Check if we're running in a Jupyter environment
    def is_jupyter():
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    if is_jupyter():
        # In Jupyter/Colab, parse empty args to use defaults
        args = parser.parse_args([])
        logger.info("Running in Jupyter environment - using default parameters")
    else:
        # Normal command line usage
        args = parser.parse_args()

    # Set up logging level
    if args.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)

    # Convert args to dict, excluding output_dir
    params = {k: v for k, v in vars(args).items() if k != 'output_dir'}

    logger.info("Starting enhanced Sanskrit text generation with advanced effects")
    logger.info(f"Parameters: {params}")

    # Generate enhanced samples
    try:
        generate_enhanced_sanskrit_samples(
            text=sanskrit_text,
            font_path=os.path.join(params['font_dir'], params['font']),
            output_dir=args.output_dir,
            params=params
        )
        logger.info("Enhanced Sanskrit text generation completed successfully")
    except Exception as e:
        logger.error(f"Error in enhanced generation: {e}")
        raise

# Alternative: Simple function for direct Colab usage
def generate_colab_samples(base_images=5, width=400, height=320, enable_advanced_effects=True):
    """Simplified function for direct use in Google Colab"""
    sanskrit_text = """
    କବି ସମ୍ରାଟ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ (୧୬୭୦ - ୧୭୪୦, ଅନ୍ୟମତେ ୧୭୨୦) ସପ୍ତଦଶ ଶତାବ୍ଦୀର ପ୍ରମୁଖ ଓଡ଼ିଶୀ ସଙ୍ଗୀତକାର କବି ଥିଲେ । ସେ ସପ୍ତଦଶ ଶତାବ୍ଦୀର ଶେଷ ପର୍ଯ୍ୟାୟରେ ପୁରାତନ ରାଜଶାସିତ ଘୁମୁସରର ରାଜବଂଶରେ କୁଲାଡ଼ଗଡଜନ୍ମଲାଭ କରିଥିଲେ। ରୀତିଯୁଗର ତାଙ୍କର ପାଣ୍ଡିତ୍ୟପୂର୍ଣ୍ଣ ସାହିତ୍ୟ କୃତି ପାଇଁ ତାଙ୍କୁ କବି ସମ୍ରାଟ ଉପାଧିରେ ଭୂଷିତ କରାଯାଇଅଛି । ଓଡ଼ିଆ ସାହିତ୍ୟରେ ଥିବା ଅସଂଖ୍ୟ କବିଙ୍କ ମଧ୍ୟରେ ସେ ଅସାଧାରଣ ପ୍ରସିଦ୍ଧି ଲାଭ କରିଅଛନ୍ତି । ଏକ ରାଜ ପରିବାରରେ ଜନ୍ମିତ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ ରାଜପଦଠାରୁ ଦୂରରେ ରହି ଓଡ଼ିଆ ସାହିତ୍ୟରେ ମନୋନିବେଶ କରିଥିଲେ ।
    """

    # Create custom parameters
    params = ENHANCED_DEFAULT_PARAMS.copy()
    params.update({
        'base_images': base_images,
        'width': width,
        'height': height,
        'enable_advanced_effects': enable_advanced_effects,
        'output_dir': 'colab_output'
    })

    # Create output directory
    import os
    os.makedirs(params['output_dir'], exist_ok=True)

    logger.info(f"Generating {base_images} Sanskrit samples with advanced effects: {enable_advanced_effects}")

    # Generate samples
    images = generate_enhanced_sanskrit_samples(
        text=sanskrit_text,
        font_path=os.path.join(params['font_dir'], params['font']),
        output_dir=params['output_dir'],
        params=params
    )

    logger.info(f"Generated samples saved to: {params['output_dir']}")
    return images

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
    import itertools
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

# Updated main function for comprehensive generation
def main():
    """Enhanced main function with comprehensive effect usage"""
    sanskrit_text = """
    କବি ସମ୍ରାଟ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ (୧୬୭୦ - ୧୭୪୦, ଅନ୍ୟମତେ ୧୭୨୦) ସପ୍ତଦଶ ଶତାବ୍ଦୀର ପ୍ରମୁଖ ଓଡ଼ିଶୀ ସଙ୍ଗୀତକାର କବି ଥିଲେ । ସେ ସପ୍ତଦଶ ଶତାବ୍ଦୀର ଶେଷ ପର୍ଯ୍ୟାୟରେ ପୁରାତନ ରାଜଶାସିତ ଘୁମୁସରର ରାଜବଂଶରେ କୁଲାଡ଼ଗଡଜନ୍ମଲାଭ କରିଥିଲେ। ରୀତିଯୁଗର ତାଙ୍କର ପାଣ୍ଡିତ୍ୟପୂର୍ଣ୍ଣ সাহିত্ୟ କୃତি ପାଇଁ ତାଙ୍କୁ କବି ସମ୍ରାଟ ଉପାଧିରେ ଭୂଷିତ କରାଯାଇଅଛି । ଓଡ଼ିଆ ସାହିତ୍ୟରେ ଥିବା ଅସଂଖ୍ୟ କବିଙ୍କ ମଧ୍ୟରେ ସେ ଅସାଧାରଣ ପ୍ରସିଦ୍ଧି ଲାଭ କରିଅଛନ୍ତି । ଏକ ରାଜ ପରିବାରରେ ଜନ୍ମିତ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ ରାଜପଦଠାରୁ ଦୂରରେ ରହି ଓଡ଼ିଆ ସାହିତ୍ୟରେ ମନୋନିବେଶ କରିଥିଲେ ।
    """

    # Check if running in Jupyter environment
    def is_jupyter():
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    # For Colab/Jupyter usage
    if is_jupyter():
        logger.info("Running in Jupyter environment - generating comprehensive dataset")

        # Generate comprehensive dataset
        logger.info("Generating comprehensive dataset with all effects...")
        comprehensive_images = generate_comprehensive_dataset(
            text=sanskrit_text,
            output_dir='comprehensive_output',
            params=ENHANCED_DEFAULT_PARAMS
        )

        # Generate ultra-realistic samples
        logger.info("Generating ultra-realistic samples...")
        ultra_realistic_images = generate_ultra_realistic_samples(
            text=sanskrit_text,
            output_dir='ultra_realistic_output',
            params=ENHANCED_DEFAULT_PARAMS
        )

        logger.info(f"Generated {len(comprehensive_images)} comprehensive images")
        logger.info(f"Generated {len(ultra_realistic_images)} ultra-realistic images")

        # Display first few samples
        if comprehensive_images:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Generated Sanskrit Text Samples with All Effects', fontsize=16)

            for i, ax in enumerate(axes.flat):
                if i < len(comprehensive_images):
                    ax.imshow(comprehensive_images[i])
                    ax.axis('off')
                    ax.set_title(f'Sample {i+1}')

            plt.tight_layout()
            plt.show()

        return comprehensive_images, ultra_realistic_images

    else:
        # Command line usage with argument parsing
        parser = argparse.ArgumentParser(description='Comprehensive Sanskrit text generation')
        parser.add_argument('--mode', choices=['comprehensive', 'ultra-realistic', 'both'],
                          default='comprehensive', help='Generation mode')
        parser.add_argument('--output-dir', type=str, default='output',
                          help='Output directory')
        parser.add_argument('--style-focus', type=str, choices=['lined_paper', 'old_paper', 'birch', 'parchment'],
                          help='Focus on specific style')

        args = parser.parse_args()

        if args.mode in ['comprehensive', 'both']:
            comprehensive_images = generate_comprehensive_dataset(
                text=sanskrit_text,
                output_dir=f"{args.output_dir}/comprehensive",
                params=ENHANCED_DEFAULT_PARAMS
            )

        if args.mode in ['ultra-realistic', 'both']:
            ultra_realistic_images = generate_ultra_realistic_samples(
                text=sanskrit_text,
                output_dir=f"{args.output_dir}/ultra_realistic",
                style_focus=args.style_focus,
                params=ENHANCED_DEFAULT_PARAMS
            )

        logger.info("Generation completed successfully")

# Colab-specific functions
def generate_all_effects_colab(base_images=4, width=600, height=400):
    """Generate samples using all effects systematically - Colab optimized"""
    sanskrit_text = """
    କବି ସମ୍ରାଟ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ (୧୬୭୦ - ୧୭୪୦, ଅନ୍ୟମତେ ୧୭୨୦) ସପ୍ତଦଶ ଶତାବ୍ଦୀର ପ୍ରମୁଖ ଓଡ଼ିଶୀ ସଙ୍ଗୀତକାର କବି ଥିଲେ । ସେ ସପ୍ତଦଶ ଶତାବ୍ଦୀର ଶେଷ ପର୍ଯ୍ୟାୟରେ ପୁରାତନ ରାଜଶାସିତ ଘୁମୁସରର ରାଜବଂଶରେ କୁଲାଡ଼ଗଡଜନ୍ମଲାଭ କରିଥିଲେ। ରୀତିଯୁଗର ତାଙ୍କର ପାଣ୍ଡିତ୍ୟପୂର୍ଣ୍ଣ ସাହିত্ୟ କୃତি ପାଇଁ ତାଙ୍କୁ କବି ସମ୍ରାଟ ଉପାଧିରେ ଭୂଷିତ କରାଯାଇଅଛି ।
    """

    params = ENHANCED_DEFAULT_PARAMS.copy()
    params.update({
        'width': width,
        'height': height,
        'base_images': base_images
    })

    # Generate comprehensive dataset
    logger.info("Generating comprehensive dataset with all effects...")
    comprehensive_images = generate_comprehensive_dataset(
        text=sanskrit_text,
        output_dir='colab_comprehensive_output',
        params=params
    )

    # Generate ultra-realistic samples
    logger.info("Generating ultra-realistic samples...")
    ultra_realistic_images = generate_ultra_realistic_samples(
        text=sanskrit_text,
        output_dir='colab_ultra_realistic_output',
        params=params
    )

    logger.info(f"Generated {len(comprehensive_images)} comprehensive images")
    logger.info(f"Generated {len(ultra_realistic_images)} ultra-realistic images")

    return comprehensive_images, ultra_realistic_images

if __name__ == "__main__":
    main()