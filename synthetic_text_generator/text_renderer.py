"""
Text rendering module for Sanskrit/Oriya text with various effects
"""

import os
import math
import random
import logging
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .backgrounds import create_enhanced_background

logger = logging.getLogger(__name__)


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
