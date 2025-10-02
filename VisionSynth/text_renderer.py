import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

def render_text(
    text: str, 
    font_path: str, 
    width: int, 
    height: int, 
    font_size: int,
    img: Image.Image,
    line_spacing: float = 1.0
) -> Optional[Image.Image]:
    
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
        
        words = text.strip().replace('\n', ' ').split()
        
        y_position = np.random.randint(25, 75)
        margin = 25
        
        available_width = width - 2 * margin
        space_width = draw.textlength(" ", font=font)
        
        current_line = []
        current_line_width = 0
        
        all_lines = []
        for word in words:
            word_width = draw.textlength(word, font=font)
            
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
                    
            if current_line:
                all_lines.append(current_line)
                
        for line in all_lines:
            line_text = " ".join(line)
            line_width = draw.textlength(line_text, font=font)
            x_position = (width - line_width) // 2
            
            if y_position + font_size > height - margin:
                break
            
            for word in line:
                word_width = draw.textlength(word, font=font)
                draw.text((x_position, y_position), word, font=font)
                x_position += word_width + space_width

            line_spacing_factor = 1.0 + np.random.uniform(-0.1, 1.0) * line_spacing
            y_position += int(font_size * 1.2 * line_spacing_factor)

        return img

    except Exception as e:
        print(f"Error rendering text: {e}")
        return None