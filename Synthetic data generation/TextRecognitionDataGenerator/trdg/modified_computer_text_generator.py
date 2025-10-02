import random as rnd
from typing import Tuple, List
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from trdg.utils import get_text_width, get_text_height

# Thai Unicode reference: https://jrgraphix.net/r/Unicode/0E00-0E7F
TH_TONE_MARKS = [
    "0xe47",
    "0xe48",
    "0xe49",
    "0xe4a",
    "0xe4b",
    "0xe4c",
    "0xe4d",
    "0xe4e",
]
TH_UNDER_VOWELS = ["0xe38", "0xe39", "\0xe3A"]
TH_UPPER_VOWELS = ["0xe31", "0xe34", "0xe35", "0xe36", "0xe37"]


def generate(
    text: str,
    font: str,
    text_color: str,
    font_size: int,
    orientation: int,
    space_width: int,
    character_spacing: int,
    fit: bool,
    word_split: bool,
    stroke_width: int = 0,
    stroke_fill: str = "#282828",
    max_width: int = None,  # NEW: Maximum width for line wrapping
    line_spacing: int = 5,  # NEW: Spacing between lines
) -> Tuple:
    if orientation == 0:
        return _generate_horizontal_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            word_split,
            stroke_width,
            stroke_fill,
            max_width,
            line_spacing,
        )
    elif orientation == 1:
        return _generate_vertical_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            stroke_width,
            stroke_fill,
        )
    else:
        raise ValueError("Unknown orientation " + str(orientation))


def _compute_character_width(image_font: ImageFont, character: str) -> int:
    if len(character) == 1 and (
        "{0:#x}".format(ord(character))
        in TH_TONE_MARKS + TH_UNDER_VOWELS + TH_UNDER_VOWELS + TH_UPPER_VOWELS
    ):
        return 0
    # Casting as int to preserve the old behavior
    return round(image_font.getlength(character))


def _wrap_text_to_lines(
    text: str, 
    image_font: ImageFont, 
    max_width: int, 
    space_width: int, 
    character_spacing: int,
    word_split: bool
) -> List[str]:
    """NEW FUNCTION: Wrap text into multiple lines based on max_width"""
    if max_width is None:
        return [text]
    
    words = text.split(' ')
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        # Calculate word width including spaces and character spacing
        if word_split:
            word_width = sum(_compute_character_width(image_font, char) for char in word)
        else:
            word_width = sum(_compute_character_width(image_font, char) for char in word)
            word_width += character_spacing * (len(word) - 1)
        
        # Add space width if not the first word in line
        space_needed = space_width if current_line else 0
        
        # Check if adding this word would exceed max_width
        if current_line and current_width + space_needed + word_width > max_width:
            # Start new line
            lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_width
        else:
            # Add word to current line
            current_line.append(word)
            current_width += space_needed + word_width
    
    # Add the last line if it has content
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines if lines else [text]


def _generate_horizontal_text(
    text: str,
    font: str,
    text_color: str,
    font_size: int,
    space_width: int,
    character_spacing: int,
    fit: bool,
    word_split: bool,
    stroke_width: int = 0,
    stroke_fill: str = "#282828",
    max_width: int = None,  # NEW PARAMETER
    line_spacing: int = 5,  # NEW PARAMETER
) -> Tuple:
    image_font = ImageFont.truetype(font=font, size=font_size)
    space_width = int(get_text_width(image_font, " ") * space_width)
    
    # NEW: Wrap text into multiple lines if max_width is specified
    if max_width:
        text_lines = _wrap_text_to_lines(text, image_font, max_width, space_width, character_spacing, word_split)
    else:
        text_lines = [text]
    
    # Calculate dimensions for all lines
    line_widths = []
    line_heights = []
    all_splitted_texts = []
    all_piece_widths = []
    
    for line_text in text_lines:
        if word_split:
            splitted_text = []
            for w in line_text.split(" "):
                splitted_text.append(w)
                splitted_text.append(" ")
            if splitted_text:  # Remove last space if exists
                splitted_text.pop()
        else:
            splitted_text = line_text

        piece_widths = [
            _compute_character_width(image_font, p) if p != " " else space_width
            for p in splitted_text
        ]
        
        line_width = sum(piece_widths)
        if not word_split and line_text:
            line_width += character_spacing * (len(line_text) - 1)
        
        line_height = max([get_text_height(image_font, p) for p in splitted_text]) if splitted_text else 0
        
        line_widths.append(line_width)
        line_heights.append(line_height)
        all_splitted_texts.append(splitted_text)
        all_piece_widths.append(piece_widths)
    
    # Calculate total image dimensions
    total_width = max(line_widths) if line_widths else 0
    total_height = sum(line_heights) + line_spacing * (len(text_lines) - 1) if text_lines else 0
    
    # Create images
    txt_img = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (total_width, total_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
    txt_mask_draw.fontmode = "1"

    # Color setup
    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill_color = (
        rnd.randint(min(stroke_c1[0], stroke_c2[0]), max(stroke_c1[0], stroke_c2[0])),
        rnd.randint(min(stroke_c1[1], stroke_c2[1]), max(stroke_c1[1], stroke_c2[1])),
        rnd.randint(min(stroke_c1[2], stroke_c2[2]), max(stroke_c1[2], stroke_c2[2])),
    )
    
    # Draw each line
    current_y = 0
    char_index = 0
    
    for line_idx, (splitted_text, piece_widths, line_height) in enumerate(zip(all_splitted_texts, all_piece_widths, line_heights)):
        for i, p in enumerate(splitted_text):
            x_pos = sum(piece_widths[0:i]) + i * character_spacing * int(not word_split)
            
            txt_img_draw.text(
                (x_pos, current_y),
                p,
                fill=fill,
                font=image_font,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill_color,
            )
            txt_mask_draw.text(
                (x_pos, current_y),
                p,
                fill=((char_index + 1) // (255 * 255), (char_index + 1) // 255, (char_index + 1) % 255),
                font=image_font,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill_color,
            )
            char_index += 1
        
        current_y += line_height + line_spacing

    if fit:
        bbox = txt_img.getbbox()
        if bbox:
            return txt_img.crop(bbox), txt_mask.crop(bbox)
        else:
            return txt_img, txt_mask
    else:
        return txt_img, txt_mask


def _generate_vertical_text(
    text: str,
    font: str,
    text_color: str,
    font_size: int,
    space_width: int,
    character_spacing: int,
    fit: bool,
    stroke_width: int = 0,
    stroke_fill: str = "#282828",
) -> Tuple:
    """UNCHANGED: Vertical text generation remains the same"""
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_height = int(get_text_height(image_font, " ") * space_width)

    char_heights = [
        get_text_height(image_font, c) if c != " " else space_height for c in text
    ]
    text_width = max([get_text_width(image_font, c) for c in text])
    text_height = sum(char_heights) + character_spacing * len(text)

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask)
    txt_mask_draw.fontmode = "1"

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(c1[0], c2[0]),
        rnd.randint(c1[1], c2[1]),
        rnd.randint(c1[2], c2[2]),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill_color = (
        rnd.randint(stroke_c1[0], stroke_c2[0]),
        rnd.randint(stroke_c1[1], stroke_c2[1]),
        rnd.randint(stroke_c1[2], stroke_c2[2]),
    )

    for i, c in enumerate(text):
        txt_img_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        txt_mask_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )

    if fit:
        bbox = txt_img.getbbox()
        if bbox:
            return txt_img.crop(bbox), txt_mask.crop(bbox)
        else:
            return txt_img, txt_mask
    else:
        return txt_img, txt_mask


# BACKUP FUNCTIONS - Use these to restore original functionality
def generate_original(
    text: str,
    font: str,
    text_color: str,
    font_size: int,
    orientation: int,
    space_width: int,
    character_spacing: int,
    fit: bool,
    word_split: bool,
    stroke_width: int = 0,
    stroke_fill: str = "#282828",
) -> Tuple:
    """RESTORE FUNCTION: Original generate function without multiline support"""
    if orientation == 0:
        return _generate_horizontal_text_original(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            word_split,
            stroke_width,
            stroke_fill,
        )
    elif orientation == 1:
        return _generate_vertical_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            stroke_width,
            stroke_fill,
        )
    else:
        raise ValueError("Unknown orientation " + str(orientation))


def _generate_horizontal_text_original(
    text: str,
    font: str,
    text_color: str,
    font_size: int,
    space_width: int,
    character_spacing: int,
    fit: bool,
    word_split: bool,
    stroke_width: int = 0,
    stroke_fill: str = "#282828",
) -> Tuple:
    """RESTORE FUNCTION: Original horizontal text generation"""
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_width = int(get_text_width(image_font, " ") * space_width)

    if word_split:
        splitted_text = []
        for w in text.split(" "):
            splitted_text.append(w)
            splitted_text.append(" ")
        splitted_text.pop()
    else:
        splitted_text = text

    piece_widths = [
        _compute_character_width(image_font, p) if p != " " else space_width
        for p in splitted_text
    ]
    text_width = sum(piece_widths)
    if not word_split:
        text_width += character_spacing * (len(text) - 1)

    text_height = max([get_text_height(image_font, p) for p in splitted_text])

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
    txt_mask_draw.fontmode = "1"

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill_color = (
        rnd.randint(min(stroke_c1[0], stroke_c2[0]), max(stroke_c1[0], stroke_c2[0])),
        rnd.randint(min(stroke_c1[1], stroke_c2[1]), max(stroke_c1[1], stroke_c2[1])),
        rnd.randint(min(stroke_c1[2], stroke_c2[2]), max(stroke_c1[2], stroke_c2[2])),
    )

    for i, p in enumerate(splitted_text):
        txt_img_draw.text(
            (sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0),
            p,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )
        txt_mask_draw.text(
            (sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0),
            p,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill_color,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask