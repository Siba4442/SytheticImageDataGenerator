#!/usr/bin/env python3
"""
Programmatic Odia OCR Synthetic Data Generator
Fixes filename length issues and provides better control
"""

import os
import sys
import hashlib
from pathlib import Path
from trdg.generators import GeneratorFromStrings
import argparse

def generate_short_filename(text, index=None):
    """
    Generate a short filename from text using hash to avoid Windows path length limits
    """
    # Create a hash of the text for unique identification
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    
    # Use index if provided, otherwise just hash
    if index is not None:
        return f"odia_{index:06d}_{text_hash}"
    else:
        return f"odia_{text_hash}"

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")

def generate_odia_ocr_data(
    strings_list=None,
    count=10,
    output_dir="output",
    font_size=64,
    font_path=None,
    background_type=0,  # 0=Gaussian Noise, 1=Plain white, 2=Quasicrystal, 3=Picture
    distorsion_type=0,  # 0=None, 1=Distortion, 2=Stretch
    distorsion_orientation=0,  # 0=Vertical, 1=Horizontal, 2=Both
    is_handwritten=False,
    width=-1,
    alignment=1,  # 0=left, 1=center, 2=right
    text_color="#282828",
    orientation=0,
    space_width=1.0,
    character_spacing=0,
    margins=(5, 5, 5, 5),
    fit=False,
    output_mask=False,
    word_split=False,
    image_dir=None,
    stroke_width=0,
    stroke_fill="#282828",
    image_mode="RGB"
):
    """
    Generate Odia OCR synthetic data programmatically
    
    Args:
        strings_list: List of Odia strings to generate images for
        count: Number of images to generate
        output_dir: Directory to save generated images
        font_size: Size of the font
        font_path: Path to Odia font file
        background_type: Type of background (0-3)
        distorsion_type: Type of distortion (0-2)
        distorsion_orientation: Orientation of distortion (0-2)
        is_handwritten: Whether to use handwritten style
        width: Width of generated image (-1 for auto)
        alignment: Text alignment (0-2)
        text_color: Color of text
        orientation: Text orientation
        space_width: Width of spaces
        character_spacing: Spacing between characters
        margins: Margins (top, left, bottom, right)
        fit: Whether to fit text to image
        output_mask: Whether to output mask
        word_split: Whether to split words
        image_dir: Directory for background images
        stroke_width: Width of text stroke
        stroke_fill: Color of text stroke
        image_mode: Image mode (RGB, RGBA, L)
    """
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Default Odia strings if none provided
    if strings_list is None:
        strings_list = [
            "କଣରକ",
            "ଭବନଶୱର",
            "କଟକ", 
            "ଓଡଶ",
            "ସମଲପଲ",
            "ପର",
            "ଚଲକ",
            "ରରକଲ",
            "ସପରଭତ",
            "ଓଡଶ ରଜ୍ୟ",
            "ଭରତ ଦଶ",
            "କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର",
            "ଜଗନ୍ନାଥ ମନ୍ଦିର",
            "କଳିଙ୍ଗ ଯୁଦ୍ଧ"
        ]
    
    print(f"Generating {count} images with Odia text...")
    print(f"Output directory: {output_dir}")
    print(f"Font size: {font_size}")
    
    try:
        # Create generator
        generator = GeneratorFromStrings(
            strings=strings_list,
            count=count,
            fonts=[font_path] if font_path else [],
            language="or",  # Odia language code
            size=font_size,
            skewing_angle=0,
            random_skew=False,
            blur=0,
            random_blur=False,
            background_type=background_type,
            distorsion_type=distorsion_type,
            distorsion_orientation=distorsion_orientation,
            is_handwritten=is_handwritten,
            width=width,
            alignment=alignment,
            text_color=text_color,
            orientation=orientation,
            space_width=space_width,
            character_spacing=character_spacing,
            margins=margins,
            fit=fit,
            output_mask=output_mask,
            word_split=word_split,
            image_dir=image_dir,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            image_mode=image_mode
        )
        
        # Generate images
        generated_count = 0
        labels_file = os.path.join(output_dir, "labels.txt")
        
        with open(labels_file, "w", encoding="utf-8") as f:
            for img, lbl in generator:
                # Generate short filename
                filename = generate_short_filename(lbl, generated_count)
                image_path = os.path.join(output_dir, f"{filename}.jpg")
                
                # Save image
                img.save(image_path)
                
                # Write label to file
                f.write(f"{filename}.jpg\t{lbl}\n")
                
                generated_count += 1
                
                if generated_count % 10 == 0:
                    print(f"Generated {generated_count}/{count} images...")
        
        print(f"\nSuccessfully generated {generated_count} images!")
        print(f"Images saved in: {output_dir}")
        print(f"Labels saved in: {labels_file}")
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate Odia OCR synthetic data")
    parser.add_argument("-c", "--count", type=int, default=10, help="Number of images to generate")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-f", "--font_size", type=int, default=64, help="Font size")
    parser.add_argument("--font_path", help="Path to Odia font file")
    parser.add_argument("-w", "--width", type=int, default=-1, help="Image width")
    parser.add_argument("--background", type=int, default=0, choices=[0,1,2,3], 
                       help="Background type (0=noise, 1=white, 2=quasicrystal, 3=picture)")
    parser.add_argument("--distortion", type=int, default=0, choices=[0,1,2],
                       help="Distortion type (0=none, 1=distortion, 2=stretch)")
    
    args = parser.parse_args()
    
    # Custom Odia strings - you can modify this list
    odia_strings = [
        "କଣରକ ସୂର୍ଯ୍ୟ ମନ୍ଦିର",
        "ଭବନଶୱର ଓଡଶ",
        "କଟକ ନଗର",
        "ଜଗନ୍ନାଥ ପର",
        "ସମଲପଲ ହର",
        "କଳିଙ୍ଗ ଯୁଦ୍ଧ",
        "ଭରତ ଦଶ",
        "ଓଡଶ ରଜ୍ୟ",
        "ପର ଧମ",
        "ଚଲକ ମହଭରତ"
    ]
    
    generate_odia_ocr_data(
        strings_list=odia_strings,
        count=args.count,
        output_dir=args.output,
        font_size=args.font_size,
        font_path=args.font_path,
        width=args.width,
        background_type=args.background,
        distorsion_type=args.distortion
    )

if __name__ == "__main__":
    main()