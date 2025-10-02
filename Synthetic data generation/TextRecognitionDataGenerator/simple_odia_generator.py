#!/usr/bin/env python3
"""
Simple drop-in solution for Odia OCR generation
Save this file in your TextRecognitionDataGenerator root directory
"""

import os
import sys
import hashlib
from pathlib import Path

# Add TRDG to path
current_dir = Path(__file__).parent
trdg_path = current_dir / "trdg"
sys.path.insert(0, str(trdg_path))

try:
    from trdg.generators import GeneratorFromStrings
except ImportError:
    print("Error: Cannot import TRDG. Make sure you're running this from the TextRecognitionDataGenerator directory")
    sys.exit(1)

def generate_odia_images(count=100, output_dir="odia_output", font_size=64, width=500):
    """
    Simple function to generate Odia OCR images
    
    Args:
        count: Number of images to generate
        output_dir: Output directory name
        font_size: Font size
        width: Image width
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    print(f"✅ Created output directory: {output_dir}")
    
    # Odia strings - you can modify this list
    odia_strings = [
        "କଣରକ",
        "ଭବନଶୱର", 
        "କଟକ",
        "ଓଡଶ",
        "ସମଲପଲ",
        "ପର",
        "ଚଲକ", 
        "ରରକଲ",
        "ସପରଭତ",
        "ଜଗନ୍ନାଥ ମନ୍ଦିର",
        "କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର", 
        "ଭରତ ଦଶ",
        "ଓଡଶ ରଜ୍ୟ",
        "କଳିଙ୍ଗ ଯୁଦ୍ଧ",
        "ଓଡ଼ିଆ ଭାଷା",
        "ବଙ୍ଗୋପସାଗର",
        "ମହାନଦୀ",
        "ଚଲେନ୍ଦ୍ର ସାଗର",
        "ପର ନୃସିଂହନାଥ",
        "ଲିଙ୍ଗରାଜ ମନ୍ଦିର",
        "କଣରକ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ଏକ ପ୍ରସିଦ୍ଧ ମନ୍ଦିର",
        "ଭବନଶୱର ଓଡଶ ରଜ୍ୟର ରାଜଧାନୀ", 
        "କଟକ ନଗର ଇତିହାସ ପ୍ରସିଦ୍ଧ",
        "ଜଗନ୍ନାଥ ମନ୍ଦିର ପରର ସବୁଠାରୁ ପବିତ୍ର ସ୍ଥାନ",
        "ଓଡ଼ିଆ ସାହିତ୍ୟ ବହୁତ ସମୃଦ୍ଧ",
        "କଳିଙ୍ଗ ଯୁଦ୍ଧ ପରେ ସମ୍ରାଟ ଅଶୋକ ବୌଦ୍ଧ ଧର୍ମ ଗ୍ରହଣ କଲେ",
        "ପର ଜଗନ୍ନାଥଙ୍କ ରଥଯାତ୍ରା ବିଶ୍ୱ ପ୍ରସିଦ୍ଧ",
        "ଓଡଶ ରଜ୍ୟ ଭରତର ପୂର୍ବ ଉପକୂଳରେ ଅବସ୍ଥିତ"
    ]
    
    print(f"🎯 Generating {count} images...")
    print(f"📏 Font size: {font_size}")
    print(f"📐 Image width: {width}")
    
    try:
        # Create generator
        generator = GeneratorFromStrings(
            strings=odia_strings,
            count=count,
            fonts=[],  # Will use default fonts
            language="or",  # Odia language code
            size=font_size,
            skewing_angle=0,
            random_skew=False,
            blur=0,
            random_blur=False,
            background_type=0,  # Gaussian noise
            distorsion_type=0,  # No distortion
            distorsion_orientation=0,
            is_handwritten=False,
            width=width,
            alignment=1,  # Center
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
        )
        
        # Generate and save images
        labels_file = os.path.join(output_dir, "labels.txt")
        generated_count = 0
        
        with open(labels_file, "w", encoding="utf-8") as f:
            for img, lbl in generator:
                # Create safe filename
                text_hash = hashlib.md5(lbl.encode('utf-8')).hexdigest()[:8]
                filename = f"odia_{generated_count:06d}_{text_hash}"
                image_path = os.path.join(output_dir, f"{filename}.jpg")
                
                # Save image
                img.save(image_path)
                
                # Write label
                f.write(f"{filename}.jpg\t{lbl}\n")
                
                generated_count += 1
                
                # Progress indicator
                if generated_count % 50 == 0:
                    print(f"📸 Generated {generated_count}/{count} images...")
        
        print(f"\n✅ SUCCESS! Generated {generated_count} images")
        print(f"📁 Images saved in: {output_dir}/")
        print(f"🏷️  Labels saved in: {labels_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    """Main function with simple command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Odia OCR Data Generator")
    parser.add_argument("-c", "--count", type=int, default=100, help="Number of images (default: 100)")
    parser.add_argument("-o", "--output", default="odia_output", help="Output directory (default: odia_output)")
    parser.add_argument("-f", "--font_size", type=int, default=64, help="Font size (default: 64)")
    parser.add_argument("-w", "--width", type=int, default=500, help="Image width (default: 500)")
    
    args = parser.parse_args()
    
    print("🔥 Odia OCR Data Generator")
    print("=" * 40)
    
    success = generate_odia_images(
        count=args.count,
        output_dir=args.output,
        font_size=args.font_size,
        width=args.width
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()