#!/usr/bin/env python3
"""
Script to verify if Pillow has proper text shaping support for Odia text
"""

from PIL import Image, ImageDraw, ImageFont, features
import sys
import os

def check_pillow_features():
    """Check what features are available in current Pillow installation"""
    print("=== Pillow Feature Check ===")
    print(f"Pillow version: {Image.__version__}")
    
    # Check for text shaping features
    features_to_check = [
        'raqm',      # Complex text layout
        'fribidi',   # Bidirectional text
        'harfbuzz',  # Text shaping engine
    ]
    
    for feature in features_to_check:
        status = "✓ Available" if features.check(feature) else "✗ Not available"
        print(f"{feature}: {status}")
    
    return any(features.check(f) for f in features_to_check)

def test_odia_rendering(font_path=None):
    """Test Odia text rendering with matras"""
    print("\n=== Odia Text Rendering Test ===")
    
    # Sample Odia text with matras
    odia_texts = [
        "ଓଡ଼ିଆ",      # Odia (with matras)
        "ଭାରତ",       # Bharat
        "ମୋର ନାମ",    # My name
        "କେମିତି ଅଛ",   # How are you
    ]
    
    # Try to find a suitable font
    font_paths = [
        font_path,
        "C:/Windows/Fonts/NotoSansOriya-Regular.ttf",
        "C:/Windows/Fonts/kalinga.ttf",
        "/System/Library/Fonts/NotoSansOriya.ttf",  # macOS
        "/usr/share/fonts/truetype/noto/NotoSansOriya-Regular.ttf",  # Linux
    ]
    
    font = None
    for path in font_paths:
        if path and os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 48)
                print(f"Using font: {path}")
                break
            except Exception as e:
                print(f"Failed to load font {path}: {e}")
                continue
    
    if not font:
        print("Warning: Using default font - Odia text may not render correctly")
        font = ImageFont.load_default()
    
    # Create test images
    for i, text in enumerate(odia_texts):
        img = Image.new('RGB', (400, 100), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Draw the text
            draw.text((10, 30), text, font=font, fill='black')
            
            # Save the image
            output_path = f"odia_test_{i+1}.png"
            img.save(output_path)
            print(f"Generated: {output_path} - Text: {text}")
            
        except Exception as e:
            print(f"Error rendering '{text}': {e}")

def check_dll_location():
    """Check if fribidi.dll is in the correct location"""
    print("\n=== DLL Location Check ===")
    
    # Common locations to check
    import site
    venv_scripts = os.path.join(sys.prefix, 'Scripts')
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else "Not found"
    
    locations_to_check = [
        venv_scripts,
        os.path.join(site_packages, 'PIL'),
        os.path.dirname(sys.executable),
        os.getcwd(),
    ]
    
    dll_found = False
    for location in locations_to_check:
        dll_path = os.path.join(location, 'fribidi.dll')
        if os.path.exists(dll_path):
            print(f"✓ Found fribidi.dll at: {dll_path}")
            dll_found = True
        else:
            print(f"✗ Not found at: {dll_path}")
    
    if not dll_found:
        print("\n⚠️  fribidi.dll not found in checked locations")
        print("Download from: https://github.com/python-pillow/pillow-wheels/raw/main/fribidi.dll")
    
    return dll_found

def main():
    print("Odia Text Rendering Diagnostic Tool")
    print("=" * 40)
    
    # Check Pillow features
    has_text_shaping = check_pillow_features()
    
    # Check DLL location
    dll_found = check_dll_location()
    
    # Test Odia rendering
    test_odia_rendering(r"C:\Users\sibap\Programming\Odia_gen_ai\VLM-synthetic-data-generation\fonts\NotoSansOriya.ttf")
    
    print("\n=== Summary ===")
    if has_text_shaping and dll_found:
        print("✓ Setup appears correct for Odia text rendering")
    elif has_text_shaping:
        print("⚠️  Text shaping available but DLL location unclear")
    else:
        print("✗ Text shaping support not detected")
        print("Recommendations:")
        print("1. Ensure fribidi.dll is in your Scripts folder")
        print("2. Consider upgrading Pillow: pip install --upgrade Pillow")
        print("3. Restart Python after adding DLL")

if __name__ == "__main__":
    main()