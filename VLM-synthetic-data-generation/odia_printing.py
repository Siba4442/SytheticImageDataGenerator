#!/usr/bin/env python3
"""
Enhanced script to verify Pillow text shaping support for Odia text
with improved multiline output formatting
"""

from PIL import Image, ImageDraw, ImageFont, features
import sys
import os

def print_section(title, content_func):
    """Print a section with proper formatting"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    content_func()
    print(f"{'='*60}")

def check_pillow_features():
    """Check what features are available in current Pillow installation"""
    print(f"Pillow Version: {Image.__version__}")
    print()
    
    # Check for text shaping features
    features_to_check = [
        ('raqm', 'Complex text layout engine'),
        ('fribidi', 'Bidirectional text support'),
        ('harfbuzz', 'Advanced text shaping engine'),
    ]
    
    print("Feature Availability:")
    print("-" * 40)
    available_features = []
    
    for feature, description in features_to_check:
        is_available = features.check(feature)
        status = "✓ AVAILABLE" if is_available else "✗ NOT AVAILABLE"
        print(f"  {feature:<12} | {status:<15} | {description}")
        if is_available:
            available_features.append(feature)
    
    print()
    if available_features:
        print(f"✓ Found {len(available_features)} text shaping feature(s): {', '.join(available_features)}")
        return True
    else:
        print("✗ No text shaping features detected")
        return False

def check_dll_location():
    """Check if fribidi.dll is in the correct location"""
    print("Searching for fribidi.dll in common locations...")
    print()
    
    # Get system paths
    import site
    venv_scripts = os.path.join(sys.prefix, 'Scripts')
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
    
    locations_to_check = [
        (venv_scripts, "Virtual Environment Scripts"),
        (os.path.dirname(sys.executable), "Python Executable Directory"),
        (os.path.join(site_packages, 'PIL') if site_packages else None, "PIL Package Directory"),
        (os.getcwd(), "Current Working Directory"),
    ]
    
    print("Search Results:")
    print("-" * 50)
    dll_found = False
    
    for location, description in locations_to_check:
        if location is None:
            continue
            
        dll_path = os.path.join(location, 'fribidi.dll')
        if os.path.exists(dll_path):
            print(f"  ✓ FOUND    | {description}")
            print(f"             | Path: {dll_path}")
            dll_found = True
        else:
            print(f"  ✗ MISSING  | {description}")
            print(f"             | Expected: {dll_path}")
        print()
    
    if not dll_found:
        print("⚠️  RECOMMENDATION:")
        print("   Download fribidi.dll from:")
        print("   https://github.com/python-pillow/pillow-wheels/raw/main/fribidi.dll")
        print(f"   Place it in: {venv_scripts}")
    
    return dll_found

def test_odia_rendering(font_path=None):
    """Test Odia text rendering with matras"""
    print("Testing Odia text rendering capabilities...")
    print()
    
    # Sample Odia text with matras and complex characters
    odia_test_cases = [
        ("ଓଡ଼ିଆ", "Odia (language name with matras)"),
        ("ଭାରତ", "Bharat (India)"),
        ("ମୋର ନାମ", "My name"),
        ("କେମିତି ଅଛ", "How are you"),
        ("ସୁପ୍ରଭାତ", "Good morning"),
        ("ଧନ୍ୟବାଦ", "Thank you"),
    ]
    
    # Font search paths
    font_search_paths = [
        (font_path, "Custom provided font"),
        ("C:/Windows/Fonts/NotoSansOriya-Regular.ttf", "Windows Noto Sans Oriya"),
        ("C:/Windows/Fonts/kalinga.ttf", "Windows Kalinga"),
        ("/System/Library/Fonts/NotoSansOriya.ttf", "macOS Noto Sans Oriya"),
        ("/usr/share/fonts/truetype/noto/NotoSansOriya-Regular.ttf", "Linux Noto Sans Oriya"),
    ]
    
    # Try to find and load a suitable font
    font = None
    print("Font Search:")
    print("-" * 30)
    
    for path, description in font_search_paths:
        if path and os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 48)
                print(f"  ✓ LOADED   | {description}")
                print(f"             | Path: {path}")
                break
            except Exception as e:
                print(f"  ✗ FAILED   | {description}")
                print(f"             | Error: {str(e)}")
        elif path:
            print(f"  ✗ MISSING  | {description}")
            print(f"             | Path: {path}")
    
    if not font:
        print(f"  ⚠️  FALLBACK | Using default system font")
        print(f"             | Warning: Odia text may not render correctly")
        font = ImageFont.load_default()
    
    print()
    print("Text Rendering Test Results:")
    print("-" * 40)
    
    successful_renders = 0
    for i, (text, description) in enumerate(odia_test_cases, 1):
        try:
            # Create test image
            img = Image.new('RGB', (500, 120), 'white')
            draw = ImageDraw.Draw(img)
            
            # Draw the text
            draw.text((20, 40), text, font=font, fill='black')
            
            # Add description
            draw.text((20, 90), f"({description})", font=ImageFont.load_default(), fill='gray')
            
            # Save the image
            output_path = f"odia_test_{i:02d}.png"
            img.save(output_path)
            
            print(f"  ✓ SUCCESS  | Test {i:02d}: {text}")
            print(f"             | Description: {description}")
            print(f"             | Output: {output_path}")
            successful_renders += 1
            
        except Exception as e:
            print(f"  ✗ FAILED   | Test {i:02d}: {text}")
            print(f"             | Error: {str(e)}")
        print()
    
    print(f"Rendering Summary: {successful_renders}/{len(odia_test_cases)} tests successful")
    return successful_renders > 0

def print_recommendations(has_features, dll_found, rendering_works):
    """Print detailed recommendations based on test results"""
    print("Diagnostic Results & Recommendations:")
    print("-" * 45)
    
    if has_features and dll_found and rendering_works:
        print("  🎉 STATUS: EXCELLENT")
        print("     Your setup is fully configured for Odia text rendering!")
        print("     All text shaping features are available and working.")
        
    elif has_features and rendering_works:
        print("  ✅ STATUS: GOOD")
        print("     Text shaping is working, though DLL location is unclear.")
        print("     Your Odia text should render correctly.")
        
    elif has_features and dll_found:
        print("  ⚠️  STATUS: NEEDS TESTING")
        print("     Text shaping features detected but rendering had issues.")
        print("     Recommendations:")
        print("     • Check if your font supports Odia script")
        print("     • Try a different Odia font")
        print("     • Restart your Python environment")
        
    elif has_features:
        print("  ⚠️  STATUS: PARTIAL SETUP")
        print("     Text shaping available but DLL location unclear.")
        print("     Recommendations:")
        print("     • Download fribidi.dll to your Scripts folder")
        print("     • Restart Python after adding the DLL")
        
    else:
        print("  ❌ STATUS: NEEDS SETUP")
        print("     Text shaping support not detected.")
        print("     Recommendations:")
        print("     • Download fribidi.dll from official source")
        print("     • Place it in your Python Scripts directory")
        print("     • Consider upgrading Pillow: pip install --upgrade Pillow")
        print("     • Restart your Python environment")
        
    print()
    print("Additional Resources:")
    print("• Pillow Documentation: https://pillow.readthedocs.io/")
    print("• Odia Fonts: https://fonts.google.com/?subset=oriya")
    print("• fribidi.dll: https://github.com/python-pillow/pillow-wheels/")

def main():
    """Main diagnostic function with enhanced output"""
    print("🔍 ODIA TEXT RENDERING DIAGNOSTIC TOOL")
    print("=" * 60)
    print("This tool will check your system's capability to render")
    print("Odia text with proper complex script support.")
    print("=" * 60)
    
    # Run all diagnostic tests
    has_features = False
    dll_found = False
    rendering_works = False
    
    print_section("PILLOW FEATURES CHECK", lambda: globals().update({'has_features': check_pillow_features()}))
    print_section("DLL LOCATION CHECK", lambda: globals().update({'dll_found': check_dll_location()}))
    print_section("ODIA RENDERING TEST", lambda: globals().update({'rendering_works': test_odia_rendering(r"C:\Users\sibap\Programming\Odia_gen_ai\VLM-synthetic-data-generation\fonts\NotoSansOriya.ttf")}))
    
    # Update variables from globals (workaround for lambda scope)
    has_features = globals().get('has_features', False)
    dll_found = globals().get('dll_found', False)
    rendering_works = globals().get('rendering_works', False)
    
    print_section("FINAL DIAGNOSIS", lambda: print_recommendations(has_features, dll_found, rendering_works))
    
    print("\n" + "="*60)
    print("Diagnostic complete! Check the generated PNG files to verify")
    print("that Odia text renders correctly with proper matra positioning.")
    print("="*60)

if __name__ == "__main__":
    main()