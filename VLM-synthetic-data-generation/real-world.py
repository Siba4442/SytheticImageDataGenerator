#!/usr/bin/env python3
"""
Enhanced script to render multiline Odia text with realistic effects for VLM training
Adds background textures, printer defects, ink effects, and various degradations
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import random
import os
from scipy.ndimage import gaussian_filter
import cv2

class RealisticOdiaImageGenerator:
    def __init__(self):
        self.font_paths = [
            r"C:\Users\sibap\Programming\Odia_gen_ai\VLM-synthetic-data-generation\fonts\NotoSansOriya.ttf",
            "C:/Windows/Fonts/NotoSansOriya-Regular.ttf",
            "C:/Windows/Fonts/kalinga.ttf"
        ]
        
    def load_font(self, size=40):
        """Load Odia font with fallback"""
        for path in self.font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except:
                    continue
        return ImageFont.load_default()
    
    def create_paper_texture(self, width, height):
        """Create realistic paper texture background"""
        # Base paper color with slight variations
        base_color = random.randint(240, 255)
        
        # Create noise for paper texture
        noise = np.random.normal(0, 8, (height, width, 3))
        
        # Create base paper
        paper = np.full((height, width, 3), [base_color, base_color-2, base_color-1], dtype=np.float32)
        
        # Add texture noise
        paper += noise
        
        # Add paper grain pattern
        grain_x = np.sin(np.linspace(0, 20*np.pi, width)) * 3
        grain_y = np.sin(np.linspace(0, 15*np.pi, height)) * 2
        grain = np.outer(grain_y, grain_x)
        
        for i in range(3):
            paper[:, :, i] += grain
        
        # Add subtle stains and marks
        if random.random() < 0.7:
            stain_count = random.randint(2, 8)
            for _ in range(stain_count):
                x = random.randint(0, width-50)
                y = random.randint(0, height-50)
                stain_size = random.randint(20, 80)
                stain_intensity = random.randint(-15, -5)
                
                # Create circular stain
                y_coords, x_coords = np.ogrid[:height, :width]
                mask = (x_coords - x)**2 + (y_coords - y)**2 <= stain_size**2
                paper[mask] += stain_intensity
        
        # Clip values
        paper = np.clip(paper, 0, 255).astype(np.uint8)
        
        return Image.fromarray(paper)
    
    def add_printer_drum_defects(self, img):
        """Add printer drum artifacts - regular patterns"""
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Add regular banding (drum defects)
        if random.random() < 0.6:
            band_spacing = random.randint(40, 120)
            band_intensity = random.randint(5, 20)
            
            for y in range(0, height, band_spacing):
                band_width = random.randint(2, 8)
                if y + band_width < height:
                    # Darken or lighten bands
                    modifier = random.choice([-band_intensity, band_intensity])
                    img_array[y:y+band_width, :] = np.clip(
                        img_array[y:y+band_width, :].astype(np.int16) + modifier, 0, 255
                    ).astype(np.uint8)
        
        # Add roller marks (vertical patterns)
        if random.random() < 0.4:
            roller_spacing = random.randint(30, 80)
            for x in range(0, width, roller_spacing):
                if random.random() < 0.7:
                    thickness = random.randint(1, 3)
                    intensity = random.randint(-15, -5)
                    if x + thickness < width:
                        img_array[:, x:x+thickness] = np.clip(
                            img_array[:, x:x+thickness].astype(np.int16) + intensity, 0, 255
                        ).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def add_ink_mottling(self, img):
        """Add ink mottling effects - uneven ink distribution"""
        img_array = np.array(img)
        
        # Create mottling mask
        height, width = img_array.shape[:2]
        
        # Generate Perlin-like noise for ink variation
        scale = random.uniform(0.02, 0.08)
        noise = np.random.perlin2d((height, width), (scale, scale)) if hasattr(np.random, 'perlin2d') else \
                np.random.normal(0, 0.3, (height, width))
        
        # Smooth the noise
        noise = gaussian_filter(noise, sigma=random.uniform(2, 6))
        
        # Apply mottling to darker regions (text areas)
        gray = np.mean(img_array, axis=2)
        text_mask = gray < 200
        
        mottling_intensity = random.uniform(0.1, 0.4)
        for i in range(3):
            channel = img_array[:, :, i].astype(np.float32)
            channel[text_mask] += noise[text_mask] * mottling_intensity * 50
            img_array[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def add_letterpress_impression(self, img, text_positions):
        """Add letterpress/embossed effect around text"""
        img_array = np.array(img)
        
        # Create shadow/highlight effect for each text region
        for pos in text_positions:
            x, y, w, h = pos
            
            # Add subtle shadow offset
            shadow_offset = random.randint(1, 3)
            highlight_offset = random.randint(-2, -1)
            
            # Create emboss effect
            if y + shadow_offset < img_array.shape[0] and x + shadow_offset < img_array.shape[1]:
                shadow_region = img_array[y:y+h, x:x+w]
                if shadow_region.size > 0:
                    # Darken for shadow
                    img_array[y+shadow_offset:y+h+shadow_offset, x+shadow_offset:x+w+shadow_offset] = \
                        np.clip(img_array[y+shadow_offset:y+h+shadow_offset, x+shadow_offset:x+w+shadow_offset].astype(np.int16) - 15, 0, 255).astype(np.uint8)
                    
                    # Lighten for highlight
                    img_array[y+highlight_offset:y+h+highlight_offset, x+highlight_offset:x+w+highlight_offset] = \
                        np.clip(img_array[y+highlight_offset:y+h+highlight_offset, x+highlight_offset:x+w+highlight_offset].astype(np.int16) + 10, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def add_lighting_gradient(self, img):
        """Add realistic lighting gradients"""
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Create gradient mask
        gradient_type = random.choice(['radial', 'linear', 'corner'])
        
        if gradient_type == 'radial':
            # Radial gradient from center
            center_x, center_y = width // 2, height // 2
            y_coords, x_coords = np.ogrid[:height, :width]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            gradient = distances / max_distance
            
        elif gradient_type == 'linear':
            # Linear gradient
            direction = random.choice(['horizontal', 'vertical'])
            if direction == 'horizontal':
                gradient = np.linspace(0, 1, width)
                gradient = np.tile(gradient, (height, 1))
            else:
                gradient = np.linspace(0, 1, height)
                gradient = np.tile(gradient.reshape(-1, 1), (1, width))
                
        else:  # corner
            # Corner vignette
            y_coords, x_coords = np.ogrid[:height, :width]
            corner_x, corner_y = random.choice([(0, 0), (width, 0), (0, height), (width, height)])
            distances = np.sqrt((x_coords - corner_x)**2 + (y_coords - corner_y)**2)
            gradient = distances / np.max(distances)
        
        # Apply gradient
        intensity = random.uniform(0.1, 0.3)
        modifier = (gradient - 0.5) * intensity * 100
        
        for i in range(3):
            img_array[:, :, i] = np.clip(
                img_array[:, :, i].astype(np.float32) + modifier, 0, 255
            ).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def add_line_degradation(self, img):
        """Add line breaks, fading, and character degradation"""
        img_array = np.array(img)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Find text regions
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Add random erosion/dilation to simulate wear
        if random.random() < 0.6:
            kernel_size = random.randint(1, 3)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if random.random() < 0.5:
                binary = cv2.erode(binary, kernel, iterations=1)
            else:
                binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Convert back to RGB
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # Blend with original
        alpha = random.uniform(0.7, 0.9)
        result = alpha * img_array + (1 - alpha) * (255 - binary_rgb)
        
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    def add_shadow_effects(self, img):
        """Add realistic shadow effects"""
        # Convert to array
        img_array = np.array(img)
        
        # Create shadow layer
        shadow = img_array.copy()
        
        # Make shadows darker
        shadow = np.clip(shadow.astype(np.int16) - random.randint(20, 50), 0, 255).astype(np.uint8)
        
        # Blur shadow
        shadow_pil = Image.fromarray(shadow)
        shadow_pil = shadow_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))
        
        # Offset shadow
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(1, 4)
        
        # Create new image with shadow
        new_img = Image.new('RGB', img.size, 'white')
        new_img.paste(shadow_pil, (offset_x, offset_y))
        new_img = Image.alpha_composite(new_img.convert('RGBA'), img.convert('RGBA'))
        
        return new_img.convert('RGB')
    
    def add_ink_bleeding(self, img):
        """Add ink bleeding effect"""
        img_array = np.array(img)
        
        # Create bleeding mask for text areas
        gray = np.mean(img_array, axis=2)
        text_mask = gray < 180
        
        # Apply slight blur to text areas
        for i in range(3):
            channel = img_array[:, :, i].astype(np.float32)
            blurred_channel = gaussian_filter(channel, sigma=random.uniform(0.5, 1.5))
            
            # Blend original and blurred
            blend_factor = random.uniform(0.2, 0.5)
            channel[text_mask] = (1 - blend_factor) * channel[text_mask] + blend_factor * blurred_channel[text_mask]
            
            img_array[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def create_realistic_multiline_odia_image(self, output_name="realistic_odia_multiline.png"):
        """Create realistic multiline Odia text image with all effects"""
        
        # Odia text lines
        odia_lines = [
            "ଓଡ଼ିଆ ଭାଷା",
            "ମୋର ନାମ ରାମ", 
            "ଆଜି ଏକ ସୁନ୍ଦର ଦିନ",
            "କେମିତି ଅଛ ତୁମେ?",
            "ଧନ୍ୟବାଦ ଓ ନମସ୍କାର",
            "ଭାରତ ଏକ ମହାନ ଦେଶ"
        ]
        
        # Load font
        font_size = random.randint(35, 50)
        font = self.load_font(font_size)
        
        # Calculate image dimensions
        img_width = random.randint(600, 800)
        line_height = font_size + random.randint(15, 25)
        img_height = len(odia_lines) * line_height + random.randint(60, 100)
        
        # Create paper texture background
        img = self.create_paper_texture(img_width, img_height)
        draw = ImageDraw.Draw(img)
        
        # Track text positions for letterpress effect
        text_positions = []
        
        # Draw text with variations
        y_position = random.randint(20, 40)
        for line in odia_lines:
            x_position = random.randint(15, 35)
            
            # Add slight rotation to individual lines
            if random.random() < 0.3:
                line_img = Image.new('RGBA', (img_width, line_height + 20), (255, 255, 255, 0))
                line_draw = ImageDraw.Draw(line_img)
                line_draw.text((x_position, 10), line, font=font, fill='black')
                
                # Rotate slightly
                angle = random.uniform(-1.5, 1.5)
                line_img = line_img.rotate(angle, expand=False, fillcolor=(255, 255, 255, 0))
                
                img.paste(line_img, (0, y_position), line_img)
            else:
                # Get text bbox for letterpress effect
                bbox = draw.textbbox((x_position, y_position), line, font=font)
                text_positions.append((bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]))
                
                # Add slight color variation to text
                text_color = (
                    random.randint(0, 30),
                    random.randint(0, 20), 
                    random.randint(0, 25)
                )
                draw.text((x_position, y_position), line, font=font, fill=text_color)
            
            y_position += line_height + random.randint(-5, 5)
        
        # Apply all effects randomly
        effects = [
            (self.add_printer_drum_defects, 0.7),
            (self.add_ink_mottling, 0.8),
            (lambda x: self.add_letterpress_impression(x, text_positions), 0.5),
            (self.add_lighting_gradient, 0.9),
            (self.add_line_degradation, 0.6),
            (self.add_shadow_effects, 0.4),
            (self.add_ink_bleeding, 0.7)
        ]
        
        for effect_func, probability in effects:
            if random.random() < probability:
                img = effect_func(img)
        
        # Final adjustments
        if random.random() < 0.3:
            # Adjust contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() < 0.4:
            # Adjust brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.9, 1.1))
        
        # Save image
        img.save(output_name, quality=random.randint(80, 95))
        print(f"Created realistic image: {output_name}")
        
        return output_name
    
    def create_realistic_paragraph_odia(self, output_name="realistic_odia_paragraph.png"):
        """Create realistic paragraph-style Odia text"""
        
        odia_paragraph = """ଓଡ଼ିଆ ଭାରତର ଏକ ପ୍ରାଚୀନ ଭାଷା।
ଏହା ଓଡ଼ିଶାର ସରକାରୀ ଭାଷା।
ଓଡ଼ିଆ ସାହିତ୍ୟ ବହୁତ ସମୃଦ୍ଧ।
ଜଗନ୍ନାଥ ମନ୍ଦିର ପୁରୀରେ ଅଛି।
ଓଡ଼ିଶାର ସଂସ୍କୃତି ଅତି ସୁନ୍ଦର।"""
        
        # Load font
        font_size = random.randint(30, 45)
        font = self.load_font(font_size)
        
        # Create textured background
        img_width = random.randint(700, 900)
        img_height = random.randint(400, 600)
        img = self.create_paper_texture(img_width, img_height)
        
        draw = ImageDraw.Draw(img)
        
        # Draw text with effects
        x_offset = random.randint(30, 50)
        y_offset = random.randint(30, 50)
        
        draw.multiline_text(
            (x_offset, y_offset), 
            odia_paragraph, 
            font=font, 
            fill=(random.randint(0, 25), random.randint(0, 15), random.randint(0, 20)),
            spacing=random.randint(8, 15)
        )
        
        # Apply effects
        effects_to_apply = random.sample([
            self.add_printer_drum_defects,
            self.add_ink_mottling,
            self.add_lighting_gradient,
            self.add_line_degradation,
            self.add_ink_bleeding
        ], k=random.randint(3, 5))
        
        for effect in effects_to_apply:
            img = effect(img)
        
        # Save
        img.save(output_name, quality=random.randint(75, 90))
        print(f"Created realistic paragraph: {output_name}")
        
        return output_name
    
    def generate_training_dataset(self, num_images=50):
        """Generate multiple realistic images for VLM training"""
        print(f"Generating {num_images} realistic Odia images for VLM training...")
        
        os.makedirs("realistic_odia_dataset", exist_ok=True)
        
        for i in range(num_images):
            # Alternate between multiline and paragraph styles
            if i % 2 == 0:
                output_path = f"realistic_odia_dataset/multiline_{i:03d}.png"
                self.create_realistic_multiline_odia_image(output_path)
            else:
                output_path = f"realistic_odia_dataset/paragraph_{i:03d}.png"
                self.create_realistic_paragraph_odia(output_path)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_images} images...")
        
        print(f"Dataset generation complete! Check 'realistic_odia_dataset' folder.")

def main():
    generator = RealisticOdiaImageGenerator()
    
    print("Creating realistic Odia images with various effects...")
    
    # Create single examples
    generator.create_realistic_multiline_odia_image("sample_realistic_multiline.png")
    generator.create_realistic_paragraph_odia("sample_realistic_paragraph.png")
    
    # Generate training dataset
    choice = input("Generate full training dataset? (y/n): ").lower()
    if choice == 'y':
        num_images = int(input("Number of images to generate (default 50): ") or "50")
        generator.generate_training_dataset(num_images)
    
    print("Done! Check the generated PNG files.")

if __name__ == "__main__":
    main()