import cv2
import numpy as np

def create_printer_drum_effect(image_path, output_path=None, 
                              band_height=25, intensity=0.3, 
                              pattern_type='alternating', noise_level=0.1):
    """
    Apply printer drum defect effect to an image
    
    Parameters:
    - image_path: path to input image
    - output_path: path to save output (optional)
    - band_height: height of each defect band in pixels
    - intensity: strength of the effect (0.0 to 1.0)
    - pattern_type: 'alternating', 'random', or 'gradient'
    - noise_level: amount of random noise to add (0.0 to 1.0)
    """
    
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_path
    
    height, width = img.shape[:2]
    
    # Create the drum defect pattern
    drum_pattern = np.ones((height, width), dtype=np.float32)
    
    # Calculate number of bands
    num_bands = height // band_height + 1
    
    for i in range(num_bands):
        start_y = i * band_height
        end_y = min((i + 1) * band_height, height)
        
        if pattern_type == 'alternating':
            # Alternating dark and light bands
            if i % 2 == 0:
                drum_pattern[start_y:end_y, :] = 1.0 - intensity
            else:
                drum_pattern[start_y:end_y, :] = 1.0 + intensity * 0.5
                
        elif pattern_type == 'random':
            # Random intensity for each band
            band_intensity = np.random.uniform(1.0 - intensity, 1.0 + intensity)
            drum_pattern[start_y:end_y, :] = band_intensity
            
        elif pattern_type == 'gradient':
            # Gradient effect within each band
            band_gradient = np.linspace(1.0 - intensity, 1.0 + intensity, end_y - start_y)
            drum_pattern[start_y:end_y, :] = band_gradient[:, np.newaxis]
    
    # Add some random noise to make it more realistic
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, (height, width))
        drum_pattern += noise
    
    # Clip values to valid range
    drum_pattern = np.clip(drum_pattern, 0.3, 1.5)
    
    # Apply the drum pattern to each channel
    result = img.copy().astype(np.float32)
    for channel in range(3):
        result[:, :, channel] *= drum_pattern
    
    # Clip to valid pixel range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    return result

def create_advanced_drum_effect(image_path, output_path=None):
    """
    Create a more realistic printer drum effect with multiple defect types
    """
    
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Create base drum pattern
    result = img.copy().astype(np.float32)
    
    # Effect 1: Horizontal banding (main drum defect)
    band_height = np.random.randint(20, 40)
    for i in range(0, height, band_height):
        end_y = min(i + band_height, height)
        
        # Create varying intensity across the band
        band_intensity = 0.7 + 0.4 * np.sin(i * 0.1)
        
        # Apply with some horizontal variation
        for x in range(width):
            variation = 0.9 + 0.2 * np.sin(x * 0.01)
            final_intensity = band_intensity * variation
            result[i:end_y, x] *= final_intensity
    
    # Effect 2: Vertical streaks (roller marks)
    streak_width = 3
    for x in range(0, width, np.random.randint(50, 150)):
        streak_intensity = np.random.uniform(0.6, 1.4)
        end_x = min(x + streak_width, width)
        result[:, x:end_x] *= streak_intensity
    
    # Effect 3: Random spots and marks
    num_spots = np.random.randint(5, 15)
    for _ in range(num_spots):
        spot_x = np.random.randint(0, width-20)
        spot_y = np.random.randint(0, height-20)
        spot_size = np.random.randint(5, 20)
        spot_intensity = np.random.uniform(0.3, 0.8)
        
        # Create circular spot
        y, x = np.ogrid[:spot_size, :spot_size]
        mask = (x - spot_size//2)**2 + (y - spot_size//2)**2 <= (spot_size//2)**2
        
        end_y = min(spot_y + spot_size, height)
        end_x = min(spot_x + spot_size, width)
        
        spot_region = result[spot_y:end_y, spot_x:end_x]
        spot_region[mask[:end_y-spot_y, :end_x-spot_x]] *= spot_intensity
    
    # Effect 4: Overall aging/fading
    age_factor = np.random.uniform(0.85, 0.95)
    result *= age_factor
    
    # Add slight blur to simulate print quality
    result = cv2.GaussianBlur(result, (3, 3), 0.5)
    
    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    return result

def demo_effects():
    """
    Demo function to show different effect types
    """
    # Create a sample image if you don't have one
    sample_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    
    # Add some text-like patterns
    for i in range(20, 380, 25):
        sample_img[i:i+15, 50:550] = 50
        sample_img[i+5:i+10, 60:540] = 240
    
    # Apply different effects
    effects = [
        ('alternating', create_printer_drum_effect(sample_img, pattern_type='alternating')),
        ('random', create_printer_drum_effect(sample_img, pattern_type='random')),
        ('advanced', create_advanced_drum_effect(sample_img))
    ]
    
    # Save results
    cv2.imwrite('original.jpg', cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR))
    for name, effect_img in effects:
        cv2.imwrite(f'{name}_effect.jpg', cv2.cvtColor(effect_img, cv2.COLOR_RGB2BGR))
    
    print("Effects saved as image files")

# Example usage
if __name__ == "__main__":
    # Demo the effects
    demo_effects()
    
    # Apply to your textbook image
    result = create_printer_drum_effect('output_text_layout.jpg', 'output_with_drum_effect.jpg')
    # 
    # Or use the advanced effect
    # result = create_advanced_drum_effect('textbook_page.jpg', 'output_advanced_effect.jpg')