from PIL import Image, ImageFilter
import numpy as np

def drop_shadow_effect(image_path, output_path, shadow_offset=4, shadow_blur=4, shadow_opacity=0.6):
    # Load the image (ensure RGBA for transparency)
    img = Image.open(image_path).convert('RGBA')
    width, height = img.size

    # Extract alpha channel to create shadow
    alpha = img.split()[3]
    alpha_array = np.array(alpha)

    # Create shadow image
    shadow = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    shadow_array = np.zeros((height, width, 4), dtype=np.uint8)

    # Offset alpha for shadow
    offset_x, offset_y = shadow_offset, shadow_offset
    for y in range(height):
        for x in range(width):
            if y - offset_y >= 0 and x - offset_x >= 0:
                if alpha_array[y, x] > 0:
                    shadow_array[y - offset_y, x - offset_x, 3] = int(alpha_array[y, x] * shadow_opacity)

    shadow = Image.fromarray(shadow_array)
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))  # Apply blur to shadow

    # Combine shadow and original image
    result = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    result.paste(shadow, (0, 0), shadow)  # Apply shadow
    result.paste(img, (0, 0), img)  # Overlay original image

    # Save the result
    result.save(output_path)

# Example usage
if __name__ == "__main__":
    input_image = "output_text_layout.jpg"  # Replace with your image path
    output_image = "shadow_text_image.png"
    drop_shadow_effect(input_image, output_image, shadow_offset=4, shadow_blur=4, shadow_opacity=0.6)