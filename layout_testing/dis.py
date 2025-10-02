from PIL import Image
import numpy as np

def drum_distortion(image_path, output_path, distortion_strength=0.3):
    # Load the image
    img = Image.open(image_path)
    width, height = img.size
    img_array = np.array(img)

    # Create output array
    output_array = np.zeros_like(img_array)

    # Center of the image
    cx, cy = width / 2, height / 2

    # Apply drum-like distortion
    for y in range(height):
        for x in range(width):
            # Normalize coordinates to [-1, 1]
            nx = (x - cx) / cx
            ny = (y - cy) / cy

            # Calculate radial distance
            r = np.sqrt(nx**2 + ny**2)

            # Apply distortion (barrel-like for drum effect)
            if r != 0:
                # Distortion formula: scales radius to create curved effect
                distortion = 1 + distortion_strength * (r**2)
                new_r = r * distortion
                theta = np.arctan2(ny, nx)

                # Map back to pixel coordinates
                new_x = cx + (new_r * cx * np.cos(theta))
                new_y = cy + (new_r * cy * np.sin(theta))

                # Ensure new coordinates are within bounds
                if 0 <= new_x < width and 0 <= new_y < height:
                    # Use nearest-neighbor interpolation
                    output_array[y, x] = img_array[int(new_y), int(new_x)]
                else:
                    output_array[y, x] = [255, 255, 255] if img_array.shape[-1] == 3 else [255, 255, 255, 255]
            else:
                output_array[y, x] = img_array[y, x]

    # Convert back to image and save
    distorted_img = Image.fromarray(output_array)
    distorted_img.save(output_path)

# Example usage
if __name__ == "__main__":
    input_image = "output_text_layout.jpg"  # Replace with your image path
    output_image = "distorted_text_image.png"
    drum_distortion(input_image, output_image, distortion_strength=0.3)