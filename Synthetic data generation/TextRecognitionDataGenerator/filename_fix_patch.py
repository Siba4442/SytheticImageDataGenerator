import os
import shutil
import re
from pathlib import Path

def create_improved_data_generator():
    """
    Create an improved version of data_generator.py that handles long filenames properly
    """
    
    # The improved data generator code with proper filename handling
    improved_code = '''import os
import random as rnd
import hashlib

from PIL import Image, ImageFilter, ImageStat

from trdg import computer_text_generator, background_generator, distorsion_generator
from trdg.utils import mask_to_bboxes, make_filename_valid

try:
    from trdg import handwritten_text_generator
except ImportError as e:
    print("Missing modules for handwritten text generation.")


def create_safe_filename(text, max_length=50):
    """Create a safe filename from text that won't exceed filesystem limits"""
    # Create hash for uniqueness
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    
    # Clean and truncate the text
    # Remove problematic characters and limit length
    clean_text = re.sub(r'[<>:"/\\|?*]', '', text)  # Remove invalid filename chars
    clean_text = clean_text.replace(' ', '_')[:max_length]  # Replace spaces and truncate
    
    # If text is empty after cleaning, use hash only
    if not clean_text:
        return f"text_{text_hash}"
    
    return f"{clean_text}_{text_hash}"


class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
        Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(
        cls,
        index: int,
        text: str,
        font: str,
        out_dir: str,
        size: int,
        extension: str,
        skewing_angle: int,
        random_skew: bool,
        blur: int,
        random_blur: bool,
        background_type: int,
        distorsion_type: int,
        distorsion_orientation: int,
        is_handwritten: bool,
        name_format: int,
        width: int,
        alignment: int,
        text_color: str,
        orientation: int,
        space_width: int,
        character_spacing: int,
        margins: int,
        fit: bool,
        output_mask: bool,
        word_split: bool,
        image_dir: str,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
        image_mode: str = "RGB",
        output_bboxes: int = 0,
    ) -> Image:
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image, mask = handwritten_text_generator.generate(text, text_color)
        else:
            image, mask = computer_text_generator.generate(
                text,
                font,
                text_color,
                size,
                orientation,
                space_width,
                character_spacing,
                fit,
                word_split,
                stroke_width,
                stroke_fill,
            )
        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distortion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img  # Mind = blown
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
            distorted_img, distorted_mask = distorsion_generator.sin(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img, distorted_mask = distorsion_generator.cos(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img, distorted_mask = distorsion_generator.random(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(
                distorted_img.size[0]
                * (float(size - vertical_margin) / float(distorted_img.size[1]))
            )
            resized_img = distorted_img.resize(
                (new_width, size - vertical_margin), Image.Resampling.LANCZOS
            )
            resized_mask = distorted_mask.resize(
                (new_width, size - vertical_margin), Image.Resampling.NEAREST
            )
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(distorted_img.size[1])
                * (float(size - horizontal_margin) / float(distorted_img.size[0]))
            )
            resized_img = distorted_img.resize(
                (size - horizontal_margin, new_height), Image.Resampling.LANCZOS
            )
            resized_mask = distorted_mask.resize(
                (size - horizontal_margin, new_height), Image.Resampling.NEAREST
            )
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background_img = background_generator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background_img = background_generator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background_img = background_generator.quasicrystal(
                background_height, background_width
            )
        else:
            background_img = background_generator.image(
                background_height, background_width, image_dir
            )
        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )

        ##############################################################
        # Comparing average pixel value of text and background image #
        ##############################################################
        try:
            resized_img_st = ImageStat.Stat(resized_img, resized_mask.split()[2])
            background_img_st = ImageStat.Stat(background_img)

            resized_img_px_mean = sum(resized_img_st.mean[:2]) / 3
            background_img_px_mean = sum(background_img_st.mean) / 3

            if abs(resized_img_px_mean - background_img_px_mean) < 15:
                print("value of mean pixel is too similar. Ignore this image")

                print("resized_img_st \\n {}".format(resized_img_st.mean))
                print("background_img_st \\n {}".format(background_img_st.mean))

                return
        except Exception as err:
            return

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        if alignment == 0 or width == -1:
            background_img.paste(resized_img, (margin_left, margin_top), resized_img)
            background_mask.paste(resized_mask, (margin_left, margin_top))
        elif alignment == 1:
            background_img.paste(
                resized_img,
                (int(background_width / 2 - new_text_width / 2), margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (int(background_width / 2 - new_text_width / 2), margin_top),
            )
        else:
            background_img.paste(
                resized_img,
                (background_width - new_text_width - margin_right, margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (background_width - new_text_width - margin_right, margin_top),
            )

        ############################################
        # Change image mode (RGB, grayscale, etc.) #
        ############################################

        background_img = background_img.convert(image_mode)
        background_mask = background_mask.convert(image_mode)

        #######################
        # Apply gaussian blur #
        #######################

        gaussian_filter = ImageFilter.GaussianBlur(
            radius=blur if not random_blur else rnd.random() * blur
        )
        final_image = background_img.filter(gaussian_filter)
        final_mask = background_mask.filter(gaussian_filter)

        #####################################
        # Generate name for resulting image #
        #####################################
        # We remove spaces if space_width == 0
        if space_width == 0:
            text = text.replace(" ", "")
            
        # Use safe filename generation instead of raw text
        if name_format == 0:
            name = "{}_{}" .format(create_safe_filename(text), str(index))
        elif name_format == 1:
            name = "{}_{}" .format(str(index), create_safe_filename(text))
        elif name_format == 2:
            name = str(index)
        else:
            print("{} is not a valid name format. Using default.".format(name_format))
            name = "{}_{}" .format(create_safe_filename(text), str(index))

        # Additional safety check - make sure the name is still valid
        name = make_filename_valid(name, allow_unicode=True)
        
        # Ensure total filename length doesn't exceed 200 characters (leaving room for extension)
        if len(name) > 200:
            name = name[:200]
            
        image_name = "{}.{}".format(name, extension)
        mask_name = "{}_mask.png".format(name)
        box_name = "{}_boxes.txt".format(name)
        tess_box_name = "{}.box".format(name)

        # Save the image
        if out_dir is not None:
            final_image.save(os.path.join(out_dir, image_name))
            if output_mask == 1:
                final_mask.save(os.path.join(out_dir, mask_name))
            if output_bboxes == 1:
                bboxes = mask_to_bboxes(final_mask)
                with open(os.path.join(out_dir, box_name), "w") as f:
                    for bbox in bboxes:
                        f.write(" ".join([str(v) for v in bbox]) + "\\n")
            if output_bboxes == 2:
                bboxes = mask_to_bboxes(final_mask, tess=True)
                with open(os.path.join(out_dir, tess_box_name), "w") as f:
                    for bbox, char in zip(bboxes, text):
                        f.write(
                            " ".join([char] + [str(v) for v in bbox] + ["0"]) + "\\n"
                        )
        else:
            if output_mask == 1:
                return final_image, final_mask
            return final_image
'''
    
    return improved_code

def apply_patch():
    """Apply the patch to fix filename length issues"""
    
    # Find the data_generator.py file
    trdg_path = Path("trdg/data_generator.py")
    
    if not trdg_path.exists():
        print("❌ data_generator.py not found. Make sure you're in the TextRecognitionDataGenerator directory.")
        return False
    
    # Create backup
    backup_path = trdg_path.with_suffix('.py.backup')
    if not backup_path.exists():
        shutil.copy2(trdg_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
    
    # Write the improved version
    try:
        with open(trdg_path, 'w', encoding='utf-8') as f:
            f.write(create_improved_data_generator())
        
        print("✅ Successfully patched data_generator.py")
        print("🔧 Added safe filename generation with the following improvements:")
        print("   • Limits filename length to prevent filesystem errors")
        print("   • Uses hash for uniqueness while keeping readable text")
        print("   • Removes invalid filename characters")
        print("   • Handles Unicode text (Odia) properly")
        return True
        
    except Exception as e:
        print(f"❌ Error writing patched file: {str(e)}")
        return False

def restore_backup():
    """Restore the original file from backup"""
    
    trdg_path = Path("trdg/data_generator.py")
    backup_path = trdg_path.with_suffix('.py.backup')
    
    if backup_path.exists():
        shutil.copy2(backup_path, trdg_path)
        print("✅ Restored original data_generator.py from backup")
        return True
    else:
        print("❌ No backup file found")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_backup()
    else:
        print("🔧 Applying patch to fix filename length issues...")
        if apply_patch():
            print("\\n🎉 Patch applied successfully!")
            print("\\n📝 You can now run your original command:")
            print("   python run.py -l or -c 10 -w 500 -f 64")
            print("\\n🔄 To restore the original file, run:")
            print("   python filename_fix_patch.py restore")
        else:
            print("\\n❌ Patch failed. Please check the error messages above.")