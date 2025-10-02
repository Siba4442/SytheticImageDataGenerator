#!/usr/bin/env python3
"""
Document Layout Analysis and OCR Pipeline

This script uses a YOLO model from the ultralytics library to detect layout 
elements in a document image, performs OCR on each detected region using 
Tesseract, and reconstructs the document with the extracted text placed 
in its original layout.

This script is self-contained and does not require any helper files.

Command-Line Usage:
    python <script_name>.py --model_path <path_to_model> --image_path <path_to_image> [options]

Example:
    python document_pipeline.py \
        --model_path "models/doclayout_yolo_docstructbench_imgsz1024.pt" \
        --image_path "input_document.jpg" \
        --output_path "output_layout.jpg" \
        --tesseract_path r"C:\Program Files\Tesseract-OCR\tesseract.exe"
"""
import cv2
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import sys
import os

# Add the current directory to the system path.
# This is necessary to ensure that the ultralytics library can find the
# custom 'doclayout_yolo' module required by the specific model you are using.
# This module is not a standard package and is expected to be in the same
# directory as the script.
sys.path.append(os.getcwd())

# Import YOLO directly from the ultralytics library
from ultralytics import YOLO

class DocumentLayoutOCRPipeline:
    """
    A class to encapsulate the document layout detection and OCR pipeline.
    """
    def __init__(self, model_path, tesseract_path=None):
        """
        Initialize the pipeline with YOLO model and Tesseract OCR.
        
        Args:
            model_path (str): Path to the YOLOv10 model file.
            tesseract_path (str, optional): Path to the Tesseract executable. Defaults to None.
        """
        # The YOLO class from ultralytics can load any YOLO version, including YOLOv10
        self.model = YOLO(model_path)
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def detect_layout(self, image_path, imgsz=1024, conf=0.2):
        """
        Detect layout elements in an image using the YOLO model.
        
        Args:
            image_path (str): Path to the input image.
            imgsz (int): Image size for inference.
            conf (float): Confidence threshold for detection.
            
        Returns:
            list: A list of dictionaries, where each dictionary contains metadata for a detected layout element.
        """
        results = self.model.predict(image_path, imgsz=imgsz, conf=conf)
        
        metadata = []
        if not results:
            return metadata

        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = bbox
                
                confidence = boxes.conf[i].cpu().numpy()
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detection_metadata = {
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2)
                    }
                }
                metadata.append(detection_metadata)
        
        return metadata
    
    def extract_layout_regions(self, image_path, metadata):
        """
        Extract individual layout regions from the original image based on detection metadata.
        
        Args:
            image_path (str): Path to the original image.
            metadata (list): Layout detection metadata from `detect_layout`.
            
        Returns:
            list: A list of dictionaries, each containing the cropped region image, its metadata, and a region ID.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image from path: {image_path}")
        regions = []
        
        for i, item in enumerate(metadata):
            bbox = item['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # Crop the region from the image
            region_img = image[y1:y2, x1:x2]
            
            regions.append({
                'image': region_img,
                'metadata': item,
                'region_id': i
            })
        
        return regions
    
    def perform_ocr_on_regions(self, regions, ocr_config='--psm 6'):
        """
        Perform OCR on each extracted layout region.
        
        Args:
            regions (list): A list of region dictionaries from `extract_layout_regions`.
            ocr_config (str): Configuration string for Tesseract OCR.
            
        Returns:
            list: The input list of regions, with the 'ocr_text' key added to each dictionary.
        """
        for region in regions:
            # Convert image from BGR (OpenCV) to RGB (Tesseract/Pillow)
            rgb_image = cv2.cvtColor(region['image'], cv2.COLOR_BGR2RGB)
            
            try:
                text = pytesseract.image_to_string(rgb_image, config=ocr_config)
                region['ocr_text'] = text.strip()
            except Exception as e:
                print(f"OCR failed for region {region['region_id']} ({region['metadata']['class_name']}): {e}")
                region['ocr_text'] = ""
        
        return regions

    def create_text_image(self, original_image_path, regions_with_text, output_path):
        """
        Create a new blank image and draw the OCR'd text onto it, preserving the original layout.
        
        Args:
            original_image_path (str): Path to the original image to get dimensions.
            regions_with_text (list): List of regions with OCR text.
            output_path (str): Path to save the generated output image.
        """
        original = cv2.imread(original_image_path)
        height, width = original.shape[:2]
        
        # Create a new blank image with a white background
        new_image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(new_image)
        
        for region in regions_with_text:
            if not region['ocr_text']:
                continue
                
            bbox = region['metadata']['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            max_width = x2 - x1
            max_height = y2 - y1
            
            # Find the best font size that fits the text within the bounding box
            best_size = self._find_best_font_size(region['ocr_text'], max_width, max_height)
            try:
                font = ImageFont.truetype("arial.ttf", best_size)
            except IOError:
                print("Arial font not found. Using default font.")
                font = ImageFont.load_default()

            # Wrap text to fit within the bounding box width
            lines = self._wrap_text(region['ocr_text'], max_width, font)
            
            # Draw the wrapped text lines onto the new image
            line_height = best_size * 1.2  # Approximate line height
            y_offset = y1
            for line in lines:
                if y_offset + line_height > y2:
                    break # Stop if text overflows the bounding box height
                draw.text((x1, y_offset), line, fill='black', font=font)
                y_offset += line_height
        
        # Save the final image
        new_image.save(output_path)
        print(f"Text layout image saved to: {output_path}")

    def _wrap_text(self, text, max_width, font):
        """
        Wrap a string of text to fit within a specified maximum width.
        
        Args:
            text (str): The text to wrap.
            max_width (int): The maximum width in pixels for a line.
            font (ImageFont): The font object used for measuring text.
            
        Returns:
            list: A list of strings, where each string is a wrapped line.
        """
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Check width if this word is added to the current line
            test_line = ' '.join(current_line + [word])
            # Use font.getbbox for modern Pillow versions
            bbox = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                # Add the current line to lines and start a new one
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
            
        return lines

    def _find_best_font_size(self, text, max_width, max_height):
        """
        Find the largest font size that fits the text within a bounding box.
        
        Args:
            text (str): The text to fit.
            max_width (int): Maximum width of the bounding box.
            max_height (int): Maximum height of the bounding box.
            
        Returns:
            int: The optimal font size.
        """
        low, high = 6, 120  # Min and max font sizes to try
        best_size = low
        
        while low <= high:
            mid = (low + high) // 2
            try:
                font = ImageFont.truetype("arial.ttf", mid)
            except IOError:
                font = ImageFont.load_default()

            lines = self._wrap_text(text, max_width, font)
            line_height = mid * 1.2
            total_height = len(lines) * line_height
            
            if total_height <= max_height:
                best_size = mid
                low = mid + 1  # Try a larger size
            else:
                high = mid - 1  # Try a smaller size
        
        return best_size

    def run_full_pipeline(self, image_path, output_path, imgsz, conf, save_intermediate):
        """
        Run the complete pipeline from layout detection to final text image creation.
        
        Args:
            image_path (str): Input image path.
            output_path (str): Output image path.
            imgsz (int): Image size for inference.
            conf (float): Confidence threshold for detection.
            save_intermediate (bool): Whether to save intermediate JSON results.
            
        Returns:
            dict: A dictionary containing the final results.
        """
        print("Step 1: Detecting layout...")
        metadata = self.detect_layout(image_path, imgsz=imgsz, conf=conf)
        print(f"Detected {len(metadata)} layout elements.")
        
        if save_intermediate:
            with open("layout_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            print("Saved layout metadata to layout_metadata.json")
        
        print("Step 2: Extracting layout regions...")
        regions = self.extract_layout_regions(image_path, metadata)
        
        print("Step 3: Performing OCR on regions...")
        regions_with_text = self.perform_ocr_on_regions(regions)
        
        if save_intermediate:
            ocr_results = [{
                'region_id': r['region_id'],
                'class_name': r['metadata']['class_name'],
                'bbox': r['metadata']['bbox'],
                'ocr_text': r['ocr_text']
            } for r in regions_with_text]
            with open("ocr_results.json", "w") as f:
                json.dump(ocr_results, f, indent=2)
            print("Saved OCR results to ocr_results.json")

        print("Step 4: Creating final text layout image...")
        self.create_text_image(image_path, regions_with_text, output_path)
        
        return {
            'metadata': metadata,
            'regions_with_text': regions_with_text,
            'output_path': output_path
        }

def main():
    """
    Main function to parse command-line arguments and run the pipeline.
    """
    parser = argparse.ArgumentParser(description="Document Layout OCR Pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input document image.")
    parser.add_argument("--output_path", type=str, default="output_text_layout.jpg", help="Path to save the output image.")
    parser.add_argument("--tesseract_path", type=str, default=None, help="Optional path to the Tesseract executable.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for YOLO inference.")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold for YOLO detection.")
    parser.add_argument("--no_save_intermediate", action="store_true", help="Disable saving of intermediate JSON files (metadata and OCR results).")

    args = parser.parse_args()
    
    try:
        pipeline = DocumentLayoutOCRPipeline(
            model_path=args.model_path,
            tesseract_path=args.tesseract_path
        )
        
        results = pipeline.run_full_pipeline(
            image_path=args.image_path,
            output_path=args.output_path,
            imgsz=args.imgsz,
            conf=args.conf,
            save_intermediate=not args.no_save_intermediate
        )
        
        print("\nPipeline completed successfully!")
        print(f"Output image saved to: {results['output_path']}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
