import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from doclayout_yolo import YOLOv10

class DocumentLayoutOCRPipeline:
    def __init__(self, model_path, tesseract_path=None):
        """
        Initialize the pipeline with YOLO model and Tesseract OCR
        
        Args:
            model_path: Path to the YOLO model
            tesseract_path: Path to tesseract executable (if needed)
        """
        self.model = YOLOv10(model_path)
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def detect_layout(self, image_path, imgsz=1024, conf=0.2):
        """
        Detect layout elements using YOLO
        
        Args:
            image_path: Path to input image
            imgsz: Image size for inference
            conf: Confidence threshold
            
        Returns:
            List of detection metadata
        """
        results = self.model.predict(image_path, imgsz=imgsz, conf=conf)
        
        metadata = []
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = bbox
                
                confidence = boxes.conf[i].cpu().numpy()
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detection_metadata = {
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    },
                    "dimensions": {
                        "width": float(width),
                        "height": float(height)
                    },
                    "center": {
                        "x": float(center_x),
                        "y": float(center_y)
                    }
                }
                metadata.append(detection_metadata)
        
        return metadata
    
    def extract_layout_regions(self, image_path, metadata):
        """
        Extract individual layout regions from the original image
        
        Args:
            image_path: Path to original image
            metadata: Layout detection metadata
            
        Returns:
            List of cropped region images and their metadata
        """
        image = cv2.imread(image_path)
        regions = []
        
        for i, item in enumerate(metadata):
            bbox = item['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # Crop the region
            region = image[y1:y2, x1:x2]
            
            regions.append({
                'image': region,
                'metadata': item,
                'region_id': i
            })
        
        return regions
    
    def perform_ocr_on_regions(self, regions, ocr_config='--psm 6'):
        """
        Perform OCR on each layout region
        
        Args:
            regions: List of region dictionaries
            ocr_config: Tesseract configuration
            
        Returns:
            List of regions with OCR text added
        """
        for region in regions:
            # Convert BGR to RGB for Tesseract
            rgb_image = cv2.cvtColor(region['image'], cv2.COLOR_BGR2RGB)
            
            # Perform OCR
            try:
                text = pytesseract.image_to_string(rgb_image, config=ocr_config)
                region['ocr_text'] = text.strip()
            except Exception as e:
                print(f"OCR failed for region {region['region_id']}: {e}")
                region['ocr_text'] = ""
        
        return regions
    def create_text_image(self, original_image_path, regions_with_text, output_path):
        """
        Create a new image with text placed at original layout locations
        
        Args:
            original_image_path: Path to original image
            regions_with_text: Regions with OCR text
            output_path: Path to save the output image
        """
        # Load original image to get dimensions
        original = cv2.imread(original_image_path)
        height, width = original.shape[:2]
        
        # Create a new blank image (white background)
        new_image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(new_image)
        
        # Try to load a font; fallback to default if not available
        try:
            font_path = "arial.ttf"
        except:
            font_path = None
        
        for region in regions_with_text:
            if not region['ocr_text']:
                continue
                
            bbox = region['metadata']['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            max_width = x2 - x1
            max_height = y2 - y1
            
            # Find the best font size
            best_size = self._find_best_font_size(region['ocr_text'], max_width, max_height)
            font = ImageFont.truetype(font_path, best_size)
            
            # Wrap text with the best font size
            lines = self._wrap_text(region['ocr_text'], max_width, font)
            
            # Calculate line height
            line_height = best_size * 1.2
            
            # Draw text lines
            y_offset = y1
            for i, line in enumerate(lines):
                draw.text((x1, y_offset), line, fill='black', font=font)
                y_offset += line_height
        
        # Save the new image
        new_image.save(output_path)
        print(f"Text image saved to: {output_path}")
    
    def _wrap_text(self, text, max_width, font):
        """
        Wrap text to fit within specified width
        
        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
            font: Font object
            
        Returns:
            List of text lines
        """
        words = text.split()
        lines = []
        current_line = []
        
        dummy_image = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_image)
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)  # Word is too long
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _find_best_font_size(self, text, max_width, max_height):
        """
        Find the largest font size that fits the text within the given dimensions
        
        Args:
            text: Text to fit
            max_width: Maximum width of the bounding box
            max_height: Maximum height of the bounding box
            
        Returns:
            Best font size
        """
        low = 6  # Minimum font size
        high = 100  # Maximum font size
        best_size = low
        
        dummy_image = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_image)
        
        while low <= high:
            mid = (low + high) // 2
            font = ImageFont.truetype("arial.ttf", mid)
            lines = self._wrap_text(text, max_width, font)
            line_height = mid * 1.2  # Line height factor of 1.2
            total_height = len(lines) * line_height
            
            if total_height <= max_height:
                best_size = mid
                low = mid + 1  # Try larger
            else:
                high = mid - 1  # Try smaller
        
        return best_size
    
    def run_full_pipeline(self, image_path, output_path="text_layout_image.jpg", 
                         save_intermediate=True):
        """
        Run the complete pipeline from layout detection to text image creation
        
        Args:
            image_path: Input image path
            output_path: Output image path
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary with all results
        """
        print("Step 1: Detecting layout...")
        metadata = self.detect_layout(image_path)
        print(f"Detected {len(metadata)} layout elements")
        
        if save_intermediate:
            with open("layout_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        print("Step 2: Extracting layout regions...")
        regions = self.extract_layout_regions(image_path, metadata)
        
        print("Step 3: Performing OCR on regions...")
        regions_with_text = self.perform_ocr_on_regions(regions)
        
        # Print OCR results
        for i, region in enumerate(regions_with_text):
            print(f"Region {i} ({region['metadata']['class_name']}): "
                  f"{region['ocr_text'][:100]}{'...' if len(region['ocr_text']) > 100 else ''}")
        
        if save_intermediate:
            # Save OCR results
            ocr_results = []
            for region in regions_with_text:
                ocr_results.append({
                    'region_id': region['region_id'],
                    'class_name': region['metadata']['class_name'],
                    'bbox': region['metadata']['bbox'],
                    'ocr_text': region['ocr_text']
                })
            
            with open("ocr_results.json", "w") as f:
                json.dump(ocr_results, f, indent=2)
        
        print("Step 4: Creating text layout image...")
        self.create_text_image(image_path, regions_with_text, output_path)
        
        return {
            'metadata': metadata,
            'regions_with_text': regions_with_text,
            'output_path': output_path
        }

# Usage example
if __name__ == "__main__":
    # Initialize the pipeline
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pipeline = DocumentLayoutOCRPipeline("models/doclayout_yolo_docstructbench_imgsz1024.pt")
    # Run the complete pipeline
    results = pipeline.run_full_pipeline("2025.lm4uc-1.11_page-0001.jpg", "output_text_layout.jpg")
    
    print("Pipeline completed successfully!")
    print(f"Output saved to: {results['output_path']}")