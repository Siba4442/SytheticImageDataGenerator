import json
import os
from PIL import Image
from surya.layout import LayoutPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuryaLayoutProcessor:
    """
    A class to handle layout detection using Surya and store metadata for each detected box.
    """
    
    def __init__(self):
        """Initialize the Surya predictors"""
        logger.info("Initializing Surya predictors...")
        self.layout_predictor = LayoutPredictor()
        self.detection_predictor = DetectionPredictor()
        self.recognition_predictor = RecognitionPredictor()
        logger.info("Predictors initialized successfully")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image and extract layout information with metadata
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict containing layout predictions with metadata
        """
        try:
            # Load image
            image = Image.open(image_path)
            logger.info(f"Processing image: {image_path}")
            
            # Get layout predictions
            layout_predictions = self.layout_predictor([image])
            
            # Extract metadata for each box
            metadata = self._extract_box_metadata(layout_predictions[0], image_path)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return {}
    
    def process_pdf(self, pdf_path: str, page_range: str = None) -> Dict[str, List[Dict]]:
        """
        Process a PDF document and extract layout information for all pages
        
        Args:
            pdf_path (str): Path to the PDF file
            page_range (str): Optional page range (e.g., "0,5-10,20")
            
        Returns:
            Dict with page numbers as keys and layout metadata as values
        """
        try:
            import fitz  # PyMuPDF
            
            # Open PDF
            doc = fitz.open(pdf_path)
            results = {}
            
            # Process specified pages or all pages
            pages_to_process = self._parse_page_range(page_range, len(doc))
            
            for page_num in pages_to_process:
                logger.info(f"Processing page {page_num + 1} of {pdf_path}")
                
                # Convert PDF page to image
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
                img_data = pix.tobytes("png")
                
                # Create PIL Image from bytes
                from io import BytesIO
                image = Image.open(BytesIO(img_data))
                
                # Get layout predictions
                layout_predictions = self.layout_predictor([image])
                
                # Extract metadata
                metadata = self._extract_box_metadata(
                    layout_predictions[0], 
                    f"{pdf_path}_page_{page_num + 1}"
                )
                
                results[f"page_{page_num + 1}"] = metadata
            
            doc.close()
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return {}
    
    def _extract_box_metadata(self, predictions: Dict, source_file: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from layout predictions
        
        Args:
            predictions (Dict): Layout predictions from Surya
            source_file (str): Source file path
            
        Returns:
            Dict containing structured metadata for each box
        """
        metadata = {
            'source_file': source_file,
            'processed_at': datetime.now().isoformat(),
            'total_boxes': len(predictions.get('bboxes', [])),
            'image_bbox': predictions.get('image_bbox'),
            'page_number': predictions.get('page', 0),
            'layout_elements': []
        }
        
        # Process each detected box
        for i, bbox_info in enumerate(predictions.get('bboxes', [])):
            box_metadata = {
                'box_id': i,
                'label': bbox_info.get('label', 'Unknown'),
                'confidence': bbox_info.get('confidence', 0.0),
                'bbox': bbox_info.get('bbox'),  # [x1, y1, x2, y2]
                'polygon': bbox_info.get('polygon'),  # [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                'position': bbox_info.get('position', -1),  # Reading order
                'top_k_predictions': bbox_info.get('top_k', {}),  # Alternative labels
                'area': self._calculate_area(bbox_info.get('bbox')),
                'dimensions': self._get_dimensions(bbox_info.get('bbox')),
                'center_point': self._get_center_point(bbox_info.get('bbox'))
            }
            
            metadata['layout_elements'].append(box_metadata)
        
        # Sort by reading order
        metadata['layout_elements'].sort(key=lambda x: x.get('position', float('inf')))
        
        return metadata
    
    def _calculate_area(self, bbox: List[float]) -> float:
        """Calculate area of bounding box"""
        if not bbox or len(bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = bbox
        return abs((x2 - x1) * (y2 - y1))
    
    def _get_dimensions(self, bbox: List[float]) -> Dict[str, float]:
        """Get width and height of bounding box"""
        if not bbox or len(bbox) != 4:
            return {'width': 0.0, 'height': 0.0}
        x1, y1, x2, y2 = bbox
        return {
            'width': abs(x2 - x1),
            'height': abs(y2 - y1)
        }
    
    def _get_center_point(self, bbox: List[float]) -> Dict[str, float]:
        """Get center point of bounding box"""
        if not bbox or len(bbox) != 4:
            return {'x': 0.0, 'y': 0.0}
        x1, y1, x2, y2 = bbox
        return {
            'x': (x1 + x2) / 2,
            'y': (y1 + y2) / 2
        }
    
    def _parse_page_range(self, page_range: str, total_pages: int) -> List[int]:
        """Parse page range string into list of page numbers"""
        if not page_range:
            return list(range(total_pages))
        
        pages = []
        for part in page_range.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                pages.extend(range(start, min(end + 1, total_pages)))
            else:
                page_num = int(part)
                if page_num < total_pages:
                    pages.append(page_num)
        
        return sorted(list(set(pages)))
    
    def save_metadata_json(self, metadata: Dict, output_path: str):
        """Save metadata to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving metadata to {output_path}: {str(e)}")
    
    def save_metadata_csv(self, metadata: Dict, output_path: str):
        """Save metadata to CSV file (flattened structure)"""
        try:
            rows = []
            
            if 'layout_elements' in metadata:
                # Single image/page
                for element in metadata['layout_elements']:
                    row = {
                        'source_file': metadata.get('source_file'),
                        'processed_at': metadata.get('processed_at'),
                        'page_number': metadata.get('page_number', 0),
                        'box_id': element.get('box_id'),
                        'label': element.get('label'),
                        'confidence': element.get('confidence'),
                        'position': element.get('position'),
                        'bbox_x1': element.get('bbox', [0,0,0,0])[0] if element.get('bbox') else 0,
                        'bbox_y1': element.get('bbox', [0,0,0,0])[1] if element.get('bbox') else 0,
                        'bbox_x2': element.get('bbox', [0,0,0,0])[2] if element.get('bbox') else 0,
                        'bbox_y2': element.get('bbox', [0,0,0,0])[3] if element.get('bbox') else 0,
                        'area': element.get('area'),
                        'width': element.get('dimensions', {}).get('width'),
                        'height': element.get('dimensions', {}).get('height'),
                        'center_x': element.get('center_point', {}).get('x'),
                        'center_y': element.get('center_point', {}).get('y')
                    }
                    rows.append(row)
            else:
                # Multiple pages
                for page_key, page_data in metadata.items():
                    if isinstance(page_data, dict) and 'layout_elements' in page_data:
                        for element in page_data['layout_elements']:
                            row = {
                                'source_file': page_data.get('source_file'),
                                'processed_at': page_data.get('processed_at'),
                                'page_number': page_data.get('page_number', 0),
                                'box_id': element.get('box_id'),
                                'label': element.get('label'),
                                'confidence': element.get('confidence'),
                                'position': element.get('position'),
                                'bbox_x1': element.get('bbox', [0,0,0,0])[0] if element.get('bbox') else 0,
                                'bbox_y1': element.get('bbox', [0,0,0,0])[1] if element.get('bbox') else 0,
                                'bbox_x2': element.get('bbox', [0,0,0,0])[2] if element.get('bbox') else 0,
                                'bbox_y2': element.get('bbox', [0,0,0,0])[3] if element.get('bbox') else 0,
                                'area': element.get('area'),
                                'width': element.get('dimensions', {}).get('width'),
                                'height': element.get('dimensions', {}).get('height'),
                                'center_x': element.get('center_point', {}).get('x'),
                                'center_y': element.get('center_point', {}).get('y')
                            }
                            rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"CSV metadata saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV to {output_path}: {str(e)}")
    
    def generate_summary_report(self, metadata: Dict) -> Dict[str, Any]:
        """Generate a summary report of the layout analysis"""
        summary = {
            'total_elements': 0,
            'label_distribution': {},
            'confidence_stats': {
                'mean': 0.0,
                'min': 1.0,
                'max': 0.0
            },
            'area_stats': {
                'total_area': 0.0,
                'mean_area': 0.0,
                'largest_element': None,
                'smallest_element': None
            }
        }
        
        all_elements = []
        
        # Collect all elements
        if 'layout_elements' in metadata:
            all_elements = metadata['layout_elements']
        else:
            for page_data in metadata.values():
                if isinstance(page_data, dict) and 'layout_elements' in page_data:
                    all_elements.extend(page_data['layout_elements'])
        
        if not all_elements:
            return summary
        
        # Calculate statistics
        summary['total_elements'] = len(all_elements)
        
        confidences = [elem.get('confidence', 0) for elem in all_elements]
        areas = [elem.get('area', 0) for elem in all_elements]
        
        # Label distribution
        for elem in all_elements:
            label = elem.get('label', 'Unknown')
            summary['label_distribution'][label] = summary['label_distribution'].get(label, 0) + 1
        
        # Confidence statistics
        if confidences:
            summary['confidence_stats'] = {
                'mean': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences)
            }
        
        # Area statistics
        if areas:
            summary['area_stats'] = {
                'total_area': sum(areas),
                'mean_area': sum(areas) / len(areas),
                'largest_element': max(all_elements, key=lambda x: x.get('area', 0)),
                'smallest_element': min(all_elements, key=lambda x: x.get('area', float('inf')))
            }
        
        return summary


# Example usage
def main():
    """Example usage of the SuryaLayoutProcessor"""
    
    # Initialize processor
    processor = SuryaLayoutProcessor()
    
    # Example 1: Process a single image
    image_path = "image14.png"
    if os.path.exists(image_path):
        metadata = processor.process_image(image_path)
        
        # Save results
        processor.save_metadata_json(metadata, "image_layout_metadata.json")
        processor.save_metadata_csv(metadata, "image_layout_metadata.csv")
        
        # Generate summary
        summary = processor.generate_summary_report(metadata)
        print("Layout Analysis Summary:")
        print(json.dumps(summary, indent=2))
    
    # Example 2: Process a PDF
    # pdf_path = "path/to/your/document.pdf"
    # if os.path.exists(pdf_path):
    #     # Process specific pages (optional)
    #     pdf_metadata = processor.process_pdf(pdf_path, page_range="0-2,5")
        
    #     # Save results
    #     processor.save_metadata_json(pdf_metadata, "pdf_layout_metadata.json")
    #     processor.save_metadata_csv(pdf_metadata, "pdf_layout_metadata.csv")
        
    #     # Generate summary
    #     summary = processor.generate_summary_report(pdf_metadata)
    #     print("PDF Layout Analysis Summary:")
    #     print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()