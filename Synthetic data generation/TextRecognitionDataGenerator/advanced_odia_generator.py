#!/usr/bin/env python3
"""
Advanced Odia OCR Data Generator with Configuration Support
Handles large datasets and provides extensive customization
"""

import os
import json
import yaml
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from trdg.generators import GeneratorFromStrings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

class OdiaOCRGenerator:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Odia OCR Generator
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.generated_count = 0
        self.lock = threading.Lock()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "output_dir": "odia_ocr_output",
            "count": 100,
            "font_size": 64,
            "font_paths": [],
            "background_type": 0,
            "distorsion_type": 0,
            "distorsion_orientation": 0,
            "is_handwritten": False,
            "width": -1,
            "alignment": 1,
            "text_color": "#282828",
            "orientation": 0,
            "space_width": 1.0,
            "character_spacing": 0,
            "margins": [5, 5, 5, 5],
            "fit": False,
            "output_mask": False,
            "word_split": False,
            "image_dir": None,
            "stroke_width": 0,
            "stroke_fill": "#282828",
            "image_mode": "RGB",
            "max_filename_length": 50,
            "parallel_workers": 4,
            "batch_size": 50,
            "strings_file": None,
            "custom_strings": []
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        config = self._default_config()
        
        if config_path.endswith('.json'):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
        elif config_path.endswith(('.yml', '.yaml')):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be JSON or YAML")
        
        config.update(user_config)
        return config
    
    def save_config(self, config_path: str):
        """Save current configuration to file"""
        if config_path.endswith('.json'):
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        elif config_path.endswith(('.yml', '.yaml')):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def _generate_safe_filename(self, text: str, index: int) -> str:
        """Generate safe filename avoiding Windows path length limits"""
        # Create hash for uniqueness
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        
        # Clean text for filename (keep only first few Odia characters)
        clean_text = ''.join(c for c in text[:20] if c.isalnum() or c in ' -_')
        clean_text = clean_text.replace(' ', '_')
        
        # Combine index, clean text, and hash
        base_name = f"odia_{index:06d}_{clean_text}_{text_hash}"
        
        # Ensure filename isn't too long
        max_len = self.config.get('max_filename_length', 50)
        if len(base_name) > max_len:
            base_name = f"odia_{index:06d}_{text_hash}"
        
        return base_name
    
    def _load_strings(self) -> List[str]:
        """Load strings from various sources"""
        strings = []
        
        # Load from file if specified
        if self.config.get('strings_file'):
            with open(self.config['strings_file'], 'r', encoding='utf-8') as f:
                strings.extend([line.strip() for line in f if line.strip()])
        
        # Add custom strings
        if self.config.get('custom_strings'):
            strings.extend(self.config['custom_strings'])
        
        # Default Odia strings if nothing provided
        if not strings:
            strings = [
                "କଣରକ",
                "ଭବନଶୱର", 
                "କଟକ",
                "ଓଡଶ",
                "ସମଲପଲ",
                "ପର",
                "ଚଲକ",
                "ରରକଲ",
                "ସପରଭତ",
                "ଜଗନ୍ନାଥ ମନ୍ଦିର",
                "କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର",
                "ଭରତ ଦଶ",
                "ଓଡଶ ରଜ୍ୟ",
                "କଳିଙ୍ଗ ଯୁଦ୍ଧ",
                "ଓଡ଼ିଆ ଭାଷା",
                "ବଙ୍ଗୋପସାଗର",
                "ମହାନଦୀ",
                "ଚଲେନ୍ଦ୍ର ସାଗର",
                "ପର ନୃସିଂହନାଥ",
                "ଲିଙ୍ଗରାଜ ମନ୍ଦିର"
            ]
        
        return strings
    
    def _create_output_directory(self):
        """Create output directory structure"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / "images").mkdir(exist_ok=True)
        (output_dir / "labels").mkdir(exist_ok=True)
        
        return output_dir
    
    def _generate_batch(self, strings: List[str], start_index: int, batch_size: int) -> List[tuple]:
        """Generate a batch of images"""
        batch_results = []
        
        try:
            generator = GeneratorFromStrings(
                strings=strings,
                count=batch_size,
                fonts=self.config.get('font_paths', []),
                language="or",
                size=self.config['font_size'],
                skewing_angle=0,
                random_skew=False,
                blur=0,
                random_blur=False,
                background_type=self.config['background_type'],
                distorsion_type=self.config['distorsion_type'],
                distorsion_orientation=self.config['distorsion_orientation'],
                is_handwritten=self.config['is_handwritten'],
                width=self.config['width'],
                alignment=self.config['alignment'],
                text_color=self.config['text_color'],
                orientation=self.config['orientation'],
                space_width=self.config['space_width'],
                character_spacing=self.config['character_spacing'],
                margins=self.config['margins'],
                fit=self.config['fit'],
                output_mask=self.config['output_mask'],
                word_split=self.config['word_split'],
                image_dir=self.config['image_dir'],
                stroke_width=self.config['stroke_width'],
                stroke_fill=self.config['stroke_fill'],
                image_mode=self.config['image_mode']
            )
            
            batch_index = 0
            for img, lbl in generator:
                current_index = start_index + batch_index
                filename = self._generate_safe_filename(lbl, current_index)
                batch_results.append((img, lbl, filename, current_index))
                batch_index += 1
                
                if batch_index >= batch_size:
                    break
                    
        except Exception as e:
            print(f"Error in batch starting at {start_index}: {str(e)}")
            
        return batch_results
    
    def generate(self) -> bool:
        """Generate OCR data"""
        print("Starting Odia OCR data generation...")
        
        # Setup
        output_dir = self._create_output_directory()
        strings = self._load_strings()
        total_count = self.config['count']
        batch_size = self.config['batch_size']
        
        print(f"Configuration:")
        print(f"  Total images: {total_count}")
        print(f"  Output directory: {output_dir}")
        print(f"  Font size: {self.config['font_size']}")
        print(f"  Batch size: {batch_size}")
        print(f"  Available strings: {len(strings)}")
        
        # Prepare labels file
        labels_file = output_dir / "labels" / "labels.txt"
        
        try:
            with open(labels_file, "w", encoding="utf-8") as label_f:
                # Process in batches
                with ThreadPoolExecutor(max_workers=self.config['parallel_workers']) as executor:
                    futures = []
                    
                    # Submit batch jobs
                    for start_idx in range(0, total_count, batch_size):
                        current_batch_size = min(batch_size, total_count - start_idx)
                        # Randomly sample strings for this batch
                        batch_strings = random.choices(strings, k=current_batch_size)
                        
                        future = executor.submit(
                            self._generate_batch, 
                            batch_strings, 
                            start_idx, 
                            current_batch_size
                        )
                        futures.append(future)
                    
                    # Process completed batches
                    with tqdm(total=total_count, desc="Generating images") as pbar:
                        for future in as_completed(futures):
                            try:
                                batch_results = future.result()
                                
                                # Save batch results
                                for img, lbl, filename, index in batch_results:
                                    # Save image
                                    image_path = output_dir / "images" / f"{filename}.jpg"
                                    img.save(str(image_path))
                                    
                                    # Write label
                                    label_f.write(f"{filename}.jpg\t{lbl}\n")
                                    
                                    with self.lock:
                                        self.generated_count += 1
                                        pbar.update(1)
                                        
                            except Exception as e:
                                print(f"Batch failed: {str(e)}")
            
            print(f"\nGeneration completed!")
            print(f"Generated {self.generated_count} images")
            print(f"Images saved in: {output_dir / 'images'}")
            print(f"Labels saved in: {labels_file}")
            
            return True
            
        except Exception as e:
            print(f"Generation failed: {str(e)}")
            return False

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "output_dir": "odia_ocr_data",
        "count": 1000,
        "font_size": 64,
        "font_paths": [
            # Add your Odia font paths here
            # "/path/to/odia_font1.ttf",
            # "/path/to/odia_font2.ttf"
        ],
        "background_type": 0,
        "distorsion_type": 1,
        "width": 500,
        "batch_size": 100,
        "parallel_workers": 4,
        "custom_strings": [
            "କଣରକ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ଏକ ପ୍ରସିଦ୍ଧ ମନ୍ଦିର",
            "ଭବନଶୱର ଓଡଶ ରଜ୍ୟର ରାଜଧାନୀ",
            "କଟକ ନଗର ଇତିହାସ ପ୍ରସିଦ୍ଧ",
            "ଜଗନ୍ନାଥ ମନ୍ଦିର ପରର ସବୁଠାରୁ ପବିତ୍ର ସ୍ଥାନ"
        ]
    }
    
    with open("odia_ocr_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("Sample configuration created: odia_ocr_config.json")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Odia OCR Data Generator")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--create-config", action="store_true", help="Create sample configuration file")
    parser.add_argument("--count", type=int, help="Number of images to generate")
    parser.add_argument("--output", help="Output directory")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    # Initialize generator
    generator = OdiaOCRGenerator(args.config)
    
    # Override config with command line args
    if args.count:
        generator.config['count'] = args.count
    if args.output:
        generator.config['output_dir'] = args.output
    
    # Generate data
    success = generator.generate()
    
    if success:
        print("Generation completed successfully!")
    else:
        print("Generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()