"""
Hugging Face dataset processor for downloading datasets and generating synthetic text images
"""

import os
import csv
import logging
import pandas as pd
from typing import Dict, List, Optional
from urllib.parse import urlparse
import requests
from PIL import Image

# Add datasets import
from datasets import load_dataset
import datasets

from .config import ENHANCED_DEFAULT_PARAMS
from .text_renderer import render_enhanced_sanskrit
from .transformations import apply_enhanced_postprocessing

logger = logging.getLogger(__name__)


class HuggingFaceDatasetProcessor:
    """Process Hugging Face datasets and generate synthetic text images"""
    
    def __init__(self, output_dir: str = "hf_dataset_output", params: Dict = None):
        self.output_dir = output_dir
        self.params = params if params else ENHANCED_DEFAULT_PARAMS.copy()
        self.image_dir = os.path.join(output_dir, "images")
        self.csv_path = os.path.join(output_dir, "dataset.csv")
        
        # Create output directories
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Default ink colors for different styles
        self.ink_colors = {
            "lined_paper": (60, 30, 10),
            "old_paper": (20, 20, 20),
            "birch": (50, 20, 10),
            "parchment": (10, 10, 10)
        }
        
        logger.info(f"HuggingFace dataset processor initialized. Output: {output_dir}")
    
    def load_huggingface_dataset(self, dataset_name: str, config_name: str = None, 
                                split: str = None, streaming: bool = False) -> Optional[pd.DataFrame]:
        """Load dataset using the datasets library"""
        try:
            logger.info(f"Loading Hugging Face dataset: {dataset_name}")
            
            # Load dataset using datasets library
            dataset = load_dataset(
                dataset_name, 
                config_name,
                split=split,
                streaming=streaming
            )
            
            # Convert to pandas DataFrame if not streaming
            if streaming:
                logger.info("Using streaming mode - dataset will be processed iteratively")
                return dataset
            else:
                # Handle different dataset structures
                if isinstance(dataset, datasets.DatasetDict):
                    # If multiple splits, use the first one or 'train' if available
                    if split:
                        df = dataset[split].to_pandas()
                    elif 'train' in dataset:
                        df = dataset['train'].to_pandas()
                    else:
                        # Use the first available split
                        first_split = list(dataset.keys())[0]
                        df = dataset[first_split].to_pandas()
                        logger.info(f"Using split: {first_split}")
                else:
                    # Single dataset
                    df = dataset.to_pandas()
                
                logger.info(f"Loaded dataset: {len(df)} rows")
                logger.info(f"Available columns: {list(df.columns)}")
                
                return df
                
        except Exception as e:
            logger.error(f"Error loading Hugging Face dataset: {e}")
            logger.info("Falling back to URL-based download method...")
            return None
    
    def download_dataset_from_url(self, url: str, output_file: str = "dataset.csv") -> bool:
        """Download dataset from a URL (fallback method)"""
        try:
            # Convert Hugging Face dataset URL to raw CSV URL if needed
            if "huggingface.co/datasets" in url:
                # Extract dataset path and convert to raw file URL
                parsed = urlparse(url)
                dataset_path = parsed.path.strip('/')
                
                # Common patterns for HF dataset URLs
                if "/blob/main/" in url:
                    raw_url = url.replace("/blob/main/", "/raw/main/")
                elif "/tree/main" in url:
                    raw_url = url.replace("/tree/main", "/raw/main/dataset.csv")
                else:
                    # Try to construct raw URL for CSV files
                    raw_url = f"https://huggingface.co/{dataset_path}/raw/main/dataset.csv"
            else:
                raw_url = url
            
            logger.info(f"Downloading dataset from: {raw_url}")
            
            # Download the file
            response = requests.get(raw_url, stream=True)
            response.raise_for_status()
            
            # Save to file
            file_path = os.path.join(self.output_dir, output_file)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Dataset downloaded successfully to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False
    
    def load_dataset(self, file_path: str, text_column: str) -> Optional[pd.DataFrame]:
        """Load dataset from CSV file and validate text column"""
        try:
            # Try reading with different encodings
            encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'utf-16']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully loaded dataset with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise Exception("Could not load dataset with any supported encoding")
            
            # Validate text column exists
            if text_column not in df.columns:
                raise Exception(f"Column '{text_column}' not found. Available columns: {list(df.columns)}")
            
            # Remove rows with empty text
            initial_rows = len(df)
            df = df.dropna(subset=[text_column])
            df = df[df[text_column].str.strip() != ""]
            
            logger.info(f"Loaded dataset: {len(df)} rows (removed {initial_rows - len(df)} empty rows)")
            logger.info(f"Text column: '{text_column}'")
            logger.info(f"Sample texts: {df[text_column].head(3).tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def generate_images_from_dataset(self, dataset_df: pd.DataFrame, text_column: str, 
                                   max_samples: int = None) -> List[Dict]:
        """Generate synthetic images for each text in the dataset"""
        results = []
        
        # Limit samples if specified
        if max_samples and max_samples < len(dataset_df):
            dataset_df = dataset_df.head(max_samples)
            logger.info(f"Processing first {max_samples} samples")
        
        # Available styles
        styles = ["lined_paper", "old_paper", "birch", "parchment"]
        
        for idx, row in dataset_df.iterrows():
            try:
                text = str(row[text_column]).strip()
                
                if not text:
                    logger.warning(f"Skipping empty text at row {idx}")
                    continue
                
                # Choose random style
                style = styles[idx % len(styles)]  # Cycle through styles
                
                # Generate base filename
                base_filename = f"text_image_{idx:06d}"
                image_filename = f"{base_filename}.png"
                image_path = os.path.join(self.image_dir, image_filename)
                
                logger.info(f"Generating image {idx + 1}/{len(dataset_df)}: {base_filename}")
                
                # Generate base image
                img = render_enhanced_sanskrit(
                    text=text,
                    font_path=os.path.join(self.params['font_dir'], self.params['font']),
                    output_path=None,  # Don't save intermediate
                    width=self.params['width'],
                    height=self.params['height'],
                    font_size=14,  # Fixed font size for consistency
                    style=style,
                    ink_color=self.ink_colors[style],
                    params=self.params
                )
                
                if img is None:
                    logger.error(f"Failed to generate image for row {idx}")
                    continue
                
                # Apply transformations if enabled
                if self.params.get('apply_transforms', True):
                    transformed_images = apply_enhanced_postprocessing(
                        img, None, base_filename, self.params
                    )
                    # Use the last transformed image (combined effects)
                    final_img = transformed_images[-1] if len(transformed_images) > 1 else img
                else:
                    final_img = img
                
                # Save final image
                final_img.save(image_path)
                
                # Store result information
                result = {
                    'row_index': idx,
                    'image_path': os.path.relpath(image_path, self.output_dir),
                    'text': text,
                    'style': style,
                    'image_filename': image_filename
                }
                
                # Add any additional columns from the original dataset
                for col in dataset_df.columns:
                    if col != text_column:
                        result[col] = row[col]
                
                results.append(result)
                
                logger.info(f"Successfully generated: {image_filename}")
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        logger.info(f"Generated {len(results)} images successfully")
        return results
    
    def save_results_csv(self, results: List[Dict], additional_info: Dict = None):
        """Save results to CSV file"""
        try:
            if not results:
                logger.warning("No results to save")
                return
            
            # Create DataFrame from results
            df = pd.DataFrame(results)
            
            # Reorder columns to put important ones first
            important_cols = ['image_path', 'text', 'style', 'image_filename']
            other_cols = [col for col in df.columns if col not in important_cols]
            df = df[important_cols + other_cols]
            
            # Save to CSV
            df.to_csv(self.csv_path, index=False, encoding='utf-8')
            
            logger.info(f"Results saved to: {self.csv_path}")
            logger.info(f"Total rows: {len(df)}")
            
            # Save additional metadata if provided
            if additional_info:
                metadata_path = os.path.join(self.output_dir, "metadata.txt")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    f.write("Dataset Processing Metadata\n")
                    f.write("=" * 30 + "\n")
                    for key, value in additional_info.items():
                        f.write(f"{key}: {value}\n")
                
                logger.info(f"Metadata saved to: {metadata_path}")
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def process_huggingface_dataset(self, dataset_identifier: str, text_column: str, 
                                   max_samples: int = None, config_name: str = None,
                                   split: str = None) -> bool:
        """Complete workflow to process a Hugging Face dataset"""
        try:
            logger.info("Starting Hugging Face dataset processing...")
            
            # Try to parse dataset identifier
            df = None
            
            # Method 1: Try using datasets library with dataset name
            if not dataset_identifier.startswith("http"):
                # It's a dataset name like "imdb" or "user/dataset-name"
                df = self.load_huggingface_dataset(
                    dataset_identifier, 
                    config_name=config_name,
                    split=split
                )
            
            # Method 2: Fallback to URL download if datasets library fails
            if df is None and dataset_identifier.startswith("http"):
                logger.info("Trying URL-based download...")
                dataset_file = "downloaded_dataset.csv"
                if self.download_dataset_from_url(dataset_identifier, dataset_file):
                    dataset_path = os.path.join(self.output_dir, dataset_file)
                    df = self.load_dataset(dataset_path, text_column)
            
            if df is None:
                logger.error("Failed to load dataset using any method")
                return False
            
            # Validate text column
            if text_column not in df.columns:
                logger.error(f"Column '{text_column}' not found. Available columns: {list(df.columns)}")
                return False
            
            # Step 3: Generate images
            results = self.generate_images_from_dataset(df, text_column, max_samples)
            if not results:
                logger.error("No images were generated")
                return False
            
            # Step 4: Save results
            additional_info = {
                "dataset_identifier": dataset_identifier,
                "config_name": config_name,
                "split": split,
                "text_column": text_column,
                "original_rows": len(df),
                "processed_rows": len(results),
                "max_samples": max_samples or "all",
                "output_directory": self.output_dir,
                "image_directory": self.image_dir
            }
            
            self.save_results_csv(results, additional_info)
            
            logger.info("Hugging Face dataset processing completed successfully!")
            logger.info(f"Generated {len(results)} synthetic text images")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"CSV file: {self.csv_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in dataset processing workflow: {e}")
            return False
    
    def process_local_csv(self, csv_path: str, text_column: str, 
                         max_samples: int = None) -> bool:
        """Process a local CSV file"""
        try:
            logger.info(f"Processing local CSV file: {csv_path}")
            
            # Load dataset
            df = self.load_dataset(csv_path, text_column)
            if df is None:
                return False
            
            # Generate images
            results = self.generate_images_from_dataset(df, text_column, max_samples)
            if not results:
                logger.error("No images were generated")
                return False
            
            # Save results
            additional_info = {
                "source_file": csv_path,
                "text_column": text_column,
                "original_rows": len(df),
                "processed_rows": len(results),
                "max_samples": max_samples or "all",
                "output_directory": self.output_dir,
                "image_directory": self.image_dir
            }
            
            self.save_results_csv(results, additional_info)
            
            logger.info("Local CSV processing completed successfully!")
            logger.info(f"Generated {len(results)} synthetic text images")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing local CSV: {e}")
            return False
