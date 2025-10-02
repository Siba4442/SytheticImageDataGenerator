import pandas as pd
from typing import Optional , List, Dict
import os
from huggingface_processor import load_huggingface_dataset



def load_dataset(file_path: str, text_column: str) -> Optional[pd.DataFrame]:
        """Load dataset from CSV file and validate text column"""
        try:
            # Try reading with different encodings
            encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'utf-16']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
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
                        
            return df
            
        except Exception as e:
            return None
        
def save_results_csv(results: List[Dict], additional_info: Dict = None, csv_path: str = "dataset.csv", output_dir: str = "output"):
        """Save results to CSV file"""
        try:
            if not results:
                return
            
            # Create DataFrame from results
            df = pd.DataFrame(results)
            
            # Reorder columns to put important ones first
            important_cols = ['image_path', 'text', 'style', 'image_filename']
            other_cols = [col for col in df.columns if col not in important_cols]
            df = df[important_cols + other_cols]
            
            # Save to CSV
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            
            # Save additional metadata if provided
            if additional_info:
                metadata_path = os.path.join(output_dir, "metadata.txt")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    f.write("Dataset Processing Metadata\n")
                    f.write("=" * 30 + "\n")
                    for key, value in additional_info.items():
                        f.write(f"{key}: {value}\n")
                
        except Exception as e:
            pass
        
def generate_images_from_dataset(self, dataset_df: pd.DataFrame, 
                                 text_column: str, 
                                 max_samples: int = None) -> List[Dict]:
        """Generate synthetic images for each text in the dataset"""
        results = []
        
        # Limit samples if specified
        if max_samples and max_samples < len(dataset_df):
            dataset_df = dataset_df.head(max_samples)
        
        # Available styles
        styles = ["lined_paper", "old_paper", "birch", "parchment"]
        
        for idx, row in dataset_df.iterrows():
            try:
                text = str(row[text_column]).strip()
                
                if not text:
                    continue
                
                # Choose random style
                style = styles[idx % len(styles)]  # Cycle through styles
                
                # Generate base filename
                base_filename = f"text_image_{idx:06d}"
                image_filename = f"{base_filename}.png"
                image_path = os.path.join(self.image_dir, image_filename)
                
                
                # Generate base image
                img = render_enhanced_sanskrit(
                    text=text,
                    font_path=os.path.join(self.params['fonts'], self.params['font']),
                    output_path=None,  # Don't save intermediate
                    width=self.params['width'],
                    height=self.params['height'],
                    font_size=14,  # Fixed font size for consistency
                    style=style,
                    ink_color=self.ink_colors[style],
                    params=self.params
                )
                
                if img is None:
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
                
                
            except Exception as e:
                continue
        
        return results
    
    
def process_huggingface_dataset(dataset_identifier: str, 
                                text_column: str, 
                                max_samples: int = None, 
                                config_name: str = None,
                                split: str = None,
                                output_dir: str = None,
                                image_dir: str = None) -> bool:
        """Complete workflow to process a Hugging Face dataset"""
        try:
            # Try to parse dataset identifier
            df = None
            
            # Method 1: Try using datasets library with dataset name
            if not dataset_identifier.startswith("http"):
                # It's a dataset name like "imdb" or "user/dataset-name"
                df = load_huggingface_dataset(
                    dataset_identifier, 
                    config_name=config_name,
                    split=split
                )
            
            if df is None:
                return False
            
            # Validate text column
            if text_column not in df.columns:
                return False
            
            # Step 3: Generate images
            results = generate_images_from_dataset(df, text_column, max_samples)
            if not results:
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
                "output_directory": output_dir,
                "image_directory": image_dir
            }
            
            save_results_csv(results, additional_info)
                        
            return True
            
        except Exception as e:
            return False
        
def process_local_csv(self, csv_path: str, 
                      text_column: str, 
                      max_samples: int = None,
                      output_dir: str = None,
                      image_dir: str = None) -> bool:
    
    """Process a local CSV file"""
    try:
        
        # Load dataset
        df = self.load_dataset(csv_path, text_column)
        if df is None:
            return False
        
        # Generate images
        results = generate_images_from_dataset(df, text_column, max_samples)
        if not results:
            return False
        
        # Save results
        additional_info = {
            "source_file": csv_path,
            "text_column": text_column,
            "original_rows": len(df),
            "processed_rows": len(results),
            "max_samples": max_samples or "all",
            "output_directory": output_dir,
            "image_directory": image_dir
        }
        
        self.save_results_csv(results, additional_info)
        
        
        return True
        
    except Exception as e:
        return False