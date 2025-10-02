from typing import Optional
import pandas as pd
import datasets


def load_huggingface_dataset(dataset_name: str,
                            config_name: str = None,
                            split: str = None, 
                            streaming: bool = False) -> Optional[pd.DataFrame]:
        """Load dataset using the datasets library"""
        try:            
            # Load dataset using datasets library
            dataset = datasets.load_dataset(
                dataset_name,
                config_name,
                split=split,
                streaming=streaming
            )
            
            # Convert to pandas DataFrame if not streaming
            if streaming:
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
                else:
                    # Single dataset
                    df = dataset.to_pandas()                
                return df
                
        except Exception as e:
            return None
        
