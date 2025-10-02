"""
Enhanced modular main.py with HuggingFace dataset processing capabilities
"""

import os
import argparse
import logging
from typing import Dict

from synthetic_text_generator import (
    ENHANCED_DEFAULT_PARAMS,
    generate_enhanced_sanskrit_samples,
    generate_comprehensive_dataset, 
    generate_ultra_realistic_samples,
    HuggingFaceDatasetProcessor
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Enhanced main function with HuggingFace dataset processing"""
    sanskrit_text = """
    କବି ସମ୍ରାଟ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ (୧୬୭୦ - ୧୭୪୦, ଅନ୍ୟମତେ ୧୭୨୦) ସପ୍ତଦଶ ଶତାବ୍ଦୀର ପ୍ରମୁଖ ଓଡ଼ିଶୀ ସଙ୍ଗୀତକାର କବି ଥିଲେ । ସେ ସପ୍ତଦଶ ଶତାବ୍ଦୀର ଶେଷ ପର୍ଯ୍ୟାୟରେ ପୁରାତନ ରାଜଶାସିତ ଘୁମୁସରର ରାଜବଂଶରେ କୁଲାଡ଼ଗଡଜନ୍ମଲାଭ କରିଥିଲେ। ରୀତିଯୁଗର ତାଙ୍କର ପାଣ୍ଡିତ୍ୟପୂର୍ଣ୍ଣ ସାହିত୍ୟ କୃତି ପାଇଁ ତାଙ୍କୁ କବି ସମ୍ରାଟ ଉପାଧିରେ ଭୂଷିତ କରାଯାଇଅଛି । ଓଡ଼ିଆ ସାହିତ୍ୟରେ ଥିବା ଅସଂଖ୍ୟ କବିଙ୍କ ମଧ୍ୟରେ ସେ ଅସାଧାରଣ ପ୍ରସିଦ୍ଧି ଲାଭ କରିଅଛନ୍ତି । ଏକ ରାଜ ପରିବାରରେ ଜନ୍ମିତ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ ରାଜପଦଠାରୁ ଦୂରରେ ରହି ଓଡ଼ିଆ ସାହିତ୍ୟରେ ମନୋନିବେଶ କରିଥିଲେ ।
    """

    parser = argparse.ArgumentParser(description='Enhanced modular Sanskrit text generation with HuggingFace dataset processing')

    # Mode selection
    parser.add_argument('--mode', choices=['single', 'comprehensive', 'ultra-realistic', 'huggingface'], 
                       default='single', help='Generation mode')

    # Basic options
    basic = parser.add_argument_group('Basic Options')
    basic.add_argument('--output-dir', type=str, default='data/enhanced_synthetic/images',
                      help='Output directory for generated images')
    basic.add_argument('--width', type=int, default=ENHANCED_DEFAULT_PARAMS['width'],
                      help='Width of output images')
    basic.add_argument('--height', type=int, default=ENHANCED_DEFAULT_PARAMS['height'],
                      help='Height of output images')
    basic.add_argument('--base-images', type=int, default=ENHANCED_DEFAULT_PARAMS['base_images'],
                      help='Total number of base images to generate')

    # HuggingFace dataset options
    hf_group = parser.add_argument_group('HuggingFace Dataset Options')
    hf_group.add_argument('--dataset-url', type=str, 
                         help='URL of the HuggingFace dataset (CSV format)')
    hf_group.add_argument('--text-column', type=str, default='text',
                         help='Name of the column containing text data')
    hf_group.add_argument('--max-samples', type=int,
                         help='Maximum number of samples to process from dataset')
    hf_group.add_argument('--csv-file', type=str,
                         help='Local CSV file to process instead of downloading')

    # Font options
    font = parser.add_argument_group('Font Options')
    font.add_argument('--font-dir', type=str, default=ENHANCED_DEFAULT_PARAMS['font_dir'],
                     help='Directory containing font files')
    font.add_argument('--font', type=str, default=ENHANCED_DEFAULT_PARAMS['font'],
                     help='Font filename within the font directory')

    # Generation-level augmentations
    gen = parser.add_argument_group('Generation-Level Augmentations')
    gen.add_argument('--noise', type=float, default=ENHANCED_DEFAULT_PARAMS['noise'],
                    help='Background noise intensity (0.0-1.0)')
    gen.add_argument('--aging', type=float, default=ENHANCED_DEFAULT_PARAMS['aging'],
                    help='Edge aging effect (0.0-1.0)')
    gen.add_argument('--texture', type=float, default=ENHANCED_DEFAULT_PARAMS['texture'],
                    help='Texture variation (0.0-1.0)')
    gen.add_argument('--stains', type=float, default=ENHANCED_DEFAULT_PARAMS['stains'],
                    help='Number of stains (0.0-1.0)')
    gen.add_argument('--stain-intensity', type=float, default=ENHANCED_DEFAULT_PARAMS['stain_intensity'],
                    help='Intensity of stain effects (0.0-1.0)')
    gen.add_argument('--fiber-density', type=float, default=ENHANCED_DEFAULT_PARAMS['fiber_density'],
                    help='Paper fiber texture density (0.0-1.0)')
    gen.add_argument('--image-dir', type=str, default=ENHANCED_DEFAULT_PARAMS['image_dir'],
                    help='Directory to randomly sample background images from')

    # Word-level options
    word = parser.add_argument_group('Word-Level Options')
    word.add_argument('--word-position', type=float, default=ENHANCED_DEFAULT_PARAMS['word_position'],
                     help='Random word position variation (0.0-1.0)')
    word.add_argument('--ink-color', type=float, default=ENHANCED_DEFAULT_PARAMS['ink_color'],
                     help='Ink color variation (0.0-1.0)')
    word.add_argument('--line-spacing', type=float, default=ENHANCED_DEFAULT_PARAMS['line_spacing'],
                     help='Random line spacing (0.0-1.0)')
    word.add_argument('--baseline', type=float, default=ENHANCED_DEFAULT_PARAMS['baseline'],
                     help='Baseline wobble effect (0.0-1.0)')
    word.add_argument('--word-angle', type=float, default=ENHANCED_DEFAULT_PARAMS['word_angle'],
                     help='Random word angle (0.0-1.0)')

    # Post-processing options
    post = parser.add_argument_group('Post-Processing Augmentations')
    post.add_argument('--no-transforms', dest='apply_transforms', action='store_false',
                     help='Disable post-processing transforms')
    post.add_argument('--all-transforms', action='store_true',
                     help='Apply all transforms instead of random subset')
    post.add_argument('--rotation-max', type=float, default=ENHANCED_DEFAULT_PARAMS['rotation_max'],
                     help='Maximum rotation angle in degrees')
    post.add_argument('--brightness-var', type=float, default=ENHANCED_DEFAULT_PARAMS['brightness_var'],
                     help='Brightness variation factor (0.0-1.0)')
    post.add_argument('--contrast-var', type=float, default=ENHANCED_DEFAULT_PARAMS['contrast_var'],
                     help='Contrast variation factor (0.0-1.0)')
    post.add_argument('--noise-min', type=float, default=ENHANCED_DEFAULT_PARAMS['noise_min'],
                     help='Minimum noise intensity for transforms')
    post.add_argument('--noise-max', type=float, default=ENHANCED_DEFAULT_PARAMS['noise_max'],
                     help='Maximum noise intensity for transforms')
    post.add_argument('--blur-min', type=float, default=ENHANCED_DEFAULT_PARAMS['blur_min'],
                     help='Minimum blur radius')
    post.add_argument('--blur-max', type=float, default=ENHANCED_DEFAULT_PARAMS['blur_max'],
                     help='Maximum blur radius')

    # Advanced effects options
    advanced = parser.add_argument_group('Advanced Effects')
    advanced.add_argument('--no-advanced-effects', dest='enable_advanced_effects', action='store_false',
                         help='Disable advanced effects')
    advanced.add_argument('--fold-intensity', type=float, default=ENHANCED_DEFAULT_PARAMS['fold_intensity'],
                         help='Intensity of fold/crease effects (0.0-1.0)')
    advanced.add_argument('--bleed-intensity', type=float, default=ENHANCED_DEFAULT_PARAMS['bleed_intensity'],
                         help='Intensity of ink bleeding effects (0.0-1.0)')
    advanced.add_argument('--bleed-radius', type=int, default=ENHANCED_DEFAULT_PARAMS['bleed_radius'],
                         help='Radius of ink bleeding effect')
    advanced.add_argument('--corner-displacement', type=int, default=ENHANCED_DEFAULT_PARAMS['corner_displacement'],
                         help='Maximum corner displacement for perspective distortion')
    advanced.add_argument('--shadow-intensity', type=float, default=ENHANCED_DEFAULT_PARAMS['shadow_intensity'],
                         help='Intensity of shadow effects (0.0-1.0)')
    advanced.add_argument('--shadow-angle', type=float, default=ENHANCED_DEFAULT_PARAMS['shadow_angle'],
                         help='Angle of shadow effects in degrees')
    advanced.add_argument('--lens-distortion-strength', type=float, default=ENHANCED_DEFAULT_PARAMS['lens_distortion_strength'],
                         help='Strength of lens distortion effects (0.0-1.0)')
    advanced.add_argument('--no-scanner-artifacts', dest='scanner_artifacts', action='store_false',
                         help='Disable scanner artifact simulation')
    advanced.add_argument('--compression-quality', type=int, default=ENHANCED_DEFAULT_PARAMS['compression_quality'],
                         help='JPEG compression quality (1-100)')

    # Probability controls
    prob = parser.add_argument_group('Effect Probabilities')
    prob.add_argument('--advanced-effect-probability', type=float, default=ENHANCED_DEFAULT_PARAMS['advanced_effect_probability'],
                     help='Probability of applying advanced effects (0.0-1.0)')
    prob.add_argument('--fold-probability', type=float, default=ENHANCED_DEFAULT_PARAMS['fold_probability'],
                     help='Probability of applying fold effects (0.0-1.0)')
    prob.add_argument('--perspective-probability', type=float, default=ENHANCED_DEFAULT_PARAMS['perspective_probability'],
                     help='Probability of applying perspective distortion (0.0-1.0)')
    prob.add_argument('--shadow-probability', type=float, default=ENHANCED_DEFAULT_PARAMS['shadow_probability'],
                     help='Probability of applying shadow effects (0.0-1.0)')

    # Performance options
    perf = parser.add_argument_group('Performance Options')
    perf.add_argument('--use-multiprocessing', action='store_true',
                     help='Enable multiprocessing for batch generation')
    perf.add_argument('--num-processes', type=int, default=ENHANCED_DEFAULT_PARAMS['num_processes'],
                     help='Number of processes for multiprocessing')
    perf.add_argument('--debug-mode', action='store_true',
                     help='Enable debug mode with verbose logging')

    # Style focus for ultra-realistic mode
    parser.add_argument('--style-focus', type=str, choices=['lined_paper', 'old_paper', 'birch', 'parchment'],
                       help='Focus on specific style for ultra-realistic generation')

    # Set defaults
    parser.set_defaults(
        apply_transforms=ENHANCED_DEFAULT_PARAMS['apply_transforms'],
        all_transforms=ENHANCED_DEFAULT_PARAMS['all_transforms'],
        enable_advanced_effects=ENHANCED_DEFAULT_PARAMS['enable_advanced_effects'],
        scanner_artifacts=ENHANCED_DEFAULT_PARAMS['scanner_artifacts']
    )

    # Handle Jupyter/Colab environment
    def is_jupyter():
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    if is_jupyter():
        # In Jupyter/Colab, parse empty args to use defaults
        args = parser.parse_args([])
        logger.info("Running in Jupyter environment - using default parameters")
    else:
        # Normal command line usage
        args = parser.parse_args()

    # Set up logging level
    if args.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)

    # Convert args to dict, excluding special arguments
    excluded_args = {'output_dir', 'mode', 'dataset_url', 'text_column', 'max_samples', 
                     'csv_file', 'style_focus'}
    params = {k.replace('-', '_'): v for k, v in vars(args).items() if k not in excluded_args}

    logger.info(f"Starting synthetic text generation in mode: {args.mode}")
    logger.info(f"Parameters: {params}")

    try:
        if args.mode == 'huggingface':
            # HuggingFace dataset processing mode
            if not args.dataset_url and not args.csv_file:
                logger.error("For HuggingFace mode, either --dataset-url or --csv-file must be provided")
                return

            # Initialize HuggingFace processor
            processor = HuggingFaceDatasetProcessor(
                output_dir=args.output_dir,
                params=params
            )

            if args.csv_file:
                # Process local CSV file
                success = processor.process_local_csv(
                    csv_path=args.csv_file,
                    text_column=args.text_column,
                    max_samples=args.max_samples
                )
            else:
                # Process HuggingFace dataset
                success = processor.process_huggingface_dataset(
                    dataset_identifier=args.dataset_url,
                    text_column=args.text_column,
                    max_samples=args.max_samples
                )

            if success:
                logger.info("HuggingFace dataset processing completed successfully!")
            else:
                logger.error("HuggingFace dataset processing failed!")

        elif args.mode == 'single':
            # Single text generation mode
            generate_enhanced_sanskrit_samples(
                text=sanskrit_text,
                font_path=os.path.join(params['font_dir'], params['font']),
                output_dir=args.output_dir,
                params=params
            )
            logger.info("Single text generation completed successfully")

        elif args.mode == 'comprehensive':
            # Comprehensive dataset generation
            generate_comprehensive_dataset(
                text=sanskrit_text,
                output_dir=args.output_dir,
                params=params
            )
            logger.info("Comprehensive dataset generation completed successfully")

        elif args.mode == 'ultra-realistic':
            # Ultra-realistic generation
            generate_ultra_realistic_samples(
                text=sanskrit_text,
                output_dir=args.output_dir,
                style_focus=args.style_focus,
                params=params
            )
            logger.info("Ultra-realistic generation completed successfully")

    except Exception as e:
        logger.error(f"Error in generation: {e}")
        raise


# Simplified functions for direct usage
def generate_colab_samples(base_images=5, width=400, height=320, enable_advanced_effects=True):
    """Simplified function for direct use in Google Colab"""
    sanskrit_text = """
    କବି ସମ୍ରାଟ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ (୧୬୭୦ - ୧୭୪୦, ଅନ୍ୟମତେ ୧୭୨୦) ସପ୍ତଦଶ ଶତାବ୍ଦୀର ପ୍ରମୁଖ ଓଡ଼ିଶୀ ସଙ୍ଗୀତକାର କବି ଥିଲେ । ସେ ସପ୍ତଦଶ ଶତାବ୍ଦୀର ଶେଷ ପର୍ଯ୍ୟାୟରେ ପୁରାତନ ରାଜଶାସିତ ଘୁମୁସରର ରାଜବଂଶରେ କୁଲାଡ଼ଗଡଜନ୍ମଲାଭ କରିଥିଲେ। ରୀତିଯୁଗର ତାଙ୍କର ପାଣ୍ଡିତ୍ୟପୂର୍ଣ୍ଣ ସାହିତ୍ୟ କୃତି ପାଇଁ ତାଙ୍କୁ କବି ସମ୍ରାଟ ଉପାଧିରେ ଭୂଷିତ କରାଯାଇଅଛି । ଓଡ଼ିଆ ସାହିତ୍ୟରେ ଥିବା ଅସଂଖ୍ୟ କବିଙ୍କ ମଧ୍ୟରେ ସେ ଅସାଧାରଣ ପ୍ରସିଦ୍ଧି ଲାଭ କରିଅଛନ୍ତି । ଏକ ରାଜ ପରିବାରରେ ଜନ୍ମିତ ଉପେନ୍ଦ୍ର ଭଞ୍ଜ ରାଜପଦଠାରୁ ଦୂରରେ ରହି ଓଡ଼ିଆ ସାହିତ୍ୟରେ ମନୋନିବେଶ କରିଥିଲେ ।
    """

    # Create custom parameters
    params = ENHANCED_DEFAULT_PARAMS.copy()
    params.update({
        'base_images': base_images,
        'width': width,
        'height': height,
        'enable_advanced_effects': enable_advanced_effects,
    })

    output_dir = 'colab_output'
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating {base_images} Sanskrit samples with advanced effects: {enable_advanced_effects}")

    # Generate samples
    images = generate_enhanced_sanskrit_samples(
        text=sanskrit_text,
        font_path=os.path.join(params['font_dir'], params['font']),
        output_dir=output_dir,
        params=params
    )

    logger.info(f"Generated samples saved to: {output_dir}")
    return images


def process_huggingface_dataset_simple(dataset_url: str, text_column: str = 'text', 
                                      max_samples: int = None, output_dir: str = 'hf_output'):
    """Simplified function to process HuggingFace datasets"""
    processor = HuggingFaceDatasetProcessor(output_dir=output_dir)
    
    success = processor.process_huggingface_dataset(
        dataset_identifier=dataset_url,
        text_column=text_column,
        max_samples=max_samples
    )
    
    if success:
        logger.info(f"Successfully processed HuggingFace dataset!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"CSV file: {processor.csv_path}")
    else:
        logger.error("Failed to process HuggingFace dataset!")
    
    return success


def process_local_csv_simple(csv_path: str, text_column: str = 'text', 
                           max_samples: int = None, output_dir: str = 'csv_output'):
    """Simplified function to process local CSV files"""
    processor = HuggingFaceDatasetProcessor(output_dir=output_dir)
    
    success = processor.process_local_csv(
        csv_path=csv_path,
        text_column=text_column,
        max_samples=max_samples
    )
    
    if success:
        logger.info(f"Successfully processed CSV file!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"CSV file: {processor.csv_path}")
    else:
        logger.error("Failed to process CSV file!")
    
    return success


if __name__ == "__main__":
    main()
