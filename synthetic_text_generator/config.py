"""
Configuration parameters for the Synthetic Text Generator
"""

# Enhanced global parameters with all advanced features
ENHANCED_DEFAULT_PARAMS = {
    # Basic options
    'width': 400,
    'height': 320,
    'base_images': 1,

    # Font options
    'font_dir': './content/static',
    'font': 'NotoSansOriya_Condensed-Regular.ttf',

    # Generation-level augmentations
    'noise': 0.7,
    'aging': 0.6,
    'texture': 0.7,
    'stains': 0.6,
    'stain_intensity': 0.5,

    # Word-level options
    'word_position': 0.6,
    'ink_color': 0.5,
    'line_spacing': 0.4,
    'baseline': 0.3,
    'word_angle': 0.0,

    # Post-processing options
    'apply_transforms': True,
    'all_transforms': False,
    'rotation_max': 5.0,
    'brightness_var': 0.2,
    'contrast_var': 0.2,
    'noise_min': 0.01,
    'noise_max': 0.05,
    'blur_min': 0.5,
    'blur_max': 1.0,

    # Advanced effect parameters
    'fold_intensity': 0.3,
    'bleed_intensity': 0.3,
    'bleed_radius': 3,
    'corner_displacement': 20,
    'morph_operation': 'mixed',
    'morph_kernel_size': 3,
    'aging_intensity': 0.5,
    'fiber_density': 0.5,
    'enable_advanced_effects': True,
    'advanced_effect_probability': 0.7,
    'shadow_angle': 45,
    'shadow_intensity': 0.4,
    'lens_distortion_strength': 0.2,
    'scanner_artifacts': True,
    'compression_quality': 85,
    'fold_probability': 0.4,
    'crease_probability': 0.3,
    'perspective_probability': 0.5,
    'shadow_probability': 0.6,

    # Performance parameters
    'use_multiprocessing': False,
    'num_processes': 4,
    'enable_caching': True,
    'debug_mode': False,
    'image_dir': ''
}
