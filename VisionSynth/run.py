import os
import sys
import argparse
import errno

sys.path.append(os.path.dirname(os.path.abspath(__file__), ".."))


def parse_aguments():


    parser = argparse.ArgumentParser(
        description="Generate synthetic image with text for OCR and VLM."
    )

    parser.add_argument(
        "--output_dir", type=str, nargs="?", default="output/", help="Path to the output directory for images files and csv."
    )

    parser.add_argument(
        "-ic", "--input_csv", type=str, nargs="?", default="", help="Path to the input CSV file."
    )
    
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension of the output image files.",
        default=".jpg",
    )
    
    parser.add_argument(
        "-bl",
        "--blur_level",
        type=int,
        nargs="?",
        help="Define the level of blur to apply to the image. Default is 0.",
        default=0
    )
    
    parser.add_argument(
        "-rbl",
        "--random_blur_level",
        action="store_true",
        help="Apply random blur level to the image. The blur radius will be randomized between 0 and -bl.",
        default=False
    )
    
    parser.add_argument(
        "-b",
        "--background",
        type=str,
        nargs="?",
        help="What type of background to use",
        choices=["plain", "gaussian", "image"],
        default="plain"
    )
    
    parser.add_argument(
        "-ln",
        "--lined_paper",
        action="store_true",
        help="Use lined paper as background. applies lines to the background -b.",
        default=False
    )
    
    parser.add_argument(
        "-lns",
        "--line_spacing",
        type=int,
        nargs="?",
        help="Spacing between lines if -ln is set. Default is 15 pixels.",
        default=15
    )
    
    parser.add_argument(
        "-rlns",
        "--random_line_spacing",
        action="store_true",
        help="if -ln is set, randomize line spacing between min line spacing (15) and max line spacing (default 25).",
        default=False
    )
    
    parser.add_argument(
        "-min_lns",
        "--min_line_spacing",
        type=int,
        nargs="?",
        help="Minimum line spacing if -rlns is set. Default is 15 pixels.",
        default=15
    )
    
    parser.add_argument(
        "-max_lns",
        "--max_line_spacing",
        type=int,
        nargs="?",
        help="Maximum line spacing if -rlns is set. Default is 25 pixels.",
        default=25
    )

    parser.add_argument(
        "-lni",
        "--line_intensity",
        type=int,
        nargs="?",
        help="Intensity of the lines if -ln is set. Default is 100 (0-255).",
        default=100
    )
    
    parser.add_argument(
        "-lnw",
        "--line_width",
        type=int,
        nargs="?",
        help="Width of the lines if -ln is set. Default is 1 pixel.",
        default=1
    )
    
    parser.add_argument(
        "-rlnw",
        "--random_line_width",
        action="store_true",
        help="if -ln is set, randomize line width between min line width (1) and max line width (default 2).",
        default=False
    )
    
    parser.add_argument(
        "-max_lnw",
        "--max_line_width",
        type=int,
        nargs="?",
        help="Maximum line width if -rlnw is set. Default is 2 pixels.",
        default=2
    )

    parser.add_argument(
        "-old",
        "--old_paper",
        action="store_true",
        help="Use old paper texture as background.",
        default=False
    )
    
    parser.add_argument(
        "-edw",
        "--edge_width",
        type=float,
        nargs="?",
        help="Width of the edges if -old is set. Default is 0.1 (10 percent for all edges).",
        default=0.1
    )
    
    parser.add_argument(
        "-ai",
        "--aging_intensity",
        type=int,
        nargs="?",
        help="Intensity of the aging effect. Default is 15.",
        default=15
    )
    
    parser.add_argument(
        "-bh", 
        "--brich_paper",
        action="store_true",
        help="Use brownish paper texture as background.",
        default=False
    )
    
    parser.add_argument(
        "-bht",
        "--brich_texture",
        type=float,
        nargs="?",
        help="Multiplier for the spot count. (default: 1.0)",
        default=1.0
    )
    
    parser.add_argument(
        "-bhs",
        "--brich_spots",
        type=int,
        nargs="?",
        help="Number of spots to apply. (default: 150) overrides -bht if both are set.",
        default=150
    )
    
    parser.add_argument(
        "-bhr",
        "--brich_spot_radius",
        type=tuple,
        nargs="?",
        help="Radius range of the spots to apply. (default: (10,25))",
        default=(10, 25)
    )
    
    parser.add_argument(
        "-bhi",
        "--brich_sport_intensity",
        type=int,
        nargs="?",
        help="Intensity of the spots to apply. (default: 10)",
        default=10
    )
    
    parser.add_argument(
        "-bhsi",
        "--brich_spot_sign",
        type=int,
        nargs="?",
        help="Sign of the spots to apply. 1 for lighter spots, -1 for darker spots 0 for random. (default: 1)" ,
        default=-1
    )
    
    parser.add_argument(
        "-bhir",
        "--brich_irregularity",
        type=float,
        nargs="?",
        help="Irregularity of the spots to apply. (default: 0.35)",
        default=0.35
    )
    
    parser.add_argument(
        "-bhbl",
        "--brich_blur",
        type=float,
        nargs="?",
        help="gaussian blur of the spots to apply. (default: 1.2)",
        default=1.2
    )
    
    parser.add_argument(
        "-bhc",
        "--brich_capspots",
        type=int,
        nargs="?",
        help="Cap the number of spots to apply. (default: 2000)",
        default=2000
    )
    
    parser.add_argument(
        "-ph", 
        "--parchment_paper",
        action="store_true",
        help="Use parchment paper texture as background.",
        default=False
    )
    
    parser.add_argument(
        "-pht",
        "--parchment_texture",
        type=float,
        nargs="?",
        help="Multiplier for the spot count. (default: 1.0)",
        default=1.0
    )
    
    parser.add_argument(
        "-phs",
        "--parchment_spots",
        type=int,
        nargs="?",
        help="Number of spots to apply. (default: 400) overrides -pht if both are set.",
        default=400
    )
    
    parser.add_argument(
        "-phr",
        "--parchment_spot_radius",
        type=tuple,
        nargs="?",
        help="Radius range of the spots to apply. (default: (10,25))",
        default=(3, 12)
    )
    
    parser.add_argument(
        "-phi",
        "--parchment_spot_intensity",
        type=int,
        nargs="?",
        help="Intensity of the spots to apply. (default: 7)",
        default=7
    )
    
    parser.add_argument(
        "-phsi",
        "--parchment_spot_sign",
        type=int,
        nargs="?",
        help="Sign of the spots to apply. 1 for lighter spots, -1 for darker spots 0 for random. (default: -1)" ,
        default=-1
    )
    
    parser.add_argument(
        "-phir",
        "--parchment_irregularity",
        type=float,
        nargs="?",
        help="Irregularity of the spots to apply. (default: 0.35)",
        default=0.35
    )
    
    parser.add_argument(
        "-phbl",
        "--parchment_blur",
        type=float,
        nargs="?",
        help="gaussian blur of the spots to apply. (default: 1.2)",
        default=1.2
    )
    
    parser.add_argument(
        "-phc",
        "--parchment_capspots",
        type=int,
        nargs="?",
        help="Cap the number of spots to apply. (default: 2000)",
        default=2000
    )
    
    parser.add_argument(
        "-eff",
        "--effect_fiber",
        action="store_true",
        help="Apply paper fiber effect to the background.",
        default=False
    )
    
    parser.add_argument(
        "-effd",
        "--effect_fiber_density",
        type=float,
        nargs="?",
        help="Density of the paper fiber effect. Default is 0.2.",
        default=0.2
    )
    
    parser.add_argument(
        "-efc",
        "--effect_fold_creases",
        action="store_true",
        help="Apply fold creases effect to the background.",
        default=False
    )
    
    parser.add_argument(
        "-efci",
        "--effect_fold_creases_intensity",
        type=int,
        nargs="?",
        help="Intensity of the fold creases effect. Default is 20.",
        default=20
    )

    parser.add_argument(
        "-efink",
        "--effect_ink_bleed",
        action="store_true",
        help="Apply ink bleed effect to the background.",
        default=False
    )
    
    parser.add_argument(
        "-efinki",
        "--effect_ink_bleed_intensity",
        type=float,
        nargs="?",
        help="Intensity of the ink bleed effect. Default is 0.3.",
        default=0.3
    )  
    
    parser.add_argument(
        "-efinkr",
        "--effect_ink_bleed_radius",
        type=int,
        nargs="?",
        help="Radius of the ink bleed effect. Default is 3.",
        default=3
    )
    
    parser.add_argument(
        "-efsh",
        "--effect_shadow",
        action="store_true",
        help="Apply shadow effect to the background.",
        default=False
    )
    
    parser.add_argument(
        "-efshi",
        "--effect_shadow_intensity",
        type=float,
        nargs="?",
        help="Intensity of the shadow effect. Default is 30.",
        default=0.4
    )
    
    parser.add_argument(
        "-efshr",
        "--effect_shadow_radius",
        type=int,
        nargs="?",
        help="Radius of the shadow effect. Default is 5.",
        default=5
    )
    
    parser.add_argument(
        "-efsa",
        "--effect_shadows_angle",
        type=float,
        nargs="?",
        help="Angle of the shadow effect. Default is 45.",
        default=45
    )
    
    parser.add_argument(
        "-efsa",
        "--effect_shadows_angle",
        type=float,
        nargs="?",
        help="Angle of the shadow effect. Default is 45.",
        default=45
    )

    parser.add_argument(
        "-efst",
        "--effect_strain",
        action="store_true",
        help="Apply strain effect to the background.",
        default=False
    )
    
    
    parser.add_argument(
        "-name",
        "--dataset_name",
        type=str,
        nargs="?",
        help="Name of the huggingface dataset to use. (default: '')",
        default=None
    )
    
    parser.add_argument(
        "-conf",
        "--config_name",
        type=str,
        nargs="?",
        help="Config name of the huggingface dataset to use. (default: None)",
        default=None
    )
    
    parser.add_argument(
        "-sp",
        "--split",
        type=str,
        nargs="?",
        help="Split of the huggingface dataset to use. (default: None)",
        default=None
    )
    
    parser.add_argument(
        "-s",
        "--streaming",
        action="store_true",
        help="Use streaming mode for huggingface dataset. (default: False)",
        default=False
    )
    
    parser.add_argument(
        "--output_csv",
        type=str,
        nargs="?",
        help="name to the output CSV file. (default: None)",
        default="output.csv"
    )


    parser.add_argument(
        "-rs",
        "--row_start",
        type=int,
        nargs="?",
        help="Starting row for processing. (default: 0)",
        default=0
    )
    
    parser.add_argument(
        "-r",
        "--rows",
        type=str,
        nargs="?",
        help="Path to the checkpoint to resume from. (default: None)",
        default=None
    )
    
    parser.add_argument(
        "-ac"
        "--accumulate",
        action="store_true",
        help="Accumulate results to an existing CSV file. (default: False)",
        default=False
    )
    
    parser.add_argument(
        "--csv_file",
        type=str,
        nargs="?",
        help="Path to the CSV file to use for accumulation. (default: None)",
        default="output.csv"
    )
    
    parser.add_argument(
        "-f",
        "--font",
        type=str,
        nargs="?",
        help="Path to the font file to use. (default: None)",
        default=None
    )
    
    return parser.parse_args()


def main():
    args = parse_aguments()

    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
    