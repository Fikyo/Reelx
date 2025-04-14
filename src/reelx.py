from reelx.config.configuration import Config
from reelx.processors.video_processor import VideoProcessor
from reelx.utils.printer import Printer
from reelx.processors.input_processor import InputProcessor

import os
import sys
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Transform your horizontal video to vertical video')
    parser.add_argument('input_video', help='Path to input video file or Live UDP URL')
    parser.add_argument('--model_type', type=str, default=Config.MODEL_PARAMS["MODEL_TYPE_YOLO"], help='Select the Model for person detection YOLO')
    parser.add_argument('--model_verbose', type=str, default=Config.MODEL_PARAMS["MODEL_VERBOSE"], help='Enable verbose mode for the YOLO model')
    parser.add_argument('--mode', type=str, default="vertical", choices=["vertical", "square"], 
                       help='Video output mode: vertical (9:16) or square (1:1)')
    parser.add_argument('--output', type=str, default=None,help='Path to output video file')
    parser.add_argument('--confidence', type=float, default=Config.MODEL_PARAMS["OBJECT_CONFIDENCE_THRESHOLD"], help='Confidence threshold for detection')
    parser.add_argument('--classid', type=float, default=None, help='Object to be focused for ROI')
    parser.add_argument('--smoothing', type=float, default=Config.TRACKING_PARAMS["SMOOTHING_FACTOR"], help='Smoothing factor for bounding box')
    parser.add_argument('--preview_vertical_video', type=bool, default=Config.OUTPUT_VIDEO_PARAMS["PREVIEW_VERTICAL_VIDEO"], help='Show vertical video frames output during processing')
    parser.add_argument('--preview_debug_player', type=bool, default=Config.OUTPUT_VIDEO_PARAMS["PREVIEW_DEBUG_PLAYER"], help='Show Orginal video frames with BB  during processing')
    parser.add_argument('--enable_tiled_frame', type=str, default=Config.OUTPUT_VIDEO_PARAMS["PERSON_TILED"], help='Enable tiled frame output for two person detection')
    parser.add_argument('--person_model', type=str, default=Config.MODEL_PARAMS["PERSON_MODEL_PATH"], help='Path to YOLO person detection model')
    parser.add_argument('--process_noloss_frame', type=str, default=Config.OUTPUT_VIDEO_PARAMS["NO_LOSS_TILED"], help='Enable no loss frame processing')
    parser.add_argument('--noloss_tiled_position', type=str, default='BOTTOM', help='Position of cropped tiled frame (TOP/BOTTOM/MIDDLE)')    
    return parser.parse_args()

def set_config_from_args(args):
    """Set configuration parameters based on command line arguments"""
    
    Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"] = args.model_type
    
    Config.MODEL_PARAMS["MODEL_VERBOSE"] = args.model_verbose
    Config.MODEL_PARAMS["OBJECT_CONFIDENCE_THRESHOLD"] = args.confidence
    Config.MODEL_PARAMS["PERSON_MODEL_PATH"] = args.person_model

    
    Config.TRACKING_PARAMS["SMOOTHING_FACTOR"] = args.smoothing
    
    # Set video mode configuration
    if args.mode == "square":
        Config.OUTPUT_VIDEO_PARAMS["SQUARE"] = True
        Config.OUTPUT_VIDEO_PARAMS["VERTICAL"] = False
    else:  # vertical mode
        Config.OUTPUT_VIDEO_PARAMS["VERTICAL"] = True
        Config.OUTPUT_VIDEO_PARAMS["SQUARE"] = False
    Config.OUTPUT_VIDEO_PARAMS["PREVIEW_VERTICAL_VIDEO"] = args.preview_vertical_video
    Config.OUTPUT_VIDEO_PARAMS["PREVIEW_DEBUG_PLAYER"] = args.preview_debug_player
    Config.OUTPUT_VIDEO_PARAMS["PERSON_TILED"] = args.enable_tiled_frame
    Config.OUTPUT_VIDEO_PARAMS["NO_LOSS_TILED"] = args.process_noloss_frame
    Config.OUTPUT_VIDEO_PARAMS["OUTPUT_FILE"] = args.output
    Config.OUTPUT_VIDEO_PARAMS["CLASS_ID"] = args.classid
    Config.INPUT_PARAMS['CURRENT_INPUT_TYPE'] = InputProcessor(args.input_video).get_current_input_type()
    Config.CROP_PLACEMENT_PARAMS["CURRENT_PLACEMENT"] = args.noloss_tiled_position


def main(): 
    print(sys.path)
    src_path = os.path.abspath("../src") 
    args = parse_arguments()
    Printer.print_arguments(args)
    set_config_from_args(args)
    processor = VideoProcessor(
        source_path=args.input_video,
        output_path=args.output,
        mode=args.mode
    )
    processor.process_video()

if __name__ == "__main__":
    main()
