import cv2

class Printer:
    @staticmethod
    def print_arguments(args):
        """Print command line arguments in a formatted way"""
        print("Command Line Arguments:")
        for arg, value in vars(args).items():
            print(f"{arg:20}: {value}")
        print()

    @staticmethod
    def print_frame_properties(cap):
        """Print video frame properties in a formatted way"""
        print("Video Properties:")
        print(f"Frame Width         : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
        print(f"Frame Height        : {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS                 : {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Frame Count         : {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        print(f"Format              : {int(cap.get(cv2.CAP_PROP_FORMAT))}")
        print(f"Mode                : {int(cap.get(cv2.CAP_PROP_MODE))}")
        print(f"Brightness          : {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
        print(f"Contrast            : {cap.get(cv2.CAP_PROP_CONTRAST)}")
        print(f"Saturation          : {cap.get(cv2.CAP_PROP_SATURATION)}")
        print(f"Hue                 : {cap.get(cv2.CAP_PROP_HUE)}")
        print(f"Backend Name        : {cap.getBackendName()}")  
        print()

    @staticmethod
    def print_info(message):
        """Print an info message with a prefix"""
        print(f"INFO: {message}")
