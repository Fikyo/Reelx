
import cv2

class Printer:
    def print_arguments(args):
        print("Command Line Arguments:")
        for arg, value in vars(args).items():
            print(f"{arg:20}: {value}")
        print()

    def print_frame_properties(cap):
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

    def print_info(message):
        print(f"INFO: {message}")    

