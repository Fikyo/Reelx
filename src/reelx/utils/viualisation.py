import cv2
from reelx.config.configuration import Config

class Visualizer:
    debug_props = {}
    
    def __init__(self, preview_debug, preview_vertical_video):
        """
        Initialize visualizer
        Args:
            preview_debug (bool): Whether to display debug frames
            preview_vertical_video (bool): Whether to display processed frames
        """
        self.preview_debug = preview_debug
        self.preview_vertical_video = preview_vertical_video
        
    def show_frame(self, debug_frame, processed_frame, fps, frame_count, current_frame, paused):
        """
        Display frames based on preview flags and handle user input
        Args:
            debug_frame: Frame with debug visualization
            processed_frame: Processed output frame
            fps: Frames per second
            frame_count: Total number of frames
            current_frame: Current frame number
            paused: Whether playback is paused
        Returns:
            Tuple of (paused state, current frame number)
        """
        if self.preview_debug:
            cv2.imshow('Debug', debug_frame)
        if self.preview_vertical_video:
            cv2.imshow('Processed', processed_frame)
        
        if self.preview_debug or self.preview_vertical_video:
            key = cv2.waitKey(0 if paused else int(1000 / fps)) & 0xFF
            if key == 27:  # Escape key to exit
                exit(0)
            elif key == ord(" "):  # Space key to play/pause
                paused = not paused
            elif key == 2:  # Left Arrow key to rewind frame-by-frame
                current_frame = max(0, current_frame - 1)
            elif key == 3:  # Right Arrow key to forward frame-by-frame
                current_frame = min(frame_count - 1, current_frame + 1)
        return paused, current_frame

    def close(self):
        """Close all opencv windows"""
        cv2.destroyAllWindows()

    def add_title(self, frame, title):
        """
        Add centered title text to frame
        Args:
            frame: Input frame to add title to
            title: Text string to display
        Returns:
            Frame with title added
        """
        height, width = frame.shape[:2]
        
        font = Config.DISPLAY_PARAMS["FONT"]
        font_scale = Config.DISPLAY_PARAMS["FONT_SCALE"] + 1
        thickness = Config.DISPLAY_PARAMS["LINE_THICKNESS"] + 3
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        
        text_x = int((width - text_size[0]) / 2)
        text_y = int(text_size[1] + 20)  # Add padding from top
        
        cv2.putText(frame, title, (text_x, text_y), font, font_scale, 
                    Config.DISPLAY_PARAMS["COLORS"]["TITLE_TEXT"], thickness)
        
        return frame

    def draw_output_box(self, frame, dimensions):
        """
        Draw output box with increased line thickness
        Args:
            frame: Input frame to draw box on
            dimensions: Tuple of (x, y, width, height) for box dimensions
        Returns:
            Frame with box drawn
        """
        x1, y1, x2, y2 = dimensions
        thickness = Config.DISPLAY_PARAMS["LINE_THICKNESS"] + 2
        color = Config.DISPLAY_PARAMS["COLORS"]["MASTER_BOX"]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame
    
    def set_debug_props(self, key, value):
        """
        Store key-value pair in debug properties
        Args:
            key: Dictionary key
            value: Value to store
        """
        Visualizer.debug_props[key] = value
    
    def draw_debug_props(self, frame):
        """
        Draw debug properties on the bottom right of frame
        Args:
            frame: Input frame to add debug text
        Returns:
            Frame with debug properties text added
        """
        height, width = frame.shape[:2]
        
        font = Config.DISPLAY_PARAMS["FONT"]
        font_scale = Config.DISPLAY_PARAMS["FONT_SCALE"] + 0.5
        thickness = Config.DISPLAY_PARAMS["LINE_THICKNESS"] + 1
        color = Config.DISPLAY_PARAMS["COLORS"]["DEBUG_TEXT"]
        
        padding = 20
        line_height = 40  # Increased spacing between lines
        current_y = height - padding
        
        for key, value in Visualizer.debug_props.items():
            text = f"{key}:     {value}"  # Added more spaces between key and value
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = width - text_size[0] - padding
            
            cv2.putText(frame, text, (text_x, current_y), font, 
                    font_scale, color, thickness)
            current_y -= line_height
            
        return frame
