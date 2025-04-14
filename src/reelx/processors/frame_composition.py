import cv2
import numpy as np
import time
from reelx.config.configuration import Config

class FrameComposition:
    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width

    def get_master_box(self, frame, padding_factor=0):
        """
        Calculate the master box dimensions for the frame based on output mode
        """
        frame_height, frame_width = frame.shape[:2]
        
        if Config.OUTPUT_VIDEO_PARAMS["SQUARE"]:
            # Keep vertical ROI box but adjust width for square crop
            crop_width = int(frame_height * (9/16))  # Keep vertical aspect ratio
            crop_width_with_padding = min(crop_width * (1 + 2*padding_factor), frame_width)

            x1 = max(0, (frame_width - crop_width_with_padding) // 2)
            x2 = min(frame_width, x1 + crop_width_with_padding)
            
            return (x1, 0, x2, frame_height)
        else:
            # Vertical 9:16 mode
            crop_width = int(frame_height * (9/16))
            crop_width_with_padding = min(crop_width * (1 + 2*padding_factor), frame_width)

            x1 = max(0, (frame_width - crop_width_with_padding) // 2)
            x2 = min(frame_width, x1 + crop_width_with_padding)

            if x2 > frame_width:
                x2 = frame_width
                x1 = max(0, x2 - crop_width_with_padding)
            
            return (x1, 0, x2, frame_height)

    def place_main_frame(self, frame, masterbox, position):
        """
        Place the main frame in the composition based on output mode
        """
        frame_height, frame_width = frame.shape[:2]
        master_x, master_y, master_width, master_height = masterbox
        
        # For square mode, create a square crop within the vertical frame
        if Config.OUTPUT_VIDEO_PARAMS["SQUARE"]:
            # Extract the vertical ROI
            frame = frame[:, master_x:master_width]
            frame_height, frame_width = frame.shape[:2]
            
            # Create output frame
            vertical_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            
            # Calculate square size (equal to width)
            square_size = frame_width
            
            # Calculate scale to fit width
            scale = self.target_width / frame_width
            
            # Resize frame maintaining aspect ratio
            resized_frame = cv2.resize(frame, (self.target_width, int(frame_height * scale)))
            resized_height = resized_frame.shape[0]
            
            # Calculate vertical position based on content
            if position == "top":
                y_start = 0
            elif position == "middle":
                y_start = (self.target_height - resized_height) // 2
            else:  # bottom
                y_start = max(0, self.target_height - resized_height)
                
            # Place the frame in vertical video
            y_end = min(y_start + resized_height, self.target_height)
            vertical_frame[y_start:y_end, :] = resized_frame[:y_end-y_start, :]
            
            return vertical_frame, resized_height, []
        vertical_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        scale = min(self.target_width / frame_width, self.target_height / frame_height)
        new_width = min(int(frame_width * scale), self.target_width)
        new_height = min(int(frame_height * scale), self.target_height)
        
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        if position == "TOP":
            y_start = 0
            unfilled_dimensions = [(0, new_height, new_width, master_height-new_height)]
        elif position == "MIDDLE":
            y_start = max(0, min((self.target_height - new_height) // 2, self.target_height - new_height))
            unfilled_dimensions = [(0, 0, new_width, master_height // 2 -(new_height // 2)),
                                 (0, (master_height // 2) + (new_height // 2), new_width, (master_height // 2) - (new_height // 2))]
        elif position == "BOTTOM":
            y_start = max(0, self.target_height - new_height)
            unfilled_dimensions = [(0, 0, new_width, master_height-new_height)]
        else:
            y_start = 0
            unfilled_height = min(max(0, self.target_height - new_height), self.target_height)
            unfilled_dimensions = [(new_height, 0, unfilled_height, self.target_width)]
            
        y_end = min(y_start + new_height, self.target_height)
        x_end = min(new_width, self.target_width)
        
        vertical_frame[y_start:y_end, 0:x_end] = resized_frame[0:y_end-y_start, 0:x_end]
        
        return vertical_frame, new_height, unfilled_dimensions

    def place_cropped_frame(self, vertical_frame, cropped_frame, new_height, position='bottom'):
        """
        Place the cropped frame in the vertical composition
        """
        if cropped_frame is None:
            return vertical_frame

        remaining_height = self.target_height - new_height
        crop_height, crop_width = cropped_frame.shape[:2]
        crop_scale = self.target_width / crop_width
        resized_crop = cv2.resize(cropped_frame, (self.target_width, int(crop_height * crop_scale)))
        
        if resized_crop.shape[0] > remaining_height:
            height_scale = remaining_height / resized_crop.shape[0]
            crop_aspect_ratio = resized_crop.shape[1] / resized_crop.shape[0]
            new_width = int(remaining_height * crop_aspect_ratio)
            resized_crop = cv2.resize(resized_crop, (new_width, remaining_height))
            
            x_offset = (self.target_width - new_width) // 2
            crop_start = 0 if position == "top" else new_height
            crop_end = crop_start + remaining_height
            
            vertical_frame = self.fill_side_graphics(vertical_frame, cropped_frame, crop_start, crop_end, x_offset, new_width)
            vertical_frame[crop_start:crop_end, x_offset:x_offset+new_width] = resized_crop
        else:
            crop_start = 0 if position == "top" else new_height
            crop_end = crop_start + resized_crop.shape[0]
            vertical_frame[crop_start:crop_end, :] = 0
            vertical_frame[crop_start:crop_end, 0:self.target_width] = resized_crop
            
        return vertical_frame

    def fill_side_graphics(self, vertical_frame, cropped_frame, crop_start, crop_end, x_offset, new_width):
        """
        Fill the side areas with graphics based on the dominant color
        """
        pixels = cropped_frame.reshape(-1, 3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_color = centers[0].astype(np.uint8)

        left_region = vertical_frame[crop_start:crop_end, :x_offset]
        h, w = left_region.shape[:2]
        for i in range(w):
            alpha = i/w
            color = (dominant_color * alpha).astype(np.uint8)
            left_region[:,i] = color
        vertical_frame[crop_start:crop_end, :x_offset] = left_region

        right_region = vertical_frame[crop_start:crop_end, x_offset+new_width:]
        h, w = right_region.shape[:2]
        for i in range(w):
            alpha = 1 - i/w
            color = (dominant_color * alpha).astype(np.uint8)
            right_region[:,i] = color
        vertical_frame[crop_start:crop_end, x_offset+new_width:] = right_region
        
        return vertical_frame

    def fill_background_with_graphics(self, vertical_frame, frame, start_height, remaining_height):
        """
        Fill background with animated graphics
        """
        pixels = frame.reshape(-1, 3)
        dominant_color = np.mean(pixels, axis=0).astype(int)
        vertical_frame[start_height:start_height+remaining_height, :] = dominant_color
        
        t = time.time() * 2
        y_coords = np.arange(start_height, start_height+remaining_height)
        x_coords = np.arange(0, self.target_width)
        X, Y = np.meshgrid(x_coords, y_coords)
        wave = np.sin(X/30 + t) * np.cos(Y/30 + t) * 20
        wave = wave.astype(np.uint8)
        
        vertical_frame[start_height:start_height+remaining_height, :] = np.clip(
            vertical_frame[start_height:start_height+remaining_height, :] + wave[:, :, np.newaxis], 0, 255)
        
        return vertical_frame
