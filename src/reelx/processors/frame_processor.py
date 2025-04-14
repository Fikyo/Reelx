import numpy as np
import cv2
import time

from reelx.config.configuration import Config
from reelx.model.model_control import DetectorFactory, FaceDetectorFactory
from reelx.utils.viualisation import Visualizer
from reelx.processors.graphics_processor import GraphicsProcessor
from reelx.utils.printer import Printer
from reelx.processors.frame_comparison import FrameComparison
from reelx.processors.box_processor import BoxProcessor
from reelx.processors.frame_composition import FrameComposition
from reelx.processors.roi_processor import ROIProcessor

class FrameProcessor:
    def __init__(self, target_height, target_width, fps):
        self.prev_crops = None
        self.previous_frame = None
        self.target_height = target_height
        self.target_width = target_width
        self.fps = fps
        
        # Tracking history for class IDs
        self.tracking_history = {}  # {class_id: [(box, frame_number), ...]}
        self.current_frame_number = 0
        self.max_history_frames = 30  # Store last 30 frames of history
        
        # Initialize detectors
        self.person_detector = DetectorFactory.get_detector(Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"])
        self.face_detector = FaceDetectorFactory.get_detector(Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"])
        
        # Initialize processors
        self.visualizer = Visualizer(Config.OUTPUT_VIDEO_PARAMS["PREVIEW_DEBUG_PLAYER"], 
                                   Config.OUTPUT_VIDEO_PARAMS["PREVIEW_VERTICAL_VIDEO"])
        self.printer = Printer()
        self.graphics_processor = GraphicsProcessor()
        self.frame_comparison = FrameComparison()
        self.box_processor = BoxProcessor()
        self.frame_composition = FrameComposition(target_height, target_width)
        self.roi_processor = ROIProcessor(self.face_detector)  # No longer needs target dimensions since they come from unfilled_dimensions
        
        # Initialize counter for empty frames
        self.empty_frames_counter = 0

    def create_tiled_vertical_frame(self, display_frame, frame, person_boxes, similarity_index=1):
        """
        Create a tiled vertical frame from detected person boxes
        """
        if len(person_boxes) != 2:
            self.prev_crops = None
            return None, None
            
        frame_height, frame_width = frame.shape[:2]
        single_height = self.target_height // 2
        
        if similarity_index > 0.85 and self.prev_crops is not None and len(self.prev_crops) == 2:
            return self._process_previous_crops(display_frame, frame, single_height)
            
        total_box_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in person_boxes)
        self.visualizer.set_debug_props("person Area / threshold", f"{total_box_area / (frame_height * frame_width):.2f}/0.3")
        if total_box_area / (frame_height * frame_width) < 0.3:
            self.prev_crops = None
            return frame, None
            
        person_boxes.sort(key=lambda x: x[0])
        
        if self.prev_crops is not None:
            # Convert box format for smoothing with adaptive smoothing factor
            for i, box in enumerate(person_boxes):
                if i < len(self.prev_crops):
                    prev_box = self.prev_crops[i][:4]  # Get x1,y1,x2,y2 from prev_crop
                    
                    # Calculate deviation in position and size
                    width_deviation = abs((box[2] - box[0]) - (prev_box[2] - prev_box[0])) / (prev_box[2] - prev_box[0])
                    pos_deviation = abs((box[0] + box[2])/2 - (prev_box[0] + prev_box[2])/2) / frame_width
                    
                    # Use higher smoothing factor for larger deviations
                    smoothing_factor = 0.95 if (width_deviation > 0.2 or pos_deviation > 0.1) else 0.8
                    smoothed = self.box_processor.smooth_bounding_box(prev_box, (box[0], box[1], box[2], box[3]), smoothing_factor)
                    person_boxes[i] = (smoothed[0], smoothed[1], smoothed[2], smoothed[3], box[4])
        
        if person_boxes[0][2] > person_boxes[1][0]:
            person_boxes = self.box_processor.resolve_box_overlap(person_boxes)
            if person_boxes is None:
                return frame, None
        
        tiled_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        current_crops = []
        
        for i, box in enumerate(person_boxes):
            face_height = (box[3] - box[1]) // 3
            x1 = box[0]
            x2 = min(frame_width, box[2])
            y1 = max(0, box[1] - int(face_height * 0.3))
            
            crop_width = x2 - x1
            required_height = int(crop_width * 16/9)
            y2 = min(frame_height, y1 + required_height)
            
            if y2 >= frame_height:
                y2 = frame_height
                y1 = max(0, y2 - required_height)
            
            if self.prev_crops is not None and i < len(self.prev_crops):
                x1, y1, x2, y2 = self.box_processor.smooth_box_position(x1, y1, x2, y2, self.prev_crops[i])
            
            person_crop = frame[y1:y2, x1:x2]
            current_crops.append((x1, y1, x2, y2, box[4]))

            scale = self.target_width / (x2 - x1)
            new_height = int((y2 - y1) * scale)
            
            if new_height > 0:
                resized_person = cv2.resize(person_crop, (self.target_width, new_height))
                
                if new_height > single_height:
                    resized_person = resized_person[:single_height, :]
                else:
                    temp = np.zeros((single_height, self.target_width, 3), dtype=np.uint8)
                    temp[:new_height, :] = resized_person
                    resized_person = temp
                           
                y_offset = i * single_height
                tiled_frame[y_offset:y_offset+single_height, :] = resized_person

        self.prev_crops = current_crops
        cropped_frame = cv2.resize(tiled_frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        return display_frame, cropped_frame

    def _predict_tracked_boxes(self, class_id):
        """
        Predict tracked object positions using motion estimation and historical data
        """
        if class_id not in self.tracking_history or len(self.tracking_history[class_id]) < 2:
            return None
            
        history = self.tracking_history[class_id]
        recent_boxes = [boxes for boxes, _ in history[-3:]]  # Use last 3 frames
        
        if not all(recent_boxes):
            return None
            
        predicted_boxes = []
        for box_idx in range(len(recent_boxes[-1])):
            if box_idx >= len(recent_boxes[0]) or box_idx >= len(recent_boxes[1]):
                continue
                
            # Calculate velocity from last 3 positions
            velocities = []
            for i in range(len(recent_boxes)-1):
                box1 = recent_boxes[i][box_idx]
                box2 = recent_boxes[i+1][box_idx]
                dx = (box2[2] + box2[0])/2 - (box1[2] + box1[0])/2
                dy = (box2[3] + box2[1])/2 - (box1[3] + box1[1])/2
                dw = (box2[2] - box2[0]) - (box1[2] - box1[0])
                dh = (box2[3] - box2[1]) - (box1[3] - box1[1])
                velocities.append((dx, dy, dw, dh))
                
            # Average velocities with decay
            if len(velocities) >= 2:
                weights = [0.7, 0.3]  # More weight to recent movement
                avg_dx = sum(v[0] * w for v, w in zip(velocities, weights))
                avg_dy = sum(v[1] * w for v, w in zip(velocities, weights))
                avg_dw = sum(v[2] * w for v, w in zip(velocities, weights))
                avg_dh = sum(v[3] * w for v, w in zip(velocities, weights))
                
                # Get last known box
                last_box = recent_boxes[-1][box_idx]
                
                # Predict new position
                center_x = (last_box[2] + last_box[0])/2 + avg_dx
                center_y = (last_box[3] + last_box[1])/2 + avg_dy
                width = (last_box[2] - last_box[0]) + avg_dw
                height = (last_box[3] - last_box[1]) + avg_dh
                
                # Convert back to box format
                x1 = int(center_x - width/2)
                y1 = int(center_y - height/2)
                x2 = int(center_x + width/2)
                y2 = int(center_y + height/2)
                
                # Use slightly lower confidence for predicted boxes
                conf = max(0.3, last_box[4] * 0.8)  # Reduce confidence but keep minimum threshold
                
                predicted_boxes.append((x1, y1, x2, y2, conf))
                
        return predicted_boxes if predicted_boxes else None

    def _process_previous_crops(self, display_frame, frame, single_height):
        """
        Process frame using previous crop dimensions
        """
        tiled_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        for i, prev_crop in enumerate(self.prev_crops):
            x1, y1, x2, y2, conf = prev_crop
            
            if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
                return frame, None
                
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0 or x2 - x1 == 0:
                return frame, None
                
            scale = self.target_width / (x2 - x1)
            new_height = int(person_crop.shape[0] * scale)
            
            if new_height == 0:
                return frame, None
                
            resized_person = cv2.resize(person_crop, (self.target_width, new_height))
            
            if new_height > single_height:
                resized_person = resized_person[:single_height, :]
            else:
                temp = np.zeros((single_height, self.target_width, 3), dtype=np.uint8)
                temp[:new_height, :] = resized_person
                resized_person = temp

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f"Person BB: {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                        
            y_offset = i * single_height
            tiled_frame[y_offset:y_offset+single_height, :] = resized_person
            
        return display_frame, tiled_frame

    def process_frame(self, frame, mode, frame_time, frame_count, class_id=None):
        """
        Process a single frame
        """
        start_time = time.time()
        debug_frame = frame.copy()
        previous_track = frame.copy()

        object_boxes = None
        self.current_frame_number += 1
        
        if class_id:
            boxes, results = self.person_detector.detect_classid(frame, class_id)
            if Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"] == Config.MODEL_PARAMS["MODEL_TYPE_YOLO"]:
                debug_frame = results[0].plot()
            
            # Update tracking history
            if len(boxes) > 0:
                if class_id not in self.tracking_history:
                    self.tracking_history[class_id] = []
                self.tracking_history[class_id].append((boxes, self.current_frame_number))
                # Keep only recent history
                self.tracking_history[class_id] = [
                    (b, f) for b, f in self.tracking_history[class_id] 
                    if self.current_frame_number - f <= self.max_history_frames
                ]
                object_boxes = boxes
                self.empty_frames_counter = 0
            else:
                # Try to predict box positions using motion tracking
                predicted_boxes = self._predict_tracked_boxes(class_id)
                if predicted_boxes:
                    object_boxes = predicted_boxes
                    self.empty_frames_counter = 0
                else:
                    self.empty_frames_counter += 1
                    
            self.visualizer.set_debug_props(f"No of {class_id}", len(boxes))
                
        elif class_id is None or object_boxes is None or len(object_boxes) == 0 or self.empty_frames_counter >= 25:
            self.empty_frames_counter = 0
            person_boxes, results = self.person_detector.detect_persons(frame)
            if Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"] == Config.MODEL_PARAMS["MODEL_TYPE_YOLO"]:
                debug_frame = results[0].plot()
            self.visualizer.set_debug_props("No of Persons", len(person_boxes))
            filtered_person_box = self.box_processor.filter_person_boxes(person_boxes, frame.shape)
            object_boxes = filtered_person_box
            self.visualizer.set_debug_props("No of person post filter", len(person_boxes))

        self.visualizer.add_title(debug_frame, "Debug Frame")  
        masterbox = self.frame_composition.get_master_box(frame)

        similarity_index = 0
        if self.previous_frame is not None:
            similarity_index = self.frame_comparison.compare_frames(frame, self.previous_frame)
        
        if Config.OUTPUT_VIDEO_PARAMS["PERSON_TILED"] and object_boxes and len(object_boxes) == 2:
            debug_frame, processed_frame = self.create_tiled_vertical_frame(debug_frame, frame, object_boxes, similarity_index)
            if processed_frame is None:
                # Fallback to normal processing if tiled frame creation fails
                object_flag = class_id and (self.empty_frames_counter <= 25 or len(object_boxes) > 0)
                debug_frame, processed_frame = self.get_noloss_region_of_interest(debug_frame, frame, object_boxes, masterbox, similarity_index, frame_time, frame_count, object_flag)
        elif Config.OUTPUT_VIDEO_PARAMS["NO_LOSS_TILED"]:
            object_flag = class_id and (self.empty_frames_counter <= 25 or len(object_boxes) > 0)
            debug_frame, processed_frame = self.get_noloss_region_of_interest(debug_frame, frame, object_boxes, masterbox, similarity_index, frame_time, frame_count, object_flag)
        else:
            # Create unfilled_dimensions for non-tiled mode
            unfilled_dimensions = [(0, 0, frame.shape[1], self.target_width, self.target_height)]
            if class_id and (self.empty_frames_counter <= 25 or len(object_boxes) > 0):
                debug_frame, processed_frame = self.roi_processor.get_object_region_of_interest(debug_frame, frame, object_boxes, masterbox, similarity_index, unfilled_dimensions)
            else:
                debug_frame, processed_frame = self.roi_processor.get_region_of_interest(debug_frame, frame, object_boxes, masterbox, similarity_index, unfilled_dimensions)
            
        self.previous_frame = previous_track

        fps = 1.0 / (time.time() - start_time)
        self.visualizer.set_debug_props("Similarity Index", f"{similarity_index:.2f}")
        self.visualizer.set_debug_props("Output FPS", f"{fps:.2f}")
        self.visualizer.set_debug_props("Model Type", Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"])
        self.visualizer.set_debug_props("Object Conf", Config.MODEL_PARAMS["OBJECT_CONFIDENCE_THRESHOLD"])
        self.visualizer.set_debug_props("Input Type", Config.INPUT_PARAMS["CURRENT_INPUT_TYPE"])  
        self.visualizer.set_debug_props("Empty Frames", self.empty_frames_counter)
        debug_frame = self.visualizer.draw_debug_props(debug_frame)
        return debug_frame, processed_frame

    def get_noloss_region_of_interest(self, debug_frame, frame, person_boxes, masterbox, similarity_index, frame_time, frame_count, object_flag):
        """
        Get region of interest with no loss of information
        """
        vertical_frame, new_height, unfilled_dimensions = self.frame_composition.place_main_frame(frame, masterbox, Config.CROP_PLACEMENT_PARAMS["CURRENT_PLACEMENT"])
        return debug_frame, vertical_frame

    @staticmethod
    def get_target_dimensions(frame_height, frame_width):
        """
        Calculate target dimensions based on mode while preserving aspect ratio
        """
        height = frame_height
        width = frame_width
        target_aspect_ratio = 9/16

        if Config.OUTPUT_VIDEO_PARAMS["VERTICAL"]:
            target_height = height
            target_width = int(target_height * target_aspect_ratio)
            return target_height, target_width
            
        elif Config.OUTPUT_VIDEO_PARAMS["SQUARE"]:
            # For square output in vertical video, maintain vertical dimensions
            # This will create a vertical video with black bars and a square in the middle
            target_height = height
            target_width = int(target_height * 9/16)  # Keep vertical video dimensions
            return target_height, target_width
            
        else:
            target_height = height
            target_width = int(target_height * target_aspect_ratio)
            return target_height, target_width
