import numpy as np
import cv2
import time

from reelix.config.configuration import Config
from reelix.model.model_control import DetectorFactory
from reelix.model.model_control import FaceDetectorFactory
from reelix.utils.viualisation import Visualizer
from reelix.processors.graphics_processor import GraphicsProcessor
from reelix.utils.printer import Printer

class FrameProcessor:
    
    def __init__(self, target_height, target_width, fps):
        self.prev_crops = None
        self.previous_frame = None
        self.target_height = target_height
        self.target_width = target_width
        self.previous_master = None
        self.fps = fps
        self.person_detector = DetectorFactory.get_detector(Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"])
        self.face_detector = FaceDetectorFactory.get_detector(Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"])
        self.visualizer = Visualizer(Config.OUTPUT_VIDEO_PARAMS["PREVIEW_DEBUG_PLAYER"], Config.OUTPUT_VIDEO_PARAMS["PREVIEW_VERTICAL_VIDEO"])
        self.printer = Printer()
        self.graphics_processor = GraphicsProcessor()
    
    def get_master_box(self, frame, padding_factor=0):
        frame_height, frame_width = frame.shape[:2]

        # Calculate crop width with 9:16 aspect ratio and padding
        crop_width = int(frame_height * (9/16))
        crop_width_with_padding = min(crop_width * (1 + 2*padding_factor), frame_width)

        # Center the crop box
        x1 = max(0, (frame_width - crop_width_with_padding) // 2)
        x2 = min(frame_width, x1 + crop_width_with_padding)

        # Adjust if out of bounds
        if x2 > frame_width:
            x2 = frame_width
            x1 = max(0, x2 - crop_width_with_padding)
        
        return (x1, 0, x2, frame_height)

    def smooth_bbox(self, prev_bbox, current_bbox, smoothing_factor=0.8):
        if self.previous_master is None:
            return current_bbox
            
        # Apply exponential smoothing to x, y positions of bounding box
        smoothed_x = int(prev_bbox[0] * smoothing_factor + current_bbox[0] * (1 - smoothing_factor))
        smoothed_y = int(prev_bbox[1] * smoothing_factor + current_bbox[1] * (1 - smoothing_factor))
        smoothed_w = int(prev_bbox[2] * smoothing_factor + current_bbox[2] * (1 - smoothing_factor))
        smoothed_h = int(prev_bbox[3] * smoothing_factor + current_bbox[3] * (1 - smoothing_factor))
        
        return (smoothed_x, smoothed_y, smoothed_w, smoothed_h)

    def create_tiled_vertical_frame(self, display_frame, frame, person_boxes, similarity_index=1):

        # Early validation
        if len(person_boxes) != 2:
            self.prev_crops = None
            return None, None
            
        frame_height, frame_width = frame.shape[:2]
        single_height = self.target_height // 2
        
        # Check if we can reuse previous crops
        if similarity_index > 0.85 and self.prev_crops is not None and len(self.prev_crops) == 2:
            return self._process_previous_crops(display_frame, frame, single_height)
            
        # Validate total area coverage
        total_box_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in person_boxes)
        self.visualizer.set_debug_props("person Area / threshold", f"{total_box_area / (frame_height * frame_width):.2f}/0.3")
        if total_box_area / (frame_height * frame_width) < 0.3:
            #print("No tile")
            self.prev_crops = None
            return frame, None
            
        person_boxes.sort(key=lambda x: x[0])
        
        # Apply smoothing to boxes
        if self.prev_crops is not None:
            person_boxes = self._smooth_boxes(person_boxes)
        
        # Handle overlapping boxes
        if person_boxes[0][2] > person_boxes[1][0]:
            person_boxes = self._handle_overlap(person_boxes)
            if person_boxes is None:
                return frame, None
        
        tiled_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        current_crops = []
        
        # Process each person box
        for i, box in enumerate(person_boxes):
            # Calculate crop dimensions
            face_height = (box[3] - box[1]) // 3
            x1 = box[0]
            x2 = min(frame_width, box[2])
            y1 = max(0, box[1] - int(face_height * 0.3))
            
            # Maintain 9:16 aspect ratio
            crop_width = x2 - x1
            required_height = int(crop_width * 16/9)
            y2 = min(frame_height, y1 + required_height)
            
            if y2 >= frame_height:
                y2 = frame_height
                y1 = max(0, y2 - required_height)
            
            # Apply position smoothing
            if self.prev_crops is not None and i < len(self.prev_crops):
                x1, y1, x2, y2 = self._smooth_position(x1, y1, x2, y2, self.prev_crops[i])
            
            person_crop = frame[y1:y2, x1:x2]
            current_crops.append((x1, y1, x2, y2, box[4]))

            # Resize crop
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

                # Draw bounding box
                #cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                #cv2.putText(display_frame, f"Person BB: {box[4]:.2f}", (x1, y1-10), 
                #           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                           
                y_offset = i * single_height
                tiled_frame[y_offset:y_offset+single_height, :] = resized_person

        self.prev_crops = current_crops
        cropped_frame = cv2.resize(tiled_frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        return display_frame, cropped_frame

    def _process_previous_crops(self, display_frame, frame, single_height):
        tiled_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        for i, prev_crop in enumerate(self.prev_crops):
            x1, y1, x2, y2, conf = prev_crop
            
            if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
                return frame, None
                
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return frame, None
                
            crop_width = x2 - x1
            if crop_width == 0:
                return frame, None
                
            scale = self.target_width / crop_width
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

    def _smooth_boxes(self, person_boxes):
        for i in range(len(person_boxes)):
            if i < len(self.prev_crops):
                prev_x1, prev_y1, prev_x2, prev_y2, prev_conf = self.prev_crops[i]
                curr_x1, curr_y1, curr_x2, curr_y2, person_conf = person_boxes[i]
                
                movement = max(abs(curr_x1 - prev_x1), abs(curr_x2 - prev_x2),
                            abs(curr_y1 - prev_y1), abs(curr_y2 - prev_y2))
                
                smoothing_factor = 0.95 if movement > 50 else 0.8
                
                person_boxes[i] = tuple(int(prev * smoothing_factor + curr * (1 - smoothing_factor))
                                    for prev, curr in zip((prev_x1, prev_y1, prev_x2, prev_y2),
                                                        (curr_x1, curr_y1, curr_x2, curr_y2))) + (person_conf,)
        return person_boxes

    def _handle_overlap(self, person_boxes):
        overlap = person_boxes[0][2] - person_boxes[1][0]
        overlap_percentage = overlap / (person_boxes[0][2] - person_boxes[0][0])
        self.visualizer.set_debug_props("Person Overlap %", f"{overlap_percentage:.2f}")
        if overlap_percentage > 0.8:
            self.prev_crops = None
            return None
            
        overlap_adjust = overlap // 2
        smoothing_factor = 0.7
        
        if self.prev_crops is not None:
            person_boxes[0] = (person_boxes[0][0], person_boxes[0][1],
                            int(self.prev_crops[0][2] * smoothing_factor + 
                                (person_boxes[0][2] - overlap_adjust) * (1 - smoothing_factor)),
                            person_boxes[0][3], person_boxes[0][4])
            person_boxes[1] = (int(self.prev_crops[1][0] * smoothing_factor + 
                                (person_boxes[1][0] + overlap_adjust) * (1 - smoothing_factor)),
                            person_boxes[1][1], person_boxes[1][2], person_boxes[1][3], person_boxes[1][4])
        else:
            person_boxes[0] = (person_boxes[0][0], person_boxes[0][1],
                            person_boxes[0][2] - overlap_adjust, person_boxes[0][3], person_boxes[0][4])
            person_boxes[1] = (person_boxes[1][0] + overlap_adjust, person_boxes[1][1],
                            person_boxes[1][2], person_boxes[1][3], person_boxes[1][4])
                            
        return person_boxes

    def _smooth_position(self, x1, y1, x2, y2, prev_crop):
        prev_x1, prev_y1, prev_x2, prev_y2, _ = prev_crop
        
        width_deviation = abs(x2 - x1 - (prev_x2 - prev_x1)) / (prev_x2 - prev_x1)
        height_deviation = abs(y2 - y1 - (prev_y2 - prev_y1)) / (prev_y2 - prev_y1)
        
        smoothing_factor = 0.95 if width_deviation > 0.15 or height_deviation > 0.15 else 0.8
        
        return (int(prev_x1 * smoothing_factor + x1 * (1 - smoothing_factor)),
                int(prev_y1 * smoothing_factor + y1 * (1 - smoothing_factor)),
                int(prev_x2 * smoothing_factor + x2 * (1 - smoothing_factor)),
                int(prev_y2 * smoothing_factor + y2 * (1 - smoothing_factor)))
    
    def center_master_box_on_face(self, debug_frame, frame, person_boxes, master_box):
        """
        Centers a person bounding box around a detected face.
        """
        px1, py1, px2, py2 = master_box
        person_width = px2 - px1
        
        # Run face detection directly on relevant region without creating copy
        face_results = self.face_detector.detect_faces(frame, master_box)
        #debug_frame = face_results[0].plot()
        print("Face results ", face_results)
        # Early return if no faces detected
        if not face_results[0] or not hasattr(face_results[0], 'boxes') or not face_results[0].boxes.xyxy.shape[0]:
            # If we have one person box, center on that
            if len(person_boxes) == 1:
                person_box = person_boxes[0]
                person_center_x = int((person_box[0] + person_box[2])/2)
                
                # Calculate new x coordinates to center master box on person
                half_width = person_width // 2
                new_px1 = person_center_x - half_width
                new_px2 = person_center_x + half_width
                
                # Bounds checking
                if new_px1 < 0:
                    new_px1 = 0 
                    new_px2 = person_width
                elif new_px2 > frame.shape[1]:
                    new_px2 = frame.shape[1]
                    new_px1 = new_px2 - person_width
                    
                return debug_frame, [new_px1, py1, new_px2, py2]
            
        # Get first face box coordinates and calculate center in one step
        face_box = face_results[0].boxes.xyxy[0].cpu().numpy()
        
        # Calculate face and master box areas
        face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
        master_area = (px2 - px1) * (py2 - py1)
        face_ratio = face_area / master_area
        
        # Draw face bounding box in violet color with label and confidence on debug_frame
        cv2.rectangle(debug_frame, 
                    (int(face_box[0] + px1), int(face_box[1] + py1)),
                    (int(face_box[2] + px1), int(face_box[3] + py1)), 
                    (238, 130, 238), 2)
        cv2.putText(debug_frame, f"Face: {face_results[0].boxes.conf[0]:.2f} Ratio: {face_ratio:.3f}", 
                    (int(face_box[0] + px1), int(face_box[1] + py1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 130, 238), 2)    

        # Return master box if face is too small

        if face_ratio < 0.002:
            return debug_frame, master_box
            
        face_center_x = int((face_box[0] + face_box[2])/2) + px1
        
        # Calculate new x coordinates to center person box on face
        half_width = person_width // 2
        new_px1 = face_center_x - half_width
        new_px2 = face_center_x + half_width
        
        # Bounds checking
        if new_px1 < 0:
            new_px1 = 0
            new_px2 = person_width
        elif new_px2 > frame.shape[1]:
            new_px2 = frame.shape[1]
            new_px1 = new_px2 - person_width    
        #print( "Adjusted the Master box for face!!")
        return debug_frame, [new_px1, py1, new_px2, py2]

    def get_region_of_interest(self, debug_frame, frame, person_boxes, masterbox, similarity_index, unfilled_dimensions=None):
        # Add exponential moving average for smoother transitions
        self.ema_alpha = 0.5 # Lower value = more smoothing
        
        if not hasattr(self, 'ema_master_x1'):
            self.ema_master_x1 = None
            self.ema_master_x2 = None
            
        #self.visualizer.set_debug_props("No of Persons ", len(person_boxes))
        if not person_boxes or len(person_boxes) == 0:
            cropped_frame = frame[masterbox[1]:masterbox[3], masterbox[0]:masterbox[2]]
            self.previous_master = masterbox
            self.ema_master_x1 = masterbox[0]
            self.ema_master_x2 = masterbox[2]
            # Draw thick red box around final crop region
            cv2.rectangle(debug_frame, (masterbox[0], masterbox[1]), (masterbox[2], masterbox[3]), (0,255,0), 6)
            cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)                        
            
        if len(person_boxes) == 2:
            if Config.OUTPUT_VIDEO_PARAMS["PERSON_TILED"]:
                # Check if both person boxes are mostly within master box
                box1_overlap = min(masterbox[2], person_boxes[0][2]) - max(masterbox[0], person_boxes[0][0])
                box1_width = person_boxes[0][2] - person_boxes[0][0]
                box1_coverage = box1_overlap / box1_width if box1_width > 0 else 0

                box2_overlap = min(masterbox[2], person_boxes[1][2]) - max(masterbox[0], person_boxes[1][0])
                box2_width = person_boxes[1][2] - person_boxes[1][0]
                box2_coverage = box2_overlap / box2_width if box2_width > 0 else 0

                if box1_coverage >= 0.9 and box2_coverage >= 0.9:
                    cropped_frame = frame[masterbox[1]:masterbox[3], masterbox[0]:masterbox[2]]
                    # Draw thick red box around final crop region
                    cv2.rectangle(debug_frame, (masterbox[0], masterbox[1]), (masterbox[2], masterbox[3]), (0,255,0), 6)
                    cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)            
                    return debug_frame, cropped_frame
                else:
                    debug_frame, tiled_frame = self.create_tiled_vertical_frame(debug_frame, frame, person_boxes, similarity_index)
                    if tiled_frame is not None:
                        return debug_frame, tiled_frame 
        
        # Check if exactly one person detected
        if len(person_boxes) == 1:

            # Get the single person box
            person_box = person_boxes[0]
            #print("Similarity Index ", similarity_index)
            if int(similarity_index) == 0:
                # Center the master box around the face
                debug_frame, adjusted_master_box = self.center_master_box_on_face(debug_frame, frame, person_boxes, masterbox)
            else:
                adjusted_master_box = masterbox
                
            # Apply enhanced temporal smoothing to reduce jitter
            if self.previous_master is not None:
                smoothing_factor = 0.95  # Increased smoothing
                smooth_x1 = int(self.previous_master[0] * smoothing_factor + adjusted_master_box[0] * (1 - smoothing_factor))
                smooth_x2 = int(self.previous_master[2] * smoothing_factor + adjusted_master_box[2] * (1 - smoothing_factor))
                adjusted_master_box = [smooth_x1, adjusted_master_box[1], smooth_x2, adjusted_master_box[3]]
                
            # Update master box coordinates    
            master_x1 = adjusted_master_box[0]
            master_x2 = adjusted_master_box[2]
            masterbox = adjusted_master_box
            
            # # Apply EMA smoothing
            # if self.ema_master_x1 is None:
            #     self.ema_master_x1 = master_x1
            #     self.ema_master_x2 = master_x2
            # else:
            #     self.ema_master_x1 = int(self.ema_alpha * master_x1 + (1-self.ema_alpha) * self.ema_master_x1)
            #     self.ema_master_x2 = int(self.ema_alpha * master_x2 + (1-self.ema_alpha) * self.ema_master_x2)
            #     master_x1 = self.ema_master_x1
            #     master_x2 = self.ema_master_x2
                
            masterbox = [master_x1, masterbox[1], master_x2, masterbox[3]]
            cropped_frame = frame[masterbox[1]:masterbox[3], master_x1:master_x2]
            # Draw thick red box around final crop region
            cv2.rectangle(debug_frame, (master_x1, masterbox[1]), (master_x2, masterbox[3]), (0,255,0), 6)
            cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)            
            self.previous_master = masterbox
            return debug_frame, cropped_frame   

        frame_height, frame_width = debug_frame.shape[:2]
        frame_center_x = frame_width // 2
        
        # Sort boxes by area and distance to center
        sorted_boxes = sorted(person_boxes, 
                            key=lambda box: ((box[2]-box[0])*(box[3]-box[1]), 
                                        -abs(((box[2]+box[0])//2) - frame_center_x)),
                            reverse=True)[:4]
        
        # Handle empty sorted_boxes
        if not sorted_boxes:
            # Return original frame and master box if no valid person boxes
            cropped_frame = frame[masterbox[1]:masterbox[3], masterbox[0]:masterbox[2]]
            target_width = unfilled_dimensions[0][3] if unfilled_dimensions is not None else self.target_width
            cropped_frame = cv2.resize(cropped_frame, (target_width, self.target_height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(debug_frame, (masterbox[0], masterbox[1]), (masterbox[2], masterbox[3]), (0,255,0), 6)
            cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)            
            return debug_frame, cropped_frame
            
        # Find combined bounds
        x_min = min(box[0] for box in sorted_boxes)
        x_max = max(box[2] for box in sorted_boxes)
        y_min, y_max = masterbox[1], masterbox[3]  # Use master box height
        
        # Draw boxes on debug frame only
        debug_frame_with_boxes = debug_frame.copy()
        for box in sorted_boxes:
            area = (box[2]-box[0])*(box[3]-box[1])
            cv2.rectangle(debug_frame_with_boxes, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
            cv2.putText(debug_frame_with_boxes, f"Person: {box[4]:.2f}, Area: {area}",
                        (box[0], box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
                        
        cv2.rectangle(debug_frame_with_boxes, (x_min, y_min), (x_max, y_max), (0,255,255), 2)
        cv2.putText(debug_frame_with_boxes, "Combined BB", (x_min, y_min+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # Calculate master box position centered on frame
        master_width = masterbox[2] - masterbox[0]
        master_x1 = max(0, frame_center_x - (master_width // 2))
        master_x2 = min(frame_width, master_x1 + master_width)
        
        if master_x2 == frame_width:
            master_x1 = master_x2 - master_width
            
        # Check if any person box is covered at least 80% by master box
        max_coverage = 0
        for box in sorted_boxes:
            box_width = box[2] - box[0]
            overlap_width = min(master_x2, box[2]) - max(master_x1, box[0])
            coverage = overlap_width / box_width if box_width > 0 else 0
            max_coverage = max(max_coverage, coverage)
        
        # If no box is covered well enough, center on largest box near center
        if max_coverage < 0.8:
            # Find largest box closest to center
            center_box = min(sorted_boxes[:2], 
                            key=lambda box: abs(((box[2]+box[0])//2) - frame_center_x))
            
            center_box_x = (center_box[0] + center_box[2]) // 2
            master_x1 = max(0, center_box_x - (master_width // 2))
            master_x2 = min(frame_width, master_x1 + master_width)
            
            if master_x2 == frame_width:
                master_x1 = master_x2 - master_width

        # Apply enhanced temporal smoothing
        if self.previous_master is not None:
            smoothing_factor = 0.9 # Increased smoothing
            master_x1 = int(self.previous_master[0] * smoothing_factor + master_x1 * (1 - smoothing_factor))
            master_x2 = int(self.previous_master[2] * smoothing_factor + master_x2 * (1 - smoothing_factor))
            
        # Apply additional EMA smoothing
        if self.ema_master_x1 is None:
            self.ema_master_x1 = master_x1
            self.ema_master_x2 = master_x2
        else:
            self.ema_master_x1 = int(self.ema_alpha * master_x1 + (1-self.ema_alpha) * self.ema_master_x1)
            self.ema_master_x2 = int(self.ema_alpha * master_x2 + (1-self.ema_alpha) * self.ema_master_x2)
            master_x1 = self.ema_master_x1
            master_x2 = self.ema_master_x2
                
        # Draw boxes within master box on debug frame only
        for box in sorted_boxes:
            if box[0] >= master_x1 and box[2] <= master_x2:
                cv2.rectangle(debug_frame_with_boxes, (box[0], box[1]), (box[2], box[3]), (42,42,165), 2)
                cv2.putText(debug_frame_with_boxes, "In Master Box", (box[0], box[3]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (42,42,165), 2)
        
        # Draw thick red box around final crop region
        cv2.rectangle(debug_frame_with_boxes, (master_x1, masterbox[1]), (master_x2, masterbox[3]), (0,255,0), 6)
        cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4) 

        self.previous_master = (master_x1, masterbox[1], master_x2, masterbox[3])
        cropped_frame = frame[masterbox[1]:masterbox[3], master_x1:master_x2]
        target_width = unfilled_dimensions[0][3] if unfilled_dimensions is not None else self.target_width
        cropped_frame = cv2.resize(cropped_frame, (target_width, self.target_height), interpolation=cv2.INTER_AREA)
        return debug_frame_with_boxes, cropped_frame   
    
    def fill_background_with_graphics(self, vertical_frame, frame, start_height, remaining_height, target_width):
        # Get dominant color from the frame
        pixels = frame.reshape(-1, 3)
        dominant_color = np.mean(pixels, axis=0).astype(int)
        
        # Fill remaining space with dominant color
        vertical_frame[start_height:start_height+remaining_height, :] = dominant_color
        
        # Add animated gradient wave pattern
        t = time.time() * 2  # Time factor for animation
        y_coords = np.arange(start_height, start_height+remaining_height)
        x_coords = np.arange(0, target_width)
        X, Y = np.meshgrid(x_coords, y_coords)
        wave = np.sin(X/30 + t) * np.cos(Y/30 + t) * 20
        wave = wave.astype(np.uint8)
        
        # Add wave pattern to background while preserving dominant color
        vertical_frame[start_height:start_height+remaining_height, :] = np.clip(
            vertical_frame[start_height:start_height+remaining_height, :] + wave[:, :, np.newaxis], 0, 255)
        
        return vertical_frame

    def place_main_frame(self, frame, masterbox, position=Config.CROP_PLACEMENT_PARAMS["TOP"]):
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        master_x, master_y, master_width, master_height = masterbox
        # Create black background frame with target dimensions
        vertical_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Calculate scaling factor to fit frame width
        scale = min(self.target_width / frame_width, self.target_height / frame_height)
        
        # Calculate new dimensions maintaining aspect ratio
        new_width = min(int(frame_width * scale), self.target_width)
        new_height = min(int(frame_height * scale), self.target_height)
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Calculate vertical position based on argument
        if position == Config.CROP_PLACEMENT_PARAMS["TOP"]:
            y_start = 0
            unfilled_dimensions = [(0, new_height, new_width, master_height-new_height)]
        elif position == Config.CROP_PLACEMENT_PARAMS["MIDDLE"]:
            y_start = max(0, min((self.target_height - new_height) // 2, self.target_height - new_height))
            unfilled_dimensions = [(0, 0, new_width, master_height // 2 -(new_height // 2)),
                                 (0, (master_height // 2) + (new_height // 2), new_width, (master_height // 2) - (new_height // 2))]
        elif position == Config.CROP_PLACEMENT_PARAMS["BOTTOM"]:
            y_start = max(0, self.target_height - new_height)
            unfilled_dimensions = [(0, 0, new_width, master_height-new_height)]
        else:
            # Default to top if position not recognized
            y_start = 0
            unfilled_height = min(max(0, self.target_height - new_height), self.target_height)
            unfilled_dimensions = [(new_height, 0, unfilled_height, self.target_width)]
            
        # Ensure frame placement stays within bounds
        y_end = min(y_start + new_height, self.target_height)
        x_end = min(new_width, self.target_width)
        
        # Copy resized frame into vertical frame at calculated position
        vertical_frame[y_start:y_end, 0:x_end] = resized_frame[0:y_end-y_start, 0:x_end]
        
        return vertical_frame, new_height, unfilled_dimensions 
       
    def fill_side_graphics(self, vertical_frame, cropped_frame, crop_start, crop_end, x_offset, new_width):
        # Get dominant color from cropped frame
        pixels = cropped_frame.reshape(-1, 3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_color = centers[0].astype(np.uint8)

        # Fill graphics in left side region with gradient using dominant color
        left_region = vertical_frame[crop_start:crop_end, :x_offset]
        h, w = left_region.shape[:2]
        for i in range(w):
            alpha = i/w
            color = (dominant_color * alpha).astype(np.uint8)
            left_region[:,i] = color
        vertical_frame[crop_start:crop_end, :x_offset] = left_region

        # Fill graphics in right side region with gradient using dominant color
        right_region = vertical_frame[crop_start:crop_end, x_offset+new_width:]
        h, w = right_region.shape[:2]
        for i in range(w):
            alpha = 1 - i/w
            color = (dominant_color * alpha).astype(np.uint8)
            right_region[:,i] = color
        vertical_frame[crop_start:crop_end, x_offset+new_width:] = right_region        
        return vertical_frame

    def place_cropped_frame(self, vertical_frame, cropped_frame, new_height, position='bottom'):
        if cropped_frame is not None:
            # Calculate remaining height
            remaining_height = self.target_height - new_height
            
            # Resize cropped frame to fit the width while maintaining aspect ratio
            crop_height, crop_width = cropped_frame.shape[:2]
            crop_scale = self.target_width / crop_width
            resized_crop = cv2.resize(cropped_frame, (self.target_width, int(crop_height * crop_scale)))
            
            # If resized crop is taller than remaining space, scale it down to fit
            if resized_crop.shape[0] > remaining_height:
                height_scale = remaining_height / resized_crop.shape[0]
                crop_aspect_ratio = resized_crop.shape[1] / resized_crop.shape[0]
                new_width = int(remaining_height * crop_aspect_ratio)
                resized_crop = cv2.resize(resized_crop, (new_width, remaining_height))
                
                # Center the crop horizontally
                x_offset = (self.target_width - new_width) // 2
                crop_start = 0 if position == Config.CROP_PLACEMENT_PARAMS["TOP"] else new_height
                crop_end = crop_start + remaining_height
                
                # Fill graphics in side regions
                vertical_frame = self.fill_side_graphics(vertical_frame, cropped_frame, crop_start, crop_end, x_offset, new_width)
                
                # Place the cropped frame
                vertical_frame[crop_start:crop_end, x_offset:x_offset+new_width] = resized_crop
            else:
                # Place the resized crop based on position
                crop_start = 0 if position == Config.CROP_PLACEMENT_PARAMS["TOP"] else new_height
                crop_end = crop_start + resized_crop.shape[0]
                
                # Clear target area and place crop
                vertical_frame[crop_start:crop_end, :] = 0
                vertical_frame[crop_start:crop_end, 0:self.target_width] = resized_crop
            
        return vertical_frame

    def get_noloss_region_of_interest(self, debug_frame, frame, person_boxes, masterbox, similarity_index, frame_time, frame_count, object_flag):
        vertical_frame, new_height, unfilled_dimensions = self.place_main_frame(frame, masterbox, Config.CROP_PLACEMENT_PARAMS["CURRENT_PLACEMENT"])
        return debug_frame, vertical_frame   
     
    def filter_person_boxes(self, person_boxes, frame_shape, min_size=50, boundary_threshold=20):
        """
        Filters out person boxes that are too small or too close to frame boundaries
        Args:
            person_boxes: List of person bounding boxes 
            frame_shape: Shape of the frame (height, width)
            min_size: Minimum box area to keep
            boundary_threshold: Minimum distance from frame edges
        Returns:
            Filtered list of person boxes
        """
        height, width = frame_shape[:2]
        
        # Calculate boundary regions (20% of frame dimensions)
        boundary_width = int(0.15 * width)
        boundary_height = int(0.15 * height)
        
        boxes_to_remove = []
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        
            # Calculate box area
            box_area = (x2 - x1) * (y2 - y1)
            
            # Check if box is too small
            if box_area < min_size:
                boxes_to_remove.append(i)
                continue
            
            # Calculate area of box within boundary regions
            boundary_area = 0
            
            # Left boundary
            if x1 < boundary_width:
                boundary_area += (min(boundary_width, x2) - x1) * (y2 - y1)
                
            # Right boundary    
            if x2 > (width - boundary_width):
                boundary_area += (x2 - max(width - boundary_width, x1)) * (y2 - y1)
                
            # Top boundary
            if y1 < boundary_height:
                boundary_area += (min(boundary_height, y2) - y1) * (x2 - x1)
                
            # Bottom boundary
            if y2 > (height - boundary_height):
                boundary_area += (y2 - max(height - boundary_height, y1)) * (x2 - x1)
                
            # Check if box area within boundary regions is >= 80% of total box area
            if boundary_area / box_area >= 0.8:
                boxes_to_remove.append(i)
                continue

            # Check overlap with other boxes
            for j, other_box in enumerate(person_boxes):
                if i != j:
                    ox1, oy1, ox2, oy2 = other_box[0], other_box[1], other_box[2], other_box[3]
                    
                    # Calculate intersection area
                    x_left = max(x1, ox1)
                    y_top = max(y1, oy1)
                    x_right = min(x2, ox2)
                    y_bottom = min(y2, oy2)
                    
                    if x_right > x_left and y_bottom > y_top:
                        intersection = (x_right - x_left) * (y_bottom - y_top)
                        if intersection / box_area >= 0.9:
                            boxes_to_remove.append(i)
                            break
                
        # Remove boxes in reverse order to maintain correct indices
        for i in sorted(boxes_to_remove, reverse=True):
            person_boxes.pop(i)
            
        return person_boxes 
    
    def get_object_region_of_interest(self, debug_frame, frame, object_boxes, masterbox, similarity_index):
        # Check if exactly one object detected
        if len(object_boxes) == 1:
            # Get the single object box
            object_box = object_boxes[0]
            
            # Calculate center of object box
            object_center_x = int((object_box[0] + object_box[2])/2)
            
            # Calculate new x coordinates to center master box on object
            master_width = masterbox[2] - masterbox[0]
            half_width = master_width // 2
            new_x1 = object_center_x - half_width
            new_x2 = object_center_x + half_width
            
            # Bounds checking
            if new_x1 < 0:
                new_x1 = 0
                new_x2 = master_width
            elif new_x2 > frame.shape[1]:
                new_x2 = frame.shape[1]
                new_x1 = new_x2 - master_width
                
            # Apply temporal smoothing if previous box exists
            if hasattr(self, 'prev_masterbox') and self.prev_masterbox is not None:
                smoothing_factor = 0.8
                new_x1 = int(self.prev_masterbox[0] * smoothing_factor + new_x1 * (1 - smoothing_factor))
                new_x2 = int(self.prev_masterbox[2] * smoothing_factor + new_x2 * (1 - smoothing_factor))
                
            # Create adjusted master box centered on object
            adjusted_masterbox = [new_x1, masterbox[1], new_x2, masterbox[3]]
            
            # Store current box for next frame
            self.prev_masterbox = adjusted_masterbox
            
            # Crop frame using adjusted master box coordinates
            cropped_frame = frame[adjusted_masterbox[1]:adjusted_masterbox[3], 
                                adjusted_masterbox[0]:adjusted_masterbox[2]]
            
            # Draw bounding box on debug frame
            cv2.rectangle(debug_frame, (adjusted_masterbox[0], adjusted_masterbox[1]), 
                        (adjusted_masterbox[2], adjusted_masterbox[3]), (0,255,0), 6)
            cv2.putText(debug_frame, "ROI", (adjusted_masterbox[0], adjusted_masterbox[1]+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                        
            return debug_frame, cropped_frame
            
        elif len(object_boxes) > 1:
            # Get frame center x coordinate
            frame_center_x = frame.shape[1] // 2
            
            # Sort boxes by area and distance to center
            sorted_boxes = sorted(object_boxes, 
                                key=lambda box: ((box[2]-box[0])*(box[3]-box[1]), 
                                            -abs(((box[2]+box[0])//2) - frame_center_x)),
                                reverse=True)
            
            # Get largest box closest to center
            object_box = sorted_boxes[0]
            
            # Calculate center of selected object box
            object_center_x = int((object_box[0] + object_box[2])/2)
            
            # Calculate new x coordinates to center master box on object
            master_width = masterbox[2] - masterbox[0]
            half_width = master_width // 2
            new_x1 = object_center_x - half_width
            new_x2 = object_center_x + half_width
            
            # Bounds checking
            if new_x1 < 0:
                new_x1 = 0
                new_x2 = master_width
            elif new_x2 > frame.shape[1]:
                new_x2 = frame.shape[1]
                new_x1 = new_x2 - master_width
                
            # Apply temporal smoothing if previous box exists
            if hasattr(self, 'prev_masterbox') and self.prev_masterbox is not None:
                smoothing_factor = 0.8
                new_x1 = int(self.prev_masterbox[0] * smoothing_factor + new_x1 * (1 - smoothing_factor))
                new_x2 = int(self.prev_masterbox[2] * smoothing_factor + new_x2 * (1 - smoothing_factor))
                
            # Create adjusted master box centered on object
            adjusted_masterbox = [new_x1, masterbox[1], new_x2, masterbox[3]]
            
            # Store current box for next frame
            self.prev_masterbox = adjusted_masterbox
            
            # Crop frame using adjusted master box coordinates
            cropped_frame = frame[adjusted_masterbox[1]:adjusted_masterbox[3], 
                                adjusted_masterbox[0]:adjusted_masterbox[2]]
            
            # Draw bounding box on debug frame
            cv2.rectangle(debug_frame, (adjusted_masterbox[0], adjusted_masterbox[1]), 
                        (adjusted_masterbox[2], adjusted_masterbox[3]), (0,255,0), 6)
            cv2.putText(debug_frame, "ROI", (adjusted_masterbox[0], adjusted_masterbox[1]+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                        
            return debug_frame, cropped_frame
            
        else:
            print("In else and processing previous box!")
            # If no objects detected, use previous master box if available
            if hasattr(self, 'prev_masterbox') and self.prev_masterbox is not None:
                cropped_frame = frame[self.prev_masterbox[1]:self.prev_masterbox[3],
                                    self.prev_masterbox[0]:self.prev_masterbox[2]]
                print("In else and processing previous box!")
                # Draw bounding box on debug frame
                cv2.rectangle(debug_frame, (self.prev_masterbox[0], self.prev_masterbox[1]),
                            (self.prev_masterbox[2], self.prev_masterbox[3]), (0,255,0), 6)
                cv2.putText(debug_frame, "ROI", (self.prev_masterbox[0], self.prev_masterbox[1]+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                            
                return debug_frame, cropped_frame
            
            # If no previous box, use original master box
            cropped_frame = frame[masterbox[1]:masterbox[3], masterbox[0]:masterbox[2]]
            cv2.rectangle(debug_frame, (masterbox[0], masterbox[1]),
                        (masterbox[2], masterbox[3]), (0,255,0), 6) 
            cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
            return debug_frame, cropped_frame               
    
    def process_frame(self, frame, mode, frame_time, frame_count, class_id=None):
        # Start time for FPS calculation
        start_time = time.time()
        
        # Create copy for debug visualization
        debug_frame = frame.copy()
        previous_track = frame.copy()

        # Initialize counter for empty frames if not exists
        if not hasattr(self, 'empty_frames_counter'):
            self.empty_frames_counter = 0

        # Detect objects based on provided class IDs
        if class_id:
            boxes, results = self.person_detector.detect_classid(frame, class_id)
            if Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"] == Config.MODEL_PARAMS["MODEL_TYPE_YOLO"]:
                debug_frame = results[0].plot()
            self.visualizer.set_debug_props(f"No of {class_id}", len(boxes))
            object_boxes = boxes
            
            # Increment counter if no objects detected
            if len(object_boxes) == 0:
                self.empty_frames_counter += 1
            else:
                self.empty_frames_counter = 0
                
        elif class_id is None or len(object_boxes) == 0 or self.empty_frames_counter >= 25:
            # Reset counter when switching to person detection
            self.empty_frames_counter = 0
            
            # Default to person detection if no class IDs provided or too many empty frames
            person_boxes, results = self.person_detector.detect_persons(frame)
            if Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"] == Config.MODEL_PARAMS["MODEL_TYPE_YOLO"]:
                debug_frame = results[0].plot()
            self.visualizer.set_debug_props("No of Persons", len(person_boxes))
            filtered_person_box = self.filter_person_boxes(person_boxes,frame.shape)
            object_boxes = filtered_person_box
            self.visualizer.set_debug_props("No of person post filter", len(person_boxes))

        self.visualizer.add_title(debug_frame, "Debug Frame")  
        masterbox = self.get_master_box(frame)

        similarity_index = 0
        if self.previous_frame is not None:
            similarity_index = self.compare_frames(frame, self.previous_frame)
        
        if Config.OUTPUT_VIDEO_PARAMS["NO_LOSS_TILED"]:
            object_flag = class_id and  (self.empty_frames_counter <=25 or len(object_boxes) > 0 )
            debug_frame, processed_frame = self.get_noloss_region_of_interest(debug_frame, frame, object_boxes, masterbox, similarity_index, frame_time, frame_count, object_flag)
        else:
            if class_id  and  (self.empty_frames_counter <=25 or len(object_boxes) > 0 ):
                debug_frame, processed_frame = self.get_object_region_of_interest(debug_frame, frame, object_boxes, masterbox, similarity_index)
            else:
                debug_frame, processed_frame = self.get_region_of_interest(debug_frame, frame, object_boxes, masterbox, similarity_index)
            
        #assiming the current frame to previous frame for smart processing.
        self.previous_frame = previous_track

        # Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        self.visualizer.set_debug_props("Similarity Index",f"{similarity_index:.2f}")
        self.visualizer.set_debug_props("Output FPS", f"{fps:.2f}")
        self.visualizer.set_debug_props("Model Type", Config.MODEL_PARAMS["CURRENT_MODEL_TYPE"])
        self.visualizer.set_debug_props("Object Conf", Config.MODEL_PARAMS["OBJECT_CONFIDENCE_THRESHOLD"])
        self.visualizer.set_debug_props("Input Type", Config.INPUT_PARAMS["CURRENT_INPUT_TYPE"])  
        self.visualizer.set_debug_props("Empty Frames", self.empty_frames_counter)
        debug_frame = self.visualizer.draw_debug_props(debug_frame)
        return debug_frame, processed_frame     
    
    @staticmethod
    def get_target_dimensions(frame_height, frame_width):
        """
        Calculate target dimensions based on mode while preserving aspect ratio
        Args:
            frame: Input video frame
            mode: 'vertical' or 'square' 
        Returns:
            target_height, target_width for vertical mode
            target_size for square mode
        """
        height = frame_height
        width = frame_width
        target_aspect_ratio = 9/16

        if Config.OUTPUT_VIDEO_PARAMS["VERTICAL"]:
            # For vertical mode, enforce 9:16 aspect ratio
            target_height = height  # Use input frame height            
            target_width = int(target_height * target_aspect_ratio)
            return target_height, target_width
            
        elif Config.OUTPUT_VIDEO_PARAMS["SQUARE"]:
            # For square mode, calculate dimensions to fit within 9:16
            target_height = height
            target_width = int(target_height * target_aspect_ratio)
            target_size = min(target_width, target_height)
            return target_size, target_size
            
        else:
            # Default to 9:16 dimensions if invalid mode
            target_height = height
            target_width = int(target_height * target_aspect_ratio)
            return target_height, target_width  
                      
    def compare_frames(self, current_frame, previous_frame):
        """
        Compare current and previous frames to detect changes
        Returns a similarity score between 0-1 where 1 means frames are identical
        """
        # Convert frames to grayscale
        if current_frame is None or previous_frame is None:
            return 0
            
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Ensure frames are same size
        if curr_gray.shape != prev_gray.shape:
            prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))
        
        # Calculate Mean Squared Error between frames
        err = np.sum((curr_gray.astype("float") - prev_gray.astype("float")) ** 2)
        err /= float(curr_gray.shape[0] * curr_gray.shape[1])
        
        # Convert MSE to similarity score between 0-1
        similarity_index = 1 - min(err / 255.0, 1.0)
        
        return similarity_index

    def calculate_motion_score(self, current_frame, previous_frame):
        """
        Calculate motion score between consecutive frames using optical flow
        Returns a motion score between 0-1 where 0 means no motion and 1 means maximum motion
        """
        # Handle empty frames
        if current_frame is None or previous_frame is None:
            return 0
            
        # Convert frames to grayscale
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Ensure frames are same size
        if curr_gray.shape != prev_gray.shape:
            prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, 
                                        flags=0)
                                        
        # Calculate magnitude of flow vectors
        magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        
        # Normalize motion score between 0-1
        motion_score = np.mean(magnitude) / 100.0  # Divide by 100 to normalize large movements
        motion_score = min(motion_score, 1.0)
        
        return motion_score

    def get_smoothing_threshold(self, motion_score):
        """
        Calculate smoothing threshold based on motion score
        Returns a threshold value between 0.3-0.9 where:
        - Higher motion (motion_score close to 1) = lower threshold for more aggressive smoothing
        - Lower motion (motion_score close to 0) = higher threshold for less smoothing
        """
        # Base threshold range
        min_threshold = 0.3  # More aggressive smoothing
        max_threshold = 0.9  # Less smoothing
        
        # Inverse relationship between motion and threshold
        # As motion increases, threshold decreases
        threshold = max_threshold - (motion_score * (max_threshold - min_threshold))
        
        # Ensure threshold stays within bounds
        threshold = max(min_threshold, min(threshold, max_threshold))
        
        return threshold