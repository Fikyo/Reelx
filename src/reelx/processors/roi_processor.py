import cv2
import numpy as np
from reelx.config.configuration import Config

class ROIProcessor:
    def __init__(self, face_detector):
        self.face_detector = face_detector
        self.previous_master = None
        self.prev_masterbox = None
        self.ema_master_x1 = None
        self.ema_master_x2 = None
        self.ema_alpha = 0.5

    def center_master_box_on_face(self, debug_frame, frame, person_boxes, masterbox):
        """
        Centers a person bounding box around a detected face.
        """
        px1, py1, px2, py2 = masterbox
        person_width = px2 - px1
        
        face_results = self.face_detector.detect_faces(frame, masterbox)
        print("Face results ", face_results)
        
        if not face_results[0] or not hasattr(face_results[0], 'boxes') or not face_results[0].boxes.xyxy.shape[0]:
            if len(person_boxes) == 1:
                person_box = person_boxes[0]
                person_center_x = int((person_box[0] + person_box[2])/2)
                
                half_width = person_width // 2
                new_px1 = person_center_x - half_width
                new_px2 = person_center_x + half_width
                
                if new_px1 < 0:
                    new_px1 = 0 
                    new_px2 = person_width
                elif new_px2 > frame.shape[1]:
                    new_px2 = frame.shape[1]
                    new_px1 = new_px2 - person_width
                    
                return debug_frame, [new_px1, py1, new_px2, py2]
            
        face_box = face_results[0].boxes.xyxy[0].cpu().numpy()
        face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
        master_area = (px2 - px1) * (py2 - py1)
        face_ratio = face_area / master_area
        
        cv2.rectangle(debug_frame, 
                    (int(face_box[0] + px1), int(face_box[1] + py1)),
                    (int(face_box[2] + px1), int(face_box[3] + py1)), 
                    (238, 130, 238), 2)
        cv2.putText(debug_frame, f"Face: {face_results[0].boxes.conf[0]:.2f} Ratio: {face_ratio:.3f}", 
                    (int(face_box[0] + px1), int(face_box[1] + py1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 130, 238), 2)    

        if face_ratio < 0.002:
            return debug_frame, masterbox
            
        face_center_x = int((face_box[0] + face_box[2])/2) + px1
        half_width = person_width // 2
        new_px1 = face_center_x - half_width
        new_px2 = face_center_x + half_width
        
        if new_px1 < 0:
            new_px1 = 0
            new_px2 = person_width
        elif new_px2 > frame.shape[1]:
            new_px2 = frame.shape[1]
            new_px1 = new_px2 - person_width    

        return debug_frame, [new_px1, py1, new_px2, py2]

    def get_object_region_of_interest(self, debug_frame, frame, object_boxes, masterbox, similarity_index, unfilled_dimensions=None):
        """
        Get region of interest based on object detection
        """
        if len(object_boxes) == 1:
            object_box = object_boxes[0]
            object_center_x = int((object_box[0] + object_box[2])/2)
            
            master_width = masterbox[2] - masterbox[0]
            half_width = master_width // 2
            new_x1 = object_center_x - half_width
            new_x2 = object_center_x + half_width
            
            if new_x1 < 0:
                new_x1 = 0
                new_x2 = master_width
            elif new_x2 > frame.shape[1]:
                new_x2 = frame.shape[1]
                new_x1 = new_x2 - master_width
                
            if self.prev_masterbox is not None:
                smoothing_factor = 0.8
                new_x1 = int(self.prev_masterbox[0] * smoothing_factor + new_x1 * (1 - smoothing_factor))
                new_x2 = int(self.prev_masterbox[2] * smoothing_factor + new_x2 * (1 - smoothing_factor))
                
            adjusted_masterbox = [new_x1, masterbox[1], new_x2, masterbox[3]]
            self.prev_masterbox = adjusted_masterbox
            
            cropped_frame = frame[adjusted_masterbox[1]:adjusted_masterbox[3], 
                                adjusted_masterbox[0]:adjusted_masterbox[2]]
            
            target_width = unfilled_dimensions[0][3] if unfilled_dimensions is not None else 1080
            target_height = unfilled_dimensions[0][4] if unfilled_dimensions is not None else 1920
            cropped_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            cv2.rectangle(debug_frame, (adjusted_masterbox[0], adjusted_masterbox[1]), 
                        (adjusted_masterbox[2], adjusted_masterbox[3]), (0,255,0), 6)
            cv2.putText(debug_frame, "ROI", (adjusted_masterbox[0], adjusted_masterbox[1]+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                        
            return debug_frame, cropped_frame
            
        elif len(object_boxes) > 1:
            frame_center_x = frame.shape[1] // 2
            sorted_boxes = sorted(object_boxes, 
                                key=lambda box: ((box[2]-box[0])*(box[3]-box[1]), 
                                            -abs(((box[2]+box[0])//2) - frame_center_x)),
                                reverse=True)
            
            object_box = sorted_boxes[0]
            object_center_x = int((object_box[0] + object_box[2])/2)
            
            master_width = masterbox[2] - masterbox[0]
            half_width = master_width // 2
            new_x1 = object_center_x - half_width
            new_x2 = object_center_x + half_width
            
            if new_x1 < 0:
                new_x1 = 0
                new_x2 = master_width
            elif new_x2 > frame.shape[1]:
                new_x2 = frame.shape[1]
                new_x1 = new_x2 - master_width
                
            if self.prev_masterbox is not None:
                smoothing_factor = 0.8
                new_x1 = int(self.prev_masterbox[0] * smoothing_factor + new_x1 * (1 - smoothing_factor))
                new_x2 = int(self.prev_masterbox[2] * smoothing_factor + new_x2 * (1 - smoothing_factor))
                
            adjusted_masterbox = [new_x1, masterbox[1], new_x2, masterbox[3]]
            self.prev_masterbox = adjusted_masterbox
            
            cropped_frame = frame[adjusted_masterbox[1]:adjusted_masterbox[3], 
                                adjusted_masterbox[0]:adjusted_masterbox[2]]
            
            target_width = unfilled_dimensions[0][3] if unfilled_dimensions is not None else 1080
            target_height = unfilled_dimensions[0][4] if unfilled_dimensions is not None else 1920
            cropped_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            cv2.rectangle(debug_frame, (adjusted_masterbox[0], adjusted_masterbox[1]), 
                        (adjusted_masterbox[2], adjusted_masterbox[3]), (0,255,0), 6)
            cv2.putText(debug_frame, "ROI", (adjusted_masterbox[0], adjusted_masterbox[1]+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                        
            return debug_frame, cropped_frame
            
        else:
            if self.prev_masterbox is not None:
                cropped_frame = frame[self.prev_masterbox[1]:self.prev_masterbox[3],
                                    self.prev_masterbox[0]:self.prev_masterbox[2]]
                cv2.rectangle(debug_frame, (self.prev_masterbox[0], self.prev_masterbox[1]),
                            (self.prev_masterbox[2], self.prev_masterbox[3]), (0,255,0), 6)
                cv2.putText(debug_frame, "ROI", (self.prev_masterbox[0], self.prev_masterbox[1]+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                            
                return debug_frame, cropped_frame
            
            cropped_frame = frame[masterbox[1]:masterbox[3], masterbox[0]:masterbox[2]]
            target_width = unfilled_dimensions[0][3] if unfilled_dimensions is not None else 1080
            target_height = unfilled_dimensions[0][4] if unfilled_dimensions is not None else 1920
            cropped_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(debug_frame, (masterbox[0], masterbox[1]),
                        (masterbox[2], masterbox[3]), (0,255,0), 6) 
            cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
            return debug_frame, cropped_frame

    def get_region_of_interest(self, debug_frame, frame, person_boxes, masterbox, similarity_index, unfilled_dimensions=None):
        """
        Get region of interest based on person detection
        """
        if not person_boxes or len(person_boxes) == 0:
            cropped_frame = frame[masterbox[1]:masterbox[3], masterbox[0]:masterbox[2]]
            target_width = unfilled_dimensions[0][3] if unfilled_dimensions is not None else 1080
            target_height = unfilled_dimensions[0][4] if unfilled_dimensions is not None else 1920
            cropped_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(debug_frame, (masterbox[0], masterbox[1]), (masterbox[2], masterbox[3]), (0,255,0), 6)
            cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)            
            return debug_frame, cropped_frame

        if len(person_boxes) == 1:
            if int(similarity_index) == 0:
                debug_frame, adjusted_master_box = self.center_master_box_on_face(debug_frame, frame, person_boxes, masterbox)
            else:
                adjusted_master_box = masterbox
                
            if self.previous_master is not None:
                smoothing_factor = 0.95
                smooth_x1 = int(self.previous_master[0] * smoothing_factor + adjusted_master_box[0] * (1 - smoothing_factor))
                smooth_x2 = int(self.previous_master[2] * smoothing_factor + adjusted_master_box[2] * (1 - smoothing_factor))
                adjusted_master_box = [smooth_x1, adjusted_master_box[1], smooth_x2, adjusted_master_box[3]]
                
            master_x1 = adjusted_master_box[0]
            master_x2 = adjusted_master_box[2]
            masterbox = adjusted_master_box
            
            masterbox = [master_x1, masterbox[1], master_x2, masterbox[3]]
            cropped_frame = frame[masterbox[1]:masterbox[3], master_x1:master_x2]
            target_width = unfilled_dimensions[0][3] if unfilled_dimensions is not None else 1080
            target_height = unfilled_dimensions[0][4] if unfilled_dimensions is not None else 1920
            cropped_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(debug_frame, (master_x1, masterbox[1]), (master_x2, masterbox[3]), (0,255,0), 6)
            cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)            
            self.previous_master = masterbox
            return debug_frame, cropped_frame

        frame_height, frame_width = debug_frame.shape[:2]
        frame_center_x = frame_width // 2
        
        sorted_boxes = sorted(person_boxes, 
                            key=lambda box: ((box[2]-box[0])*(box[3]-box[1]), 
                                        -abs(((box[2]+box[0])//2) - frame_center_x)),
                            reverse=True)[:4]
        
        if not sorted_boxes:
            cropped_frame = frame[masterbox[1]:masterbox[3], masterbox[0]:masterbox[2]]
            target_width = unfilled_dimensions[0][3] if unfilled_dimensions is not None else 1080
            target_height = unfilled_dimensions[0][4] if unfilled_dimensions is not None else 1920
            cropped_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(debug_frame, (masterbox[0], masterbox[1]), (masterbox[2], masterbox[3]), (0,255,0), 6)
            cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)            
            return debug_frame, cropped_frame
            
        x_min = min(box[0] for box in sorted_boxes)
        x_max = max(box[2] for box in sorted_boxes)
        y_min, y_max = masterbox[1], masterbox[3]
        
        debug_frame_with_boxes = debug_frame.copy()
        for box in sorted_boxes:
            area = (box[2]-box[0])*(box[3]-box[1])
            cv2.rectangle(debug_frame_with_boxes, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
            cv2.putText(debug_frame_with_boxes, f"Person: {box[4]:.2f}, Area: {area}",
                        (box[0], box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
                        
        cv2.rectangle(debug_frame_with_boxes, (x_min, y_min), (x_max, y_max), (0,255,255), 2)
        cv2.putText(debug_frame_with_boxes, "Combined BB", (x_min, y_min+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        master_width = masterbox[2] - masterbox[0]
        master_x1 = max(0, frame_center_x - (master_width // 2))
        master_x2 = min(frame_width, master_x1 + master_width)
        
        if master_x2 == frame_width:
            master_x1 = master_x2 - master_width
            
        max_coverage = 0
        for box in sorted_boxes:
            box_width = box[2] - box[0]
            overlap_width = min(master_x2, box[2]) - max(master_x1, box[0])
            coverage = overlap_width / box_width if box_width > 0 else 0
            max_coverage = max(max_coverage, coverage)
        
        if max_coverage < 0.8:
            center_box = min(sorted_boxes[:2], 
                            key=lambda box: abs(((box[2]+box[0])//2) - frame_center_x))
            
            center_box_x = (center_box[0] + center_box[2]) // 2
            master_x1 = max(0, center_box_x - (master_width // 2))
            master_x2 = min(frame_width, master_x1 + master_width)
            
            if master_x2 == frame_width:
                master_x1 = master_x2 - master_width

        if self.previous_master is not None:
            smoothing_factor = 0.9
            master_x1 = int(self.previous_master[0] * smoothing_factor + master_x1 * (1 - smoothing_factor))
            master_x2 = int(self.previous_master[2] * smoothing_factor + master_x2 * (1 - smoothing_factor))
            
        if self.ema_master_x1 is None:
            self.ema_master_x1 = master_x1
            self.ema_master_x2 = master_x2
        else:
            self.ema_master_x1 = int(self.ema_alpha * master_x1 + (1-self.ema_alpha) * self.ema_master_x1)
            self.ema_master_x2 = int(self.ema_alpha * master_x2 + (1-self.ema_alpha) * self.ema_master_x2)
            master_x1 = self.ema_master_x1
            master_x2 = self.ema_master_x2
                
        for box in sorted_boxes:
            if box[0] >= master_x1 and box[2] <= master_x2:
                cv2.rectangle(debug_frame_with_boxes, (box[0], box[1]), (box[2], box[3]), (42,42,165), 2)
                cv2.putText(debug_frame_with_boxes, "In Master Box", (box[0], box[3]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (42,42,165), 2)
        
        cv2.rectangle(debug_frame_with_boxes, (master_x1, masterbox[1]), (master_x2, masterbox[3]), (0,255,0), 6)
        cv2.putText(debug_frame, "ROI", (masterbox[0], masterbox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4) 

        self.previous_master = (master_x1, masterbox[1], master_x2, masterbox[3])
        cropped_frame = frame[masterbox[1]:masterbox[3], master_x1:master_x2]
        target_width = unfilled_dimensions[0][3] if unfilled_dimensions is not None else 1080
        target_height = unfilled_dimensions[0][4] if unfilled_dimensions is not None else 1920
        cropped_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return debug_frame_with_boxes, cropped_frame
