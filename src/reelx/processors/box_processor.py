import cv2
import numpy as np

class BoxProcessor:
    @staticmethod
    def filter_person_boxes(person_boxes, frame_shape, min_size=50, boundary_threshold=20):
        """
        Filters out person boxes that are too small or too close to frame boundaries
        """
        height, width = frame_shape[:2]
        boundary_width = int(0.15 * width)
        boundary_height = int(0.15 * height)
        
        boxes_to_remove = []
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            box_area = (x2 - x1) * (y2 - y1)
            
            if box_area < min_size:
                boxes_to_remove.append(i)
                continue
            
            boundary_area = 0
            
            if x1 < boundary_width:
                boundary_area += (min(boundary_width, x2) - x1) * (y2 - y1)
            if x2 > (width - boundary_width):
                boundary_area += (x2 - max(width - boundary_width, x1)) * (y2 - y1)
            if y1 < boundary_height:
                boundary_area += (min(boundary_height, y2) - y1) * (x2 - x1)
            if y2 > (height - boundary_height):
                boundary_area += (y2 - max(height - boundary_height, y1)) * (x2 - x1)
                
            if boundary_area / box_area >= 0.8:
                boxes_to_remove.append(i)
                continue

            for j, other_box in enumerate(person_boxes):
                if i != j:
                    ox1, oy1, ox2, oy2 = other_box[0], other_box[1], other_box[2], other_box[3]
                    x_left = max(x1, ox1)
                    y_top = max(y1, oy1)
                    x_right = min(x2, ox2)
                    y_bottom = min(y2, oy2)
                    
                    if x_right > x_left and y_bottom > y_top:
                        intersection = (x_right - x_left) * (y_bottom - y_top)
                        if intersection / box_area >= 0.9:
                            boxes_to_remove.append(i)
                            break
                
        for i in sorted(boxes_to_remove, reverse=True):
            person_boxes.pop(i)
            
        return person_boxes

    @staticmethod
    def smooth_bounding_box(prev_bbox, current_bbox, smoothing_factor=None):
        """
        Apply adaptive motion-based smoothing to bounding box coordinates for stable tracking
        """
        if prev_bbox is None:
            return current_bbox
            
        # Calculate center points and dimensions
        prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
        prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
        prev_width = prev_bbox[2] - prev_bbox[0]
        prev_height = prev_bbox[3] - prev_bbox[1]
        
        curr_center_x = (current_bbox[0] + current_bbox[2]) / 2
        curr_center_y = (current_bbox[1] + current_bbox[3]) / 2
        curr_width = current_bbox[2] - current_bbox[0]
        curr_height = current_bbox[3] - current_bbox[1]
        
        # Calculate movement and size change
        movement = ((curr_center_x - prev_center_x) ** 2 + (curr_center_y - prev_center_y) ** 2) ** 0.5
        size_change = abs(curr_width * curr_height - prev_width * prev_height) / (prev_width * prev_height)
        
        # Determine smoothing factor based on movement
        if smoothing_factor is None:
            smoothing_factor = 0.98 if movement > prev_width * 0.5 or size_change > 0.3 else 0.8
        
        # Apply EMA smoothing with momentum for large movements
        momentum = 0.9 if movement > prev_width * 0.5 else 0.0
        if momentum > 0:
            # Add momentum to the movement direction
            dx = curr_center_x - prev_center_x
            dy = curr_center_y - prev_center_y
            curr_center_x += dx * momentum
            curr_center_y += dy * momentum
            
        # Smooth centers and dimensions separately
        smooth_center_x = prev_center_x * smoothing_factor + curr_center_x * (1 - smoothing_factor)
        smooth_center_y = prev_center_y * smoothing_factor + curr_center_y * (1 - smoothing_factor)
        smooth_width = prev_width * smoothing_factor + curr_width * (1 - smoothing_factor)
        smooth_height = prev_height * smoothing_factor + curr_height * (1 - smoothing_factor)
        
        # Convert back to box coordinates
        smoothed_x = int(smooth_center_x - smooth_width / 2)
        smoothed_y = int(smooth_center_y - smooth_height / 2)
        smoothed_w = int(smooth_center_x + smooth_width / 2)
        smoothed_h = int(smooth_center_y + smooth_height / 2)
        
        return (smoothed_x, smoothed_y, smoothed_w, smoothed_h)

    @staticmethod
    def resolve_box_overlap(person_boxes, prev_crops=None):
        """
        Resolve and adjust overlapping bounding boxes while maintaining tracking stability
        """
        if person_boxes[0][2] <= person_boxes[1][0]:
            return person_boxes

        overlap = person_boxes[0][2] - person_boxes[1][0]
        overlap_percentage = overlap / (person_boxes[0][2] - person_boxes[0][0])
        
        if overlap_percentage > 0.8:
            return None
            
        overlap_adjust = overlap // 2
        smoothing_factor = 0.7
        
        if prev_crops is not None:
            person_boxes[0] = (person_boxes[0][0], person_boxes[0][1],
                            int(prev_crops[0][2] * smoothing_factor + 
                                (person_boxes[0][2] - overlap_adjust) * (1 - smoothing_factor)),
                            person_boxes[0][3], person_boxes[0][4])
            person_boxes[1] = (int(prev_crops[1][0] * smoothing_factor + 
                                (person_boxes[1][0] + overlap_adjust) * (1 - smoothing_factor)),
                            person_boxes[1][1], person_boxes[1][2], person_boxes[1][3], person_boxes[1][4])
        else:
            person_boxes[0] = (person_boxes[0][0], person_boxes[0][1],
                            person_boxes[0][2] - overlap_adjust, person_boxes[0][3], person_boxes[0][4])
            person_boxes[1] = (person_boxes[1][0] + overlap_adjust, person_boxes[1][1],
                            person_boxes[1][2], person_boxes[1][3], person_boxes[1][4])
                            
        return person_boxes

    @staticmethod
    def smooth_box_position(x1, y1, x2, y2, prev_crop):
        """
        Apply position-based smoothing to box coordinates for continuous tracking
        """
        prev_x1, prev_y1, prev_x2, prev_y2, _ = prev_crop
        
        width_deviation = abs(x2 - x1 - (prev_x2 - prev_x1)) / (prev_x2 - prev_x1)
        height_deviation = abs(y2 - y1 - (prev_y2 - prev_y1)) / (prev_y2 - prev_y1)
        
        smoothing_factor = 0.95 if width_deviation > 0.15 or height_deviation > 0.15 else 0.8
        
        return (int(prev_x1 * smoothing_factor + x1 * (1 - smoothing_factor)),
                int(prev_y1 * smoothing_factor + y1 * (1 - smoothing_factor)),
                int(prev_x2 * smoothing_factor + x2 * (1 - smoothing_factor)),
                int(prev_y2 * smoothing_factor + y2 * (1 - smoothing_factor)))
