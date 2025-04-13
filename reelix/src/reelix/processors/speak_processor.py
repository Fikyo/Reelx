
import cv2
import numpy as np
import dlib
from imutils import face_utils
from collections import deque

# Initialize detector and predictor once, outside the function
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model_static/shape_predictor_68_face_landmarks.dat')

# Queue to store recent mouth ratios and face positions for temporal analysis
mouth_ratio_queue = deque(maxlen=10)
face_pos_queue = deque(maxlen=10)

def get_mouth_ratio(points):
    # Calculate both vertical distances for better accuracy
    vert_dist1 = np.linalg.norm(points[2] - points[10])  # vertical distance
    vert_dist2 = np.linalg.norm(points[3] - points[9])   # second vertical distance
    vert_dist = (vert_dist1 + vert_dist2) / 2  # average vertical distance
    
    # Calculate multiple horizontal distances
    horiz_dist1 = np.linalg.norm(points[0] - points[6])  # outer horizontal distance
    horiz_dist2 = np.linalg.norm(points[2] - points[4])  # inner horizontal distance
    horiz_dist = (horiz_dist1 + horiz_dist2) / 2  # average horizontal distance
    
    return vert_dist / horiz_dist

def detect_lip_movement(curr_ratio):
    # Add current ratio to queue
    mouth_ratio_queue.append(curr_ratio)
    
    if len(mouth_ratio_queue) < 5:
        return False
        
    # Calculate statistics from recent ratios
    mean_ratio = np.mean(mouth_ratio_queue)
    std_ratio = np.std(mouth_ratio_queue)
    
    # More strict thresholds for movement detection
    movement_detected = std_ratio > 0.025 and abs(curr_ratio - mean_ratio) > 0.015
    return movement_detected

def detect_head_movement(face_center):
    # Add current face center to queue
    face_pos_queue.append(face_center)
    
    if len(face_pos_queue) < 5:
        return False
        
    # Calculate movement from face positions
    positions = np.array(face_pos_queue)
    movement = np.std(positions, axis=0)
    
    # Lower threshold for head movement detection
    return np.mean(movement) > 4

def is_speaking(frame, base_mouth_threshold=0.35):
    """
    Detect if a person is speaking using multiple features:
    - Mouth openness ratio
    - Temporal lip movement analysis
    - Face size adaptive thresholding
    - Head movement detection
    
    Args:
        frame: Current video frame
        base_mouth_threshold: Base threshold for mouth openness detection
        
    Returns:
        bool: True if speaking detected, False otherwise
    """
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_frame = frame.copy()
    speaking = False
    
    faces = detector(gray, 0)
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        mouth_points = shape[48:68]
        curr_ratio = get_mouth_ratio(mouth_points)
        
        # Draw mouth landmarks
        for (x, y) in shape[48:68]:
            cv2.circle(display_frame, (x, y), 1, (0, 255, 255), -1)
            
        (x, y, w, h) = face_utils.rect_to_bb(face)
        
        # Calculate face center for head movement detection
        face_center = (x + w//2, y + h//2)
        head_moving = detect_head_movement(face_center)
        
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        face_ratio = face_area / frame_area
        
        # Adjust threshold based on face size with smoother scaling
        if face_ratio < 0.1:  # Small face
            adjusted_threshold = base_mouth_threshold * 0.85  # Slightly increased
        elif face_ratio > 0.3:  # Large face
            adjusted_threshold = base_mouth_threshold * 1.15  # Slightly decreased
        else:  # Medium face - linear interpolation
            scale = 0.85 + (face_ratio - 0.1) * 1.5  # Adjusted scaling
            adjusted_threshold = base_mouth_threshold * scale
            
        # Increased margin for more strict detection
        threshold_margin = 0.025
        
        # Combine mouth openness and movement detection, exclude if head is moving
        mouth_open = curr_ratio > (adjusted_threshold - threshold_margin)
        lip_movement = detect_lip_movement(curr_ratio)
        
        # Additional check for rapid mouth movements
        rapid_movement = len(mouth_ratio_queue) >= 5 and np.max(mouth_ratio_queue) - np.min(mouth_ratio_queue) > 0.03
        
        speaking = mouth_open and lip_movement and not head_moving and rapid_movement
            
        if speaking:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Speaking ({curr_ratio:.2f}/{adjusted_threshold:.2f})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            status = "Head Moving" if head_moving else "Not Speaking"
            cv2.putText(display_frame, f"{status} ({curr_ratio:.2f}/{adjusted_threshold:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    display_frame = cv2.resize(display_frame, (frame.shape[1]*2, frame.shape[0]*2))
    cv2.imshow('Speaking Detection', display_frame)
    cv2.waitKey(1)
    
    return speaking    

#cap = cv2.VideoCapture("../vvproj/sample_src_videos/pallki_smallq.mp4")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    speaking = is_speaking(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
