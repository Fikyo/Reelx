
from abc import ABC, abstractmethod
from reelix.config.configuration import Config
from reelix.utils.device import Device
from ultralytics import YOLO
from pkg_resources import resource_filename
import os
import cv2

class PersonDetector(ABC):
    @abstractmethod
    def detect_persons(self, image):
        pass

class YoloDetector(PersonDetector):
    def __init__(self):
        # Initialize YOLO model
        self.person_model = self._load_yolo_person_model()
        self.pose_model = self._load_yolo_pose_model()
    
    def _load_yolo_person_model(self):
        # Load YOLO model implementation
        cwd = os.getcwd()
        person_model = YOLO(Config.MODEL_PARAMS["PERSON_MODEL_PATH"])    
        device = Device.get_device()
        person_model.to(device)
        person_model.overrides = {"verbose": Config.MODEL_PARAMS["MODEL_VERBOSE"]}
        return person_model
    
    def _load_yolo_pose_model(self):
        # Load YOLO model implementation
        cwd = os.getcwd()
        pose_model = YOLO(Config.MODEL_PARAMS["POSE_MODEL_PATH"])    
        device = Device.get_device()
        pose_model.to(device)
        pose_model.overrides = {"verbose": Config.MODEL_PARAMS["MODEL_VERBOSE"]}
        return pose_model   
    
    def detect_persons(self, frame):
        # Implement YOLO person detection
        results = self.person_model(frame)
        detections = results[0].boxes.cpu().numpy()
        person_boxes = []
        for detection in detections:
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            if confidence > Config.MODEL_PARAMS["OBJECT_CONFIDENCE_THRESHOLD"] and class_id == 0:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                person_boxes.append([x1, y1, x2, y2, confidence])
        return person_boxes, results 
    
    def detect_classid(self, frame, classid):
        # Implement YOLO person detection
        results = self.person_model(frame)
        detections = results[0].boxes.cpu().numpy()
        detect_boxes = []
        for detection in detections:
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            # Print class name from model names dictionary
            class_name = self.person_model.names[class_id]
            print(f"Detected class: {class_name} and ID is : {class_id}")
            if confidence > Config.MODEL_PARAMS["OBJECT_CONFIDENCE_THRESHOLD"] and class_id == classid:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                detect_boxes.append([x1, y1, x2, y2, confidence])
        return detect_boxes, results 
    
    def detect_persons_pose(self, frame):
        # Implement YOLO person detection with pose
        #results = self.pose_model(frame)
        pose_results = self.pose_model(frame)
       
        return pose_results

    def detect_persons_track(self, frame):
        # Implement YOLO person detection with pose
        #results = self.pose_model(frame)
        track_results = self.pose_model.track(frame)
       
        return track_results

class DetectorFactory:
    @staticmethod
    def get_detector(detector_type: str) -> PersonDetector:
        print(f"Detector type: {detector_type}")
        if detector_type.lower() == Config.MODEL_PARAMS["MODEL_TYPE_YOLO"].lower():
            return YoloDetector()
        else:
            raise ValueError("Invalid detector type. Use 'yolo'")

# Face Detection Classes
class FaceDetector(ABC):
    @abstractmethod
    def detect_faces(self, image, master_box):
        pass

class YoloFaceDetector(FaceDetector):
    def __init__(self):
        # Initialize YOLO model for face detection
        self.model = self._load_yolo_face_model()
    
    def _load_yolo_face_model(self):
        # Check if face model file exists
        face_model_path = Config.MODEL_PARAMS["FACE_MODEL_PATH"]
        if not os.path.exists(face_model_path):
            # Download model file if it doesn't exist
            try:
                import requests
                print(f"Downloading face model from {Config.MODEL_PARAMS['FACE_MODEL_URL']}")
                response = requests.get(Config.MODEL_PARAMS["FACE_MODEL_URL"])
                response.raise_for_status()
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(face_model_path), exist_ok=True)
                
                # Save downloaded model
                with open(face_model_path, 'wb') as f:
                    f.write(response.content)
                print("Face model downloaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to download face model: {str(e)}")

        # Load YOLO model implementation for faces
        face_model = YOLO(face_model_path)    
        device = Device.get_device()
        face_model.to(device)
        face_model.overrides = {"verbose": Config.MODEL_PARAMS["MODEL_VERBOSE"]}
        return face_model               

    def detect_faces(self, frame, master_box):
        # Implement YOLO face detection
        px1, py1, px2, py2 = master_box
        results = self.model.predict(source=frame[py1:py2, px1:px2], conf=Config.MODEL_PARAMS["FACE_CONFIDENCE_THRESHOLD"], verbose=False) 
        return results

class FaceDetectorFactory:
    @staticmethod
    def get_detector(detector_type: str) -> FaceDetector:
        print(f"Face Detector type: {detector_type}")
        if detector_type.lower() == Config.MODEL_PARAMS["MODEL_TYPE_YOLO"].lower():
            return YoloFaceDetector()
        else:
            raise ValueError("Invalid detector type. Use 'yolo'")
