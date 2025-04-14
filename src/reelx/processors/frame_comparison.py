import cv2
import numpy as np

class FrameComparison:
    @staticmethod
    def compare_frames(current_frame, previous_frame):
        """
        Compare current and previous frames to detect changes
        Returns a similarity score between 0-1 where 1 means frames are identical
        """
        if current_frame is None or previous_frame is None:
            return 0
            
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        if curr_gray.shape != prev_gray.shape:
            prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))
        
        err = np.sum((curr_gray.astype("float") - prev_gray.astype("float")) ** 2)
        err /= float(curr_gray.shape[0] * curr_gray.shape[1])
        
        similarity_index = 1 - min(err / 255.0, 1.0)
        return similarity_index

    @staticmethod
    def calculate_motion_score(current_frame, previous_frame):
        """
        Calculate motion score between consecutive frames using optical flow
        Returns a motion score between 0-1 where 0 means no motion and 1 means maximum motion
        """
        if current_frame is None or previous_frame is None:
            return 0
            
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        if curr_gray.shape != prev_gray.shape:
            prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, 
                                        flags=0)
                                        
        magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        motion_score = np.mean(magnitude) / 100.0
        motion_score = min(motion_score, 1.0)
        
        return motion_score

    @staticmethod
    def get_smoothing_threshold(motion_score):
        """
        Calculate smoothing threshold based on motion score
        Returns a threshold value between 0.3-0.9
        """
        min_threshold = 0.3
        max_threshold = 0.9
        threshold = max_threshold - (motion_score * (max_threshold - min_threshold))
        threshold = max(min_threshold, min(threshold, max_threshold))
        return threshold
