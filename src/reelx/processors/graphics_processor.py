from reelx.config.configuration import Config
import json
import cv2
import numpy as np

class GraphicsProcessor:
    def __init__(self):
        pass

    def draw_text(self, frame, unfilled_dimensions, frame_time):
        reframed = frame.copy()
        
        # Get highlights info once upfront
        title = ''
        highlights = []
        if Config.OUTPUT_VIDEO_PARAMS["HIGHLIGHTS"] is not None:
            highlights_info = Config.OUTPUT_VIDEO_PARAMS["HIGHLIGHTS"]
            json_highlights = json.loads(highlights_info)
            title = json_highlights.get('title', '')
            highlights = json_highlights.get('highlights', [])
            
        # Get dominant color more efficiently
        filled_pixels = frame.reshape(-1, 3)
        mask = ~np.all(filled_pixels == [0,0,0], axis=1)
        dominant_color = np.array([128, 128, 128]) # Default gray
        if np.any(mask):
            pixels = filled_pixels[mask]
            pixel_colors = pixels.dot(np.array([1, 256, 65536]))
            dominant_idx = np.bincount(pixel_colors).argmax()
            dominant_color = np.array([
                dominant_idx & 255,
                (dominant_idx >> 8) & 255, 
                (dominant_idx >> 16) & 255
            ])
            
        font = cv2.FONT_HERSHEY_DUPLEX
        
        for i, dimension in enumerate(unfilled_dimensions):
            x, y, w, h = dimension
            
            if x < 0 or y < 0 or x+w > reframed.shape[1] or y+h > reframed.shape[0]:
                print(f"Warning: Rectangle {dimension} extends outside frame boundaries")
                continue
                
            reframed[y:y+h, x:x+w] = dominant_color
            
            if len(unfilled_dimensions) == 2 and i == 0:
                font_scale = w / 400
                thickness = max(1, int(font_scale * 2))
                line_height = int(h/8)

                words = title.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0][0]
                    
                    if text_size <= w - 20:
                        current_line.append(word)
                    elif current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
                        current_line = []
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                total_height = line_height * len(lines)
                start_y = y + (h - total_height)//2 + line_height
                
                for line in lines:
                    line_width = cv2.getTextSize(line, font, font_scale, thickness)[0][0]
                    text_x = x + (w - line_width) // 2
                    text_y = start_y
                    
                    cv2.putText(reframed, line, (text_x+2, text_y+2), font, font_scale, (0,0,0), thickness+1)
                    cv2.putText(reframed, line, (text_x, text_y), font, font_scale, (255,255,255), thickness)
                    start_y += line_height
                    
            highlight_dimension = unfilled_dimensions[1] if len(unfilled_dimensions) == 2 else dimension
            if dimension == highlight_dimension and highlights:
                font_scale = w / 500
                thickness = max(1, int(font_scale * 2))
                line_height = int(h/10)
                
                active_highlights = [h for h in highlights 
                                   if frame_time >= h.get('start_time', 0) and 
                                   frame_time <= h.get('end_time', float('inf'))]
                
                for highlight in active_highlights:
                    headline = highlight.get('headline', '')
                    words = headline.split()
                    lines = []
                    current_line = []
                    
                    for word in words:
                        test_line = ' '.join(current_line + [word])
                        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0][0]
                        
                        if text_size <= w - 20:
                            current_line.append(word)
                        elif current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            lines.append(word)
                            current_line = []
                    
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    total_height = line_height * len(lines)
                    start_y = y + (h - total_height)//2 + line_height
                    
                    for line in lines:
                        line_width = cv2.getTextSize(line, font, font_scale, thickness)[0][0]
                        text_x = x + (w - line_width) // 2
                        cv2.putText(reframed, line, (text_x+2, start_y+2), font, font_scale, (0,0,0), thickness+1)
                        cv2.putText(reframed, line, (text_x, start_y), font, font_scale, (255,255,255), thickness)
                        start_y += line_height
                            
        return reframed

    def draw_image(self, frame, unfilled_dimensions, image_paths):
        reframed = frame
        
        for dimension, img_path in zip(unfilled_dimensions, image_paths):
            x, y, w, h = dimension
            
            if x < 0 or y < 0 or x+w > reframed.shape[1] or y+h > reframed.shape[0]:
                print(f"Warning: Rectangle {dimension} extends outside frame boundaries")
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
                
            img_h, img_w = img.shape[:2]
            target_aspect = w / h
            img_aspect = img_w / img_h
            
            if img_aspect > target_aspect:
                new_w = int(img_h * target_aspect)
                crop_x = (img_w - new_w) // 2
                crop_y = 0
                img = img[:, crop_x:crop_x+new_w]
            else:
                new_h = int(img_w / target_aspect)
                crop_x = 0
                crop_y = (img_h - new_h) // 2
                img = img[crop_y:crop_y+new_h, :]
                
            resized_img = cv2.resize(img, (w, h))
            
            if resized_img.shape[2] == 4:
                alpha = resized_img[:, :, 3] / 255.0
                bgr = resized_img[:, :, :3]
                
                for c in range(3):
                    reframed[y:y+h, x:x+w, c] = (1-alpha) * reframed[y:y+h, x:x+w, c] + alpha * bgr[:, :, c]
            else:
                reframed[y:y+h, x:x+w] = resized_img

        return reframed

    def draw_video(self, frame, unfilled_dimensions, video_path, frame_count):
        reframed = frame.copy()
        
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return reframed
            
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        success, video_frame = video.read()
        if not success:
            print(f"Warning: Could not read frame {frame_count} from video")
            video.release()
            return reframed
        
        for dimension in unfilled_dimensions:
            x, y, w, h = dimension
            
            if x < 0 or y < 0 or x+w > reframed.shape[1] or y+h > reframed.shape[0]:
                print(f"Warning: Rectangle {dimension} extends outside frame boundaries")
                continue
                
            vid_h, vid_w = video_frame.shape[:2]
            target_aspect = w / h
            vid_aspect = vid_w / vid_h
            
            if vid_aspect > target_aspect:
                new_w = int(vid_h * target_aspect)
                crop_x = (vid_w - new_w) // 2
                crop_y = 0
                cropped_frame = video_frame[:, crop_x:crop_x+new_w]
            else:
                new_h = int(vid_w / target_aspect)
                crop_x = 0 
                crop_y = (vid_h - new_h) // 2
                cropped_frame = video_frame[crop_y:crop_y+new_h, :]
                
            resized_frame = cv2.resize(cropped_frame, (w, h))
            reframed[y:y+h, x:x+w] = resized_frame
            
        video.release()
        return reframed
