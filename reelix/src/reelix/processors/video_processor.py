import cv2
import traceback
import time
import subprocess
import numpy as np
import os


from reelix.config.configuration import Config
from reelix.processors.frame_processor import FrameProcessor
from reelix.processors.detection_processor import DetectionProcessor
from reelix.processors.ffmpeg_processor import FFMPEGProcessor
from reelix.utils.viualisation import Visualizer
from reelix.utils.printer import Printer



class VideoProcessor:
    def __init__(self, source_path, output_path, mode="vertical"):
        self.source_path = source_path
        self.output_path = output_path
        self.mode = mode
        self.vwriter = None
        self.frame_processor = None
        self.visualizer = Visualizer(Config.OUTPUT_VIDEO_PARAMS["PREVIEW_DEBUG_PLAYER"], Config.OUTPUT_VIDEO_PARAMS["PREVIEW_VERTICAL_VIDEO"])
        self.is_live_output = self._check_if_live_output()
        self.ffmpeg_process = None
    
    def _check_if_live_output(self):
        """Check if the output path is a live streaming endpoint (UDP, RTP, SRT)"""
        if self.output_path is None:
            return False
        
        # Check if output path starts with UDP, RTP, or SRT (case insensitive)
        live_protocols = ['udp:', 'rtp:', 'srt:']
        return any(self.output_path.lower().startswith(protocol) for protocol in live_protocols)
    
    def process_video(self):
        cap = cv2.VideoCapture(self.source_path)    
        if not cap.isOpened():
            raise RuntimeError("Error opening video file")
        Printer.print_frame_properties(cap)
        self.visualizer.set_debug_props("Source WxH", f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")    
        target_height, target_width = FrameProcessor.get_target_dimensions(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print(f"Target dimensions - Width: {target_width}, Height: {target_height}")
        self.visualizer.set_debug_props("Target WxH", f"{target_width}x{target_height}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.visualizer.set_debug_props("Source FPS", f"{fps:.2f}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_processor = FrameProcessor(target_height, target_width, fps)
        
        if self.output_path is not None:
            if self.is_live_output:
                print(f"Setting up live output stream to {self.output_path}")
                self._setup_ffmpeg_live_stream(fps, target_width, target_height)
            else:
                self.vwriter = self._setup_writer(fps, target_height, target_width)
                
        try:
            frame_count = 0
            paused = False
            while cap.isOpened():
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video.")
                        break
                
                frame_time = round(frame_count / fps, 2)  # Calculate frame time in seconds
                self.visualizer.set_debug_props("Current Frame", f"{frame_count}/{total_frames})")
                self.visualizer.set_debug_props("Current Time", f"{frame_time}s")
                if Config.OUTPUT_VIDEO_PARAMS["CLASS_ID"] is None:
                    debug_frame, processed_frame = self.frame_processor.process_frame(frame, self.mode, frame_time, frame_count)
                else:
                    print("Processing for the class id : ", Config.OUTPUT_VIDEO_PARAMS["CLASS_ID"])
                    debug_frame, processed_frame = self.frame_processor.process_frame(frame, self.mode, frame_time, frame_count, Config.OUTPUT_VIDEO_PARAMS["CLASS_ID"])
                paused, frame_count = self.visualizer.show_frame(debug_frame, processed_frame, fps, total_frames, frame_count, paused)
                
                if not paused and self.output_path is not None:
                    if self.is_live_output:
                        self._write_frame_to_ffmpeg(processed_frame)
                    elif self.vwriter is not None:
                        self.vwriter.write(processed_frame)
        except Exception as e:
            print(f"An error occurred during processing: {str(e)}")
            traceback.print_exc()
            
        finally:
            cap.release()
            if self.output_path is not None:
                if self.is_live_output:
                    self._close_ffmpeg_stream()
                    print("Live stream processing completed.")
                else:
                    self._close_writer()
                    ffmpeg_proessor = FFMPEGProcessor(self.source_path, self.output_path)
                    ffmpeg_proessor.generate_final_vertical_video()
                    print("Video processing completed.")

    def _setup_ffmpeg_live_stream(self, fps, width, height):
        """Set up FFmpeg process for live streaming"""
        try:
            # Create FFmpeg command for streaming
            command = [
                'ffmpeg',
                '-y',  # Overwrite output files
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',  # OpenCV uses BGR format
                '-s', f'{width}x{height}',  # Size of one frame
                '-r', str(fps),  # Frames per second
                '-i', '-',  # Input from pipe
                '-c:v', 'libx264',  # H.264 codec
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-preset', 'ultrafast',  # Fast encoding for real-time
                '-tune', 'zerolatency',  # Minimize latency
                '-f', 'mpegts' if self.output_path.lower().startswith(('udp:', 'rtp:')) else 'mpegts',  # Format based on protocol
                self.output_path  # Output URL
            ]
            
            print(f"Starting FFmpeg with command: {' '.join(command)}")
            
            # Start FFmpeg process with pipe for input
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            print("FFmpeg process started for live streaming")
            return True
            
        except Exception as e:
            print(f"Error setting up FFmpeg live stream: {str(e)}")
            return False
    
    def _write_frame_to_ffmpeg(self, frame):
        """Write a frame to the FFmpeg process for live streaming"""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:  # Check if process is still running
            try:
                # Convert frame to bytes and write to FFmpeg's stdin
                self.ffmpeg_process.stdin.write(frame.tobytes())
            except Exception as e:
                print(f"Error writing frame to FFmpeg: {str(e)}")
                self._close_ffmpeg_stream()
                return False
            return True
        return False
    
    def _close_ffmpeg_stream(self):
        """Close the FFmpeg process if it exists"""
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
            try:
                # Close stdin pipe to signal end of input
                if self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.close()
                
                # Wait for process to finish
                self.ffmpeg_process.wait(timeout=5)
                
                # Check for any errors
                if self.ffmpeg_process.returncode != 0:
                    stderr = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                    print(f"FFmpeg process ended with error code {self.ffmpeg_process.returncode}: {stderr}")
                
                self.ffmpeg_process = None
                print("FFmpeg live stream closed")
            except Exception as e:
                print(f"Error closing FFmpeg stream: {str(e)}")
                # Force terminate if needed
                try:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process = None
                except:
                    pass
    
    def _setup_writer(self, fps, target_height, target_width):
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_width, target_height))
        return out

    def _close_writer(self):
        """Closes the video writer if it exists"""
        if hasattr(self, 'vwriter') and self.vwriter is not None:
            self.vwriter.release()
