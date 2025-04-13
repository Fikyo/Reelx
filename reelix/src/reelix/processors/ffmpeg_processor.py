
import ffmpeg
import os
import uuid

class FFMPEGProcessor:
    def __init__(self, input_path=None, output_path=None):
        """Initialize FFMPEG processor with input and output paths"""
        self.input_path = input_path
        self.output_path = output_path
        self.ffmpeg_cmd = 'ffmpeg'
        self.options = []

    def set_input(self, input_path):
        """Set input file path"""
        self.input_path = input_path
        return self

    def set_output(self, output_path):
        """Set output file path"""
        self.output_path = output_path
        return self

    def create_temp_copy(self):
        """Create a temporary copy of the output file with unique identifier"""
        # Generate unique identifier
        uid = str(uuid.uuid4())
        
        # Get file extension from output path
        _, ext = os.path.splitext(self.output_path)
        
        # Create temp filename with uid
        temp_path = f"{self.output_path}_{uid}{ext}"
        
        try:
            # Copy output file to temp file
            stream = ffmpeg.input(self.output_path)
            (
                ffmpeg
                .output(stream, temp_path, c='copy')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return temp_path
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error creating temp copy: {e.stderr.decode()}")
            raise
        except Exception as e:
            print(f"Error creating temp copy: {str(e)}")
            raise

    def generate_final_vertical_video(self):
        try:
            # Extract audio and probe streams in one call
            probe = ffmpeg.probe(self.input_path)
            stream = ffmpeg.input(self.input_path)
            audio = stream.audio
            
            temp_output_path = self.create_temp_copy()
            # Check for audio stream more efficiently
            has_audio = any(s['codec_type'] == 'audio' for s in probe['streams'])
            
            # Input video stream
            video = ffmpeg.input(temp_output_path)
            
            if has_audio:
                # Merge video with audio
                (
                    ffmpeg
                    .output(video, audio, self.output_path,
                        vcodec='libx264',
                        acodec='aac')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            else:
                # Copy video only
                (
                    ffmpeg
                    .output(video, self.output_path,
                        vcodec='libx264')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
            # Clean up temp file
            os.unlink(temp_output_path)
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            raise
        except Exception as e:
            print(f"Error: {str(e)}")