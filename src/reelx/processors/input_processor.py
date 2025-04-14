from reelx.config.configuration import Config

class InputProcessor:
    def __init__(self, input_file):
        self.input_file = input_file
    
    def get_current_input_type(self):
        """
        Determine the input type based on the file path/URL
        Returns the appropriate input type from Config.INPUT_PARAMS
        """
        if self.input_file.startswith(('udp://', 'rtsp://')):
            return Config.INPUT_PARAMS["LIVE_INPUT_TYPE"]   
        return Config.INPUT_PARAMS["VIDEO_INPUT_TYPE"]
