import cv2

class Config:
    # Reels Processing Parameters
    VIDEO_PARAMS = {
        "ASPECT_RATIOS": {
            "VERTICAL": 9/16,  # For 9:16 vertical videos
            "SQUARE": 1/1,     # For 1:1 square videos
            "HORIZONTAL": 16/9 # For 16:9 horizontal videos
        },
        "FPS": 30,
        "CODEC": "mp4v",
        "VERTICAL_MODE": "VERTICAL",
        "SQUARE_MODE": "SQUARE",
        "HORIZONTAL_MODE": "HORIZONTAL",
    }

    OUTPUT_VIDEO_PARAMS = {
        "VERTICAL": False,
        "SQUARE": False,
        "NO_LOSS_TILED": False,
        "PERSON_TILED" : False,
        "DEBUG_PLAYER": False,
        "PREVIEW_VERTICAL_VIDEO": False,
        "PREVIEW_DEBUG_PLAYER": False,
        "OUTPUT_FILE": None,
        "FILL_USING_GENAI": False,
    }

    CROP_PLACEMENT_PARAMS = {
        "TOP": "TOP",
        "BOTTOM": "BOTTOM",
        "MIDDLE": "MIDDLE",
        "CURRENT_PLACEMENT": "BOTTOM"
    }
    
    # Model Parameters
    MODEL_PARAMS = {
        "MODEL_TYPE_AWS": "REKOGNITION",
        "MODEL_TYPE_YOLO": "YOLO",
        "OBJECT_CONFIDENCE_THRESHOLD": 0.8,
        "FACE_CONFIDENCE_THRESHOLD": 0.5,
        "PERSON_MODEL_PATH": "./model_static/yolo11m.pt",
        "FACE_MODEL_PATH": "./model_static/yolov11n-face.pt",
        "FACE_MODEL_URL": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
        "MODEL_VERBOSE": False,
        "CURRENT_MODEL_TYPE": "YOLO",
        "VIDEO_MODEL": "amazon.nova-reel-v1:0",
        "POSE_MODEL_PATH": "./model_static/yolo11n-pose.pt"
    }

    # Tracking Parameters
    TRACKING_PARAMS = {
        "SIMILARITY_INDEX_THRESHOLD": 0.75,
        "SMOOTHING_FACTOR": 0.7,
        "LARGE_MOVEMENT_THRESHOLD": 100,
        "LARGE_MOVEMENT_SMOOTHING": 0.95,
        "NORMAL_SMOOTHING": 0.8
    }
    # Cropping Parameters
    CROP_PARAMS = {
        "DEFAULT_PADDING": 0.1,
        "MIN_PERSON_AREA_RATIO": 0.4,
        "MAX_PERSONS": 4,
        "MIN_PERSONS": 2
    }

    INPUT_PARAMS = {
        "LIVE_INPUT_TYPE": "LIVE",
        "VIDEO_INPUT_TYPE": "VOD",
        "CURRENT_INPUT_TYPE": "VOD"
    }

    # Display Parameters
    DISPLAY_PARAMS = {
        "COLORS": {
            "PERSON_BOX": (255, 0, 0),      # Blue
            "MASTER_BOX": (0, 0, 255),    # Red
            "FACE_BOX": (0, 255, 0),         # Green
            "TITLE_TEXT": (148, 0, 211),         # Violet  
            "DEBUG_TEXT": (0, 0, 255)    # Red      
        },
        "LINE_THICKNESS": 2,
        "FONT_SCALE": 0.7,
        "FONT": cv2.FONT_HERSHEY_SIMPLEX
    }

    @classmethod
    def load_config(cls, config_file):
        """Load configuration from JSON file"""
        import json
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            for category, params in config_data.items():
                if hasattr(cls, category):
                    setattr(cls, category, params)

    @classmethod
    def save_config(cls, config_file):
        """Save current configuration to JSON file"""
        import json
        config_data = {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('__') and isinstance(value, dict)
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
            
    @classmethod
    def update_config(cls, category, param, value):
        """Update a specific configuration parameter
        Args:
            category: The parameter category (e.g. 'VIDEO_PARAMS')
            param: The parameter name or nested dict keys as tuple (e.g. ('ASPECT_RATIOS','VERTICAL'))
            value: The new value to set
        """
        if hasattr(cls, category):
            config_dict = getattr(cls, category)
            if isinstance(param, tuple):
                # Handle nested dictionary updates
                current = config_dict
                for key in param[:-1]:
                    current = current[key]
                current[param[-1]] = value
            else:
                # Handle top level updates
                config_dict[param] = value

    @classmethod
    def print_config(cls):
        """Print all configuration parameters in a readable format"""
        for category in [attr for attr in dir(cls) if not attr.startswith('__') and isinstance(getattr(cls, attr), dict)]:
            print(f"\n{category}:")
            config_dict = getattr(cls, category)
            
            def print_nested(d, indent=2):
                for key, value in d.items():
                    if isinstance(value, dict):
                        print(" " * indent + f"{key}:")
                        print_nested(value, indent + 2)
                    else:
                        print(" " * indent + f"{key}: {value}")
                        
            print_nested(config_dict)
    
    @classmethod
    def get_config(cls, category, param=None):
        """Get configuration value(s) from specified category
        Args:
            category: The parameter category (e.g. 'VIDEO_PARAMS')
            param: Optional parameter name or nested dict keys as tuple (e.g. ('ASPECT_RATIOS','VERTICAL'))
        Returns:
            The requested configuration value or entire category dict if param is None
        """
        if hasattr(cls, category):
            config_dict = getattr(cls, category)
            if param is None:
                return config_dict
            elif isinstance(param, tuple):
                # Handle nested dictionary lookups
                current = config_dict
                for key in param:
                    current = current[key]
                return current
            else:
                # Handle top level lookups
                return config_dict[param]
        return None
