import torch

class Device:
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # Metal Performance Shaders for Apple Silicon
            device_name = "Apple Silicon (Metal Performance Shaders)"
        else:
            device = torch.device("cpu")
            device_name = "CPU"
        print(f"GPU             : {device}")
        print(f"Using device    : {device_name}")
        return device
