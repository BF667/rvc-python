import torch
from rvc_python.lib.rmvpe import E2E


def get_rmvpe(model_path="base_model/rmvpe.pt"):
    """
    Load and initialize RMVPE model with automatic device detection.
    
    Args:
        model_path (str): Path to the RMVPE model checkpoint
        
    Returns:
        E2E: Loaded and initialized RMVPE model
    """
    # Automatic device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = E2E(4, 1, (2, 2))
    
    # Load checkpoint with memory-efficient options
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    
    # Set model to evaluation mode and move to device
    model.eval()
    if device.type == "cuda":
        model = model.to(device, non_blocking=True)
    else:
        model = model.to(device)
    
    return model
