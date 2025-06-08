import torch
from typing import Tuple

def get_device() -> torch.device:
    """Detect and return the optimal device (CUDA GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_synthesizer(pth_path: str, device: torch.device = None) -> Tuple[torch.nn.Module, dict]:
    """
    Load and initialize a synthesizer model with optimized device handling.
    
    Args:
        pth_path (str): Path to the model checkpoint file
        device (torch.device, optional): Target device. If None, auto-detects GPU/CPU.
    
    Returns:
        Tuple containing the initialized model and checkpoint data.
    """
    # Lazy import to reduce memory usage and improve startup time
    from rvc_python.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    # Use auto-detected device if none provided
    device = device or get_device()
    
    try:
        # Load checkpoint to CPU first to minimize GPU memory usage
        cpt = torch.load(pth_path, map_location=torch.device("cpu"))
        
        # Extract configuration
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")

        # Initialize model based on version and f0
        model_classes = {
            "v1": {
                1: SynthesizerTrnMs256NSFsid,
                0: SynthesizerTrnMs256NSFsid_nono
            },
            "v2": {
                1: SynthesizerTrnMs768NSFsid,
                0: SynthesizerTrnMs768NSFsid_nono
            }
        }
        
        try:
            net_g = model_classes[version][if_f0](*cpt["config"], is_half=False)
        except KeyError:
            raise ValueError(f"===> Unsupported model version: {version}")

        # Remove unnecessary encoder
        del net_g.enc_q
        
        # Load weights and configure model
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g = net_g.float().eval().to(device)
        net_g.remove_weight_norm()

        return net_g, cpt

    except FileNotFoundError:
        raise FileNotFoundError(f"===> Checkpoint file not found: {pth_path}")
    except Exception as e:
        raise RuntimeError(f"===> Error loading synthesizer: {str(e)}")
