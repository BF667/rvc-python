import torch
from typing import Tuple

def get_synthesizer(pth_path: str, device: torch.device = torch.device("cpu")) -> Tuple[torch.nn.Module, dict]:
    # Lazy import to reduce memory usage and startup time
    from rvc_python.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    # Load checkpoint only once and reuse
    cpt = torch.load(pth_path, map_location=torch.device("cpu"), weights_only=True)
    
    # Update config with embedding weight shape
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    
    # Get model parameters
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    
    # Use dictionary for model selection to reduce conditional branching
    model_map = {
        ("v1", 1): SynthesizerTrnMs256NSFsid,
        ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
        ("v2", 1): SynthesizerTrnMs768NSFsid,
        ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
    }
    
    # Initialize model
    model_class = model_map.get((version, if_f0))
    if model_class is None:
        raise ValueError(f"Unsupported model version: {version} or f0: {if_f0}")
    
    net_g = model_class(*cpt["config"], is_half=False)
    
    # Remove encoder and load weights
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    
    # Set model to evaluation mode and move to device
    net_g = net_g.float().eval().to(device)
    net_g.remove_weight_norm()
    
    return net_g, cpt
