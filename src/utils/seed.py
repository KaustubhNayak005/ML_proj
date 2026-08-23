import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    """
    Sets the random seed universally for reproducible results.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Ensuring determinism for CuDNN (can slow down training slightly, but ensures exact reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to: {seed}")

if __name__ == "__main__":
    set_seed(42)
