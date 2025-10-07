import os, random, numpy as np, torch
def seed_everything(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
