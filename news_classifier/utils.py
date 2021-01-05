import json

import torch

def load_json(path: str):
    """Read a json file into a dict."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data: dict, path: str):
    with open(path, 'w') as fp:
        json.dump(data, fp)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])


def get_latest_model_checkpoint(model_dir):
    checkpoints = [f for f in model_dir.glob('**/*') if f.is_file()]
    if not checkpoints:
        return
    checkpoints = sorted(checkpoints)
    return checkpoints[-1]
