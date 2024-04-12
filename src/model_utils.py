# Utility functions for loading models

import torch
from huggingface_hub import hf_hub_download


# Utility function to load model from Hugging Face Hub
def load_model_from_huggingface(model_name, filename):
    model_path = hf_hub_download(repo_id=model_name, filename=filename)
    model = torch.load(model_path)
    model.eval()
    return model

# Utility function to load model from the filesystem
def load_model_from_filesystem(model_path, num_movie_ids):
    from .predict import MovieIDPredictor  # Move import here
    model = MovieIDPredictor(num_movie_ids=num_movie_ids)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model