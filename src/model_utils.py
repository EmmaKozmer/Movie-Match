import torch
from huggingface_hub import hf_hub_download
from .models import MovieIDPredictor  # Adjust this path as needed

# Load model from filesystem
def load_model_from_filesystem(model_path, num_movie_ids):
    model = MovieIDPredictor(num_movie_ids=num_movie_ids)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load model from Hugging Face Hub
def load_model_from_huggingface(model_name, num_movie_ids, filename='movie_predictor_model_368K.pth'):
    model_path = hf_hub_download(repo_id=model_name, filename=filename)
    model = MovieIDPredictor(num_movie_ids=num_movie_ids)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
