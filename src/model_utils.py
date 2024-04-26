import torch
from huggingface_hub import hf_hub_download
from .models import MovieIDPredictor

# load model from filesystem
def load_model_from_filesystem(model_path, num_movie_ids):
    model = MovieIDPredictor(num_movie_ids=num_movie_ids) # initialize the model
    model.load_state_dict(torch.load(model_path)) # load the model state
    model.eval() # set the model to evaluation mode
    return model

# load model from Hugging Face Hub
def load_model_from_huggingface(model_name, num_movie_ids, filename='movie_predictor_model_368K.pth'):
    model_path = hf_hub_download(repo_id=model_name, filename=filename) # download the model from the Hugging Face Hub
    model = MovieIDPredictor(num_movie_ids=num_movie_ids) # initialize the model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # load the model state
    model.eval() # set the model to evaluation mode
    return model
