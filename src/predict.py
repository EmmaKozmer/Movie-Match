import os
import torch
from .model_utils import load_model_from_filesystem, load_model_from_huggingface

# Predictor class
class Predictor:
    def __init__(self, model_identifier, num_movie_ids):
        # check which model to load
        if os.path.exists(model_identifier):
            self.model = load_model_from_filesystem(model_identifier, num_movie_ids) # load model from the local filesystem
        else:
            self.model = load_model_from_huggingface(model_identifier, num_movie_ids) # load model from Hugging Face
        self.model.eval()

    def predict(self, movie_id, n):
        movie_id_tensor = torch.tensor([movie_id], dtype=torch.long) # convert movie ID to tensor

        # make prediction
        with torch.no_grad():
            outputs = self.model(movie_id_tensor) # get model outputs
            _, recommended_ids = torch.topk(outputs, n + 1, dim=1) # get top n+1 recommended movie IDs
        recommended_ids = recommended_ids[0].tolist() # convert tensor to list

        if movie_id in recommended_ids:
            recommended_ids.remove(movie_id) # remove the input movie ID from the recommendations
        else:
            recommended_ids.pop() # remove the last element if the input movie ID is not in the recommendations
        return recommended_ids[:n]