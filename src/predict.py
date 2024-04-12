# prection logic

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from .model_utils import load_model_from_filesystem, load_model_from_huggingface


class MovieIDPredictor(nn.Module):
    def __init__(self, num_movie_ids, movie_id_embedding_dim=64, transformer_heads=8, transformer_layers=1, transformer_dim=64):
        super(MovieIDPredictor, self).__init__()

        # new movie_id embedding dimension
        self.movie_id_embedding_dim = movie_id_embedding_dim
        self.transformer_dim = transformer_dim

        # ensure the embedding dimension for movie_id matches the transformer dimension
        self.movie_id_embedding = nn.Embedding(
            num_movie_ids, self.movie_id_embedding_dim)

        # TransformerEncoder expects the input dimension (d_model) to match the transformer_dim
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=self.transformer_dim, nhead=transformer_heads), num_layers=transformer_layers)

        # output layer to classify movie IDs
        self.fc_out = nn.Linear(self.transformer_dim, num_movie_ids)

    def forward(self, movie_id):
        movie_id_emb = self.movie_id_embedding(
            movie_id).view(-1, 1, self.movie_id_embedding_dim)

        x = self.transformer(movie_id_emb)
        x = x.view(-1, self.transformer_dim)  
        output = self.fc_out(x)
        return output

class Predictor:
    def __init__(self, num_movie_ids, model_source='filesystem', model_path=None, huggingface_model_name=None, huggingface_filename=None):
        if model_source == 'huggingface':
            if not huggingface_model_name or not huggingface_filename:
                raise ValueError("Huggingface model name and filename must be provided for Hugging Face source")
            self.loaded_model = load_model_from_huggingface(huggingface_model_name, huggingface_filename, num_movie_ids)
        elif model_source == 'filesystem':
            if not model_path:
                raise ValueError("Model path must be provided for filesystem source")
            self.loaded_model = load_model_from_filesystem(model_path, num_movie_ids)
        else:
            raise ValueError("Unsupported model source")

        self.loaded_model.eval()

    def predict(self, movie_id, n):
        movie_id_tensor = torch.tensor([movie_id], dtype=torch.long)
        with torch.no_grad():
            outputs = self.loaded_model(movie_id_tensor)
            _, recommended_ids = torch.topk(outputs, n + 1, dim=1)
        recommended_ids = recommended_ids[0].tolist()
        if movie_id in recommended_ids:
            recommended_ids.remove(movie_id)
        else:
            recommended_ids.pop()
        return recommended_ids[:n]

