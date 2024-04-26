import torch.nn as nn

# Movie ID predictor model
class MovieIDPredictor(nn.Module):
    def __init__(self, num_movie_ids, movie_id_embedding_dim=64, transformer_heads=8, transformer_layers=1, transformer_dim=64):
        super(MovieIDPredictor, self).__init__()
        self.movie_id_embedding_dim = movie_id_embedding_dim # movie id embedding dimension (transforms movie id to a vector of this dimension)
        self.transformer_dim = transformer_dim # transformer dimension 
        self.movie_id_embedding = nn.Embedding(num_movie_ids, self.movie_id_embedding_dim) # movie id embedding

        # transformer encoder (to learn the relationships between movie ids using self-attention)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=self.transformer_dim, nhead=transformer_heads), num_layers=transformer_layers)
        self.fc_out = nn.Linear(self.transformer_dim, num_movie_ids) # output layer (maps the transformer output to the movie id space)

    def forward(self, movie_id):
        movie_id_emb = self.movie_id_embedding(movie_id).view(-1, 1, self.movie_id_embedding_dim) # movie id embedding
        x = self.transformer(movie_id_emb) # transformer
        x = x.view(-1, self.transformer_dim) # reshape to fit the output layer
        output = self.fc_out(x) # output layer
        return output # returns final scores for each movie ID
