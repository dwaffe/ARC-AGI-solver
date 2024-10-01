import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNetwork(nn.Module):
    def __init__(self, embed_size, hidden_size, main_network_param_size):
        super(HyperNetwork, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.main_network_param_size = main_network_param_size

        self.pixel_embedding = nn.Embedding(num_embeddings=10, embedding_dim=embed_size)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=4)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, main_network_param_size)

    def forward(self, examples_input, examples_output):
        batch_size, num_examples, height, width = examples_input.shape

        # Embed pixels
        embedded_input = self.pixel_embedding(examples_input.view(batch_size, num_examples, -1))
        embedded_output = self.pixel_embedding(examples_output.view(batch_size, num_examples, -1))

        # Concatenate input and output embeddings
        embedded = torch.cat([embedded_input, embedded_output], dim=2)
        embedded = embedded.view(batch_size, num_examples, -1, self.embed_size)

        # Apply attention
        attended, _ = self.attention(embedded, embedded, embedded)

        # Process through fully connected layers
        x = F.relu(self.fc1(attended.mean(dim=1)))
        params = self.fc2(x)

        return params.view(batch_size, -1)