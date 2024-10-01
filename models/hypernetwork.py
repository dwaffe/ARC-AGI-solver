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

        # Flatten spatial dimensions and embed pixels
        embedded_input = self.pixel_embedding(examples_input.view(batch_size * num_examples, -1))
        embedded_output = self.pixel_embedding(examples_output.view(batch_size * num_examples, -1))

        # Concatenate input and output embeddings
        embedded = torch.cat([embedded_input, embedded_output], dim=1)
        
        # Reshape for attention: (seq_len, batch_size, embed_dim)
        embedded = embedded.view(-1, batch_size * num_examples, self.embed_size).transpose(0, 1)

        # Apply attention
        attended, _ = self.attention(embedded, embedded, embedded)

        # Reshape back: (batch_size, num_examples, seq_len, embed_dim)
        attended = attended.transpose(0, 1).reshape(batch_size, num_examples, -1, self.embed_size)

        # Process through fully connected layers
        x = F.relu(self.fc1(attended.mean(dim=(1, 2))))
        params = self.fc2(x)

        # Ensure the output has the correct size
        params = params.view(batch_size, -1)
        if params.size(1) != self.main_network_param_size:
            raise ValueError(f"HyperNetwork output size {params.size(1)} does not match expected size {self.main_network_param_size}")

        return params

def calculate_main_network_params(input_channels, output_channels, hidden_size):
    # Calculate the number of parameters in the MainNetwork
    conv1_params = (input_channels * hidden_size * 3 * 3) + hidden_size
    conv2_params = (hidden_size * hidden_size * 3 * 3) + hidden_size
    conv3_params = (hidden_size * output_channels * 1 * 1) + output_channels
    total_params = conv1_params + conv2_params + conv3_params
    return total_params