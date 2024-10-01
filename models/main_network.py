import torch
import torch.nn as nn
import torch.nn.functional as F

class MainNetwork(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_size):
        super(MainNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, output_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def load_parameters(self, generated_params):
        param_dict = dict(self.named_parameters())
        
        start = 0
        for name, param in param_dict.items():
            param_length = param.numel()
            param.data = generated_params[start:start+param_length].view(param.size())
            start += param_length

class DynamicMainNetwork(nn.Module):
    def __init__(self):
        super(DynamicMainNetwork, self).__init__()
        self.main_network = None

    def initialize(self, input_channels, output_channels, hidden_size, generated_params):
        self.main_network = MainNetwork(input_channels, output_channels, hidden_size)
        self.main_network.load_parameters(generated_params)

    def forward(self, x):
        if self.main_network is None:
            raise RuntimeError("MainNetwork not initialized. Call initialize() first.")
        return self.main_network(x)