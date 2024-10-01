import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.hypernetwork import HyperNetwork
from models.main_network import DynamicMainNetwork

def train_model(train_dataset, val_dataset, embed_size=32, hidden_size=128, main_network_hidden_size=64, num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    hyper_network = HyperNetwork(embed_size, hidden_size, main_network_hidden_size * 3).to(device)
    main_network = DynamicMainNetwork().to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(hyper_network.parameters(), lr=learning_rate)
    
    # Set up data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    for epoch in range(num_epochs):
        # Training
        hyper_network.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            train_inputs, train_outputs, test_inputs, test_outputs = process_batch(batch)
            
            # Generate parameters for main network
            generated_params = hyper_network(train_inputs, train_outputs)
            
            # Initialize main network with generated parameters
            main_network.initialize(1, 10, main_network_hidden_size, generated_params)
            
            # Forward pass
            predictions = main_network(test_inputs)
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(predictions.view(-1, 10), test_outputs.view(-1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        hyper_network.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                train_inputs, train_outputs, test_inputs, test_outputs = process_batch(batch)
                
                generated_params = hyper_network(train_inputs, train_outputs)
                main_network.initialize(1, 10, main_network_hidden_size, generated_params)
                
                predictions = main_network(test_inputs)
                loss = nn.CrossEntropyLoss()(predictions.view(-1, 10), test_outputs.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return hyper_network

def process_batch(batch):
    train_inputs = torch.stack(batch['train_inputs']).float().unsqueeze(1)  # Add channel dimension
    train_outputs = torch.stack(batch['train_outputs']).long()
    test_inputs = torch.stack(batch['test_inputs']).float().unsqueeze(1)  # Add channel dimension
    test_outputs = torch.stack(batch['test_outputs']).long()
    return train_inputs, train_outputs, test_inputs, test_outputs