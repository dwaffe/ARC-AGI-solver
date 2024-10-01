import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.hypernetwork import HyperNetwork, calculate_main_network_params
from models.main_network import DynamicMainNetwork

def custom_collate(batch):
    return batch

def pad_to_size(tensor, size):
    h, w = tensor.shape
    padded = torch.zeros(size, size, dtype=tensor.dtype)
    padded[:h, :w] = tensor
    return padded

def train_model(train_dataset, val_dataset, embed_size=32, hidden_size=128, main_network_hidden_size=64, num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate the number of parameters for the main network
    main_network_param_size = calculate_main_network_params(1, 10, main_network_hidden_size)
    
    # Initialize models
    hyper_network = HyperNetwork(embed_size, hidden_size, main_network_param_size).to(device)
    main_network = DynamicMainNetwork().to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(hyper_network.parameters(), lr=learning_rate)
    
    # Set up data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate)
    
    for epoch in range(num_epochs):
        # Training
        hyper_network.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            batch_loss = 0.0
            for example in batch:
                # Find maximum size for this example
                max_size = max(max(inp.shape) for inp in example['train_inputs'] + example['train_outputs'] + example['test_inputs'] + example['test_outputs'])
                
                train_inputs = torch.stack([pad_to_size(torch.tensor(inp, dtype=torch.long), max_size) for inp in example['train_inputs']])
                train_outputs = torch.stack([pad_to_size(torch.tensor(out, dtype=torch.long), max_size) for out in example['train_outputs']])
                test_inputs = torch.stack([pad_to_size(torch.tensor(inp, dtype=torch.long), max_size) for inp in example['test_inputs']])
                test_outputs = torch.stack([pad_to_size(torch.tensor(out, dtype=torch.long), max_size) for out in example['test_outputs']])
                
                train_inputs = train_inputs.unsqueeze(1).to(device)
                train_outputs = train_outputs.to(device)
                test_inputs = test_inputs.unsqueeze(1).to(device)
                test_outputs = test_outputs.to(device)
                
                # Generate parameters for main network
                generated_params = hyper_network(train_inputs, train_outputs)
                
                # Initialize main network with generated parameters
                main_network.initialize(1, 10, main_network_hidden_size, generated_params)
                
                # Forward pass
                predictions = main_network(test_inputs.float())  # Convert to float for the main network
                
                # Compute loss
                loss = nn.CrossEntropyLoss()(predictions.view(-1, 10), test_outputs.view(-1))
                batch_loss += loss
            
            # Backward pass and optimize
            batch_loss /= len(batch)
            batch_loss.backward()
            optimizer.step()
            
            train_loss += batch_loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        hyper_network.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_loss = 0.0
                for example in batch:
                    # Find maximum size for this example
                    max_size = max(max(inp.shape) for inp in example['train_inputs'] + example['train_outputs'] + example['test_inputs'] + example['test_outputs'])
                    
                    train_inputs = torch.stack([pad_to_size(torch.tensor(inp, dtype=torch.long), max_size) for inp in example['train_inputs']])
                    train_outputs = torch.stack([pad_to_size(torch.tensor(out, dtype=torch.long), max_size) for out in example['train_outputs']])
                    test_inputs = torch.stack([pad_to_size(torch.tensor(inp, dtype=torch.long), max_size) for inp in example['test_inputs']])
                    test_outputs = torch.stack([pad_to_size(torch.tensor(out, dtype=torch.long), max_size) for out in example['test_outputs']])
                    
                    train_inputs = train_inputs.unsqueeze(1).to(device)
                    train_outputs = train_outputs.to(device)
                    test_inputs = test_inputs.unsqueeze(1).to(device)
                    test_outputs = test_outputs.to(device)
                    
                    generated_params = hyper_network(train_inputs, train_outputs)
                    main_network.initialize(1, 10, main_network_hidden_size, generated_params)
                    
                    predictions = main_network(test_inputs.float())  # Convert to float for the main network
                    loss = nn.CrossEntropyLoss()(predictions.view(-1, 10), test_outputs.view(-1))
                    batch_loss += loss
                
                batch_loss /= len(batch)
                val_loss += batch_loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return hyper_network