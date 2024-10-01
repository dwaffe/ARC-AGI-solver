import argparse
import torch
from torch.utils.data import random_split
from data.arc_dataset import ARCDataset
from utils.train import train_model
from models.hypernetwork import HyperNetwork
from models.main_network import DynamicMainNetwork

def main(args):
    # Set up datasets
    full_dataset = ARCDataset(args.data_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Train the model
    trained_hyper_network = train_model(
        train_dataset,
        val_dataset,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        main_network_hidden_size=args.main_network_hidden_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Evaluate the model
    evaluate_model(trained_hyper_network, val_dataset, args)

def evaluate_model(hyper_network, dataset, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper_network.eval()
    main_network = DynamicMainNetwork().to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataset:
            train_inputs, train_outputs, test_inputs, test_outputs = process_batch(batch)
            train_inputs, train_outputs, test_inputs, test_outputs = train_inputs.to(device), train_outputs.to(device), test_inputs.to(device), test_outputs.to(device)

            generated_params = hyper_network(train_inputs, train_outputs)
            main_network.initialize(1, 10, args.main_network_hidden_size, generated_params)

            predictions = main_network(test_inputs)
            _, predicted = torch.max(predictions, 2)
            total += test_outputs.numel()
            correct += (predicted == test_outputs).sum().item()

    accuracy = 100 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")

def process_batch(batch):
    train_inputs = torch.stack(batch['train_inputs']).float().unsqueeze(1)
    train_outputs = torch.stack(batch['train_outputs']).long()
    test_inputs = torch.stack(batch['test_inputs']).float().unsqueeze(1)
    test_outputs = torch.stack(batch['test_outputs']).long()
    return train_inputs, train_outputs, test_inputs, test_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ARC solver")
    parser.add_argument("--data_dir", type=str, default="data/evaluation", help="Directory containing ARC dataset")
    parser.add_argument("--embed_size", type=int, default=32, help="Embedding size for HyperNetwork")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for HyperNetwork")
    parser.add_argument("--main_network_hidden_size", type=int, default=64, help="Hidden size for MainNetwork")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")

    args = parser.parse_args()
    main(args)