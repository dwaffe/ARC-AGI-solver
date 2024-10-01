import os
import json
import torch
from torch.utils.data import Dataset

class ARCDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tasks = []
        self.load_tasks()

    def load_tasks(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_dir, filename), 'r') as f:
                    task = json.load(f)
                    self.tasks.append(task)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        # Process input-output pairs
        train_inputs = [self.grid_to_tensor(pair['input']) for pair in task['train']]
        train_outputs = [self.grid_to_tensor(pair['output']) for pair in task['train']]
        test_inputs = [self.grid_to_tensor(pair['input']) for pair in task['test']]
        test_outputs = [self.grid_to_tensor(pair['output']) for pair in task['test']]
        
        return {
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'test_inputs': test_inputs,
            'test_outputs': test_outputs
        }

    @staticmethod
    def grid_to_tensor(grid):
        return torch.tensor(grid, dtype=torch.long)

# Example usage:
# train_dataset = ARCDataset('path/to/training/data')
# eval_dataset = ARCDataset('path/to/evaluation/data')