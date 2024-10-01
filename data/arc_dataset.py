from torch.utils.data import Dataset

class ARCDataset(Dataset):
    def __init__(self, tasks):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        # Implementacja __getitem__
        pass