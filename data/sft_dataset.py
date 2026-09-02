from torch.utils.data.dataset import Dataset


class SFTDataset(Dataset):
    def __init__(self, file_path: str, max_length: int = 1024):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass