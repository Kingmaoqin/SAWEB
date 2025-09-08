
from torch.utils.data import Dataset, DataLoader, random_split
class MultiTaskDataset(Dataset):
    def __init__(self, X_cat, X_num, targets, masks, event_all):
        self.X_cat = X_cat
        self.X_num = X_num
        self.targets = targets
        self.masks = masks
        self.event_all = event_all

    def __len__(self):
        return len(self.X_cat)  # 与 X_num 一样长

    def __getitem__(self, idx):
        x_cat = self.X_cat[idx]  # (num_cat,)
        x_num = self.X_num[idx]  # (num_num,)
        t = [self.targets[i][idx] for i in range(len(self.targets))]
        m = [self.masks[i][idx] for i in range(len(self.masks))]
        e = self.event_all[idx]
        return x_cat, x_num, t, m, e
