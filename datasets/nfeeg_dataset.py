import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import lmdb
import pickle
import sys 
import types

# ✅ Patch cho pickle
if not hasattr(np, "_core"):
    np._core = types.SimpleNamespace()
sys.modules["numpy._core"] = np
sys.modules["numpy._core.numeric"] = np

def to_tensor(array):
    return torch.from_numpy(array).float()

class N_CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(N_CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        # For UET175 with shape (22, 4, 200) or (22, 800)
        # Remove reshape or adjust to model’s expected input
        # Example: Keep as is or reshape to (22, 800) if needed
        return (data, label)  # Remove data/100 unless necessary

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch], dtype=np.int64)
        return (to_tensor(x_data), to_tensor(y_label).long())


class N_LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = N_CustomDataset(self.datasets_dir, mode='train')
        val_set = N_CustomDataset(self.datasets_dir, mode='val')
        test_set = N_CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=True,
            ),
        }
        return data_loader
