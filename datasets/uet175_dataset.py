# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import os
# import random
# import lmdb
# import pickle


# def to_tensor(array):
#     return torch.from_numpy(array).float()


# class UETCustomDataset(Dataset):
#     def __init__(
#             self,
#             data_dir,
#             mode='train',
#     ):
#         super(UETCustomDataset, self).__init__()
#         self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
#         with self.db.begin(write=False) as txn:
#             self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

#     def __len__(self):
#         return len((self.keys))

#     def __getitem__(self, idx):
#         key = self.keys[idx]
#         with self.db.begin(write=False) as txn:
#             pair = pickle.loads(txn.get(key.encode()))
#         data = pair['sample']
#         label = pair['label']
#         # print(label)
#         return (data/100, label)

#     def collate(self, batch):
#         x_data = np.array([x[0] for x in batch])
#         y_label = np.array([x[1] for x in batch], dtype=np.int64)
#         return (to_tensor(x_data), to_tensor(y_label).long())


# class UET_LoadDataset(object):
#     def __init__(self, params):
#         self.params = params
#         self.datasets_dir = params.datasets_dir

#     def get_data_loader(self):
#         train_set = UETCustomDataset(self.datasets_dir, mode='train')
#         val_set = UETCustomDataset(self.datasets_dir, mode='val')
#         test_set = UETCustomDataset(self.datasets_dir, mode='test')
#         print(len(train_set), len(val_set), len(test_set))
#         print(len(train_set)+len(val_set)+len(test_set))
#         data_loader = {
#             'train': DataLoader(
#                 train_set,
#                 batch_size=self.params.batch_size,
#                 collate_fn=train_set.collate,
#                 shuffle=True,
#             ),
#             'val': DataLoader(
#                 val_set,
#                 batch_size=self.params.batch_size,
#                 collate_fn=val_set.collate,
#                 shuffle=True,
#             ),
#             'test': DataLoader(
#                 test_set,
#                 batch_size=self.params.batch_size,
#                 collate_fn=test_set.collate,
#                 shuffle=True,
#             ),
#         }
#         return data_loader
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lmdb
import pickle

def to_tensor(array):
    return torch.from_numpy(array).float()

class UETCustomDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        super(UETCustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
            if isinstance(self.keys, np.ndarray):
                self.keys = self.keys.tolist()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        # Keep UET175 shape (22, 4, 200) or (22, 800)
        return data, label  # Remove data/100 unless needed

    def collate(self, batch):
        x_data = [x[0] for x in batch]
        y_label = [x[1] for x in batch]
        if isinstance(x_data[0], np.ndarray):
            x_data = torch.stack([torch.from_numpy(x).float() for x in x_data])
        else:
            x_data = torch.stack(x_data)
        y_label = torch.tensor(y_label, dtype=torch.long)
        return x_data, y_label

class UET_LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = UETCustomDataset(self.datasets_dir, mode='train')
        val_set = UETCustomDataset(self.datasets_dir, mode='val')
        test_set = UETCustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=4,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                num_workers=4,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                num_workers=4,
            ),
        }
        return data_loader