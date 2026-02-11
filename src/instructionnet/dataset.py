import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import os

class TAODataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.header_dtype = np.dtype([
            ("seq_length", "u4"),
            ("reserverd", "u4"),
        ])
        self.record_dtype = np.dtype([
            ('type',              'u1'),
            ('reg',               'u4'),
            ('reg_fp',            'u4'),
            ('mem_hist',          'i8', (64,)), 
            ('branch_hist',       'u4'),
            ('isMispredicted',    'u1'),
            ('isControl',         'u1'),
            ('isCondCtrl',        'u1'),
            ('isMemRef',          'u1'),
            ('fetch_latency',     'u2'),
            ('branch_mispredict', 'u1'),
            ('dcache_hit_level',  'u1'),
        ])

        if not os.path.exists(file_path):
            raise FileNotFoundError
        
        header = np.fromfile(file_path, dtype=self.header_dtype, count=1)
        self.seq_length = int(header[0]["seq_length"])
        self.data_mmap = np.memmap(
            file_path,
            dtype=self.record_dtype,
            mode="r",
            offset=8,
            shape=(self.seq_length,)
        )
    
    def __len__(self):
        return self.seq_length
    
    def __getitem__(self, index):
        record = self.data_mmap[index]
        return record


def collate_fn(batch):
    batch_np = np.stack(batch)
    types = torch.from_numpy(batch_np["type"].astype(np.int32)).unsqueeze(-1)
    reg = torch.from_numpy(batch_np["reg"].astype(np.int32)).unsqueeze(-1)
    reg_fp = torch.from_numpy(batch_np["reg_fp"].astype(np.int32)).unsqueeze(-1)
    mem_hist = torch.from_numpy(batch_np["mem_hist"].astype(np.int32))
    branch_hist = torch.from_numpy(batch_np["branch_hist"].astype(np.int32)).unsqueeze(-1)
    shifts = torch.arange(32)
    reg_bits = (reg >> shifts) & 1
    reg_fp_bits = (reg_fp >> shifts) & 1
    branch_hist_bits = (branch_hist >> shifts) & 1
    label = torch.cat([
        types,
        reg_bits,
        reg_fp_bits,
        mem_hist,
        branch_hist_bits,
    ], dim=1)

    fetch_latency = torch.from_numpy(batch_np["fetch_latency"].astype(np.int32))
    branch_mispredict = torch.from_numpy(batch_np["isMispredicted"].astype(np.int32))
    dcache_hit_level = torch.from_numpy(batch_np["dcache_hit_level"].astype(np.int32))
    ones = torch.ones_like(fetch_latency)
    ground_truth = torch.stack([
        fetch_latency, ones, branch_mispredict, ones, ones, dcache_hit_level
    ], dim=1)

    return label, ground_truth


class OverlappingSampler(Sampler):
    def __init__(self, data_source, batch_size, overlap, shuffle: bool = False):
        self.len = len(data_source)
        self.batch_size = batch_size
        self.overlap = overlap
        self.stride = batch_size - overlap
        self.shuffle = shuffle

    def __len__(self):
        return (self.len - self.overlap) // self.stride

    def __iter__(self):
        if self.shuffle:
            shuffled_indices = torch.randperm(len(self))
            for index in shuffled_indices:
                start = index * self.stride
                yield range(start, start + self.batch_size)
        else:
            for start in range(0, self.len - self.batch_size + 1, self.stride):
                yield range(start, start + self.batch_size)
