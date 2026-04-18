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
            ('pc',                    'u8'),
            ('type',                  'u1'),
            ('int_reg',               'u4'),
            ('fp_reg',                'u4'),
            ('branch_hist',           'u4'),
            ('isMispredicted',        'u1'),
            ('branch_dir_wrong',      'u1'),
            ('branch_target_wrong',   'u1'),
            ('isControl',             'u1'),
            ('isCondCtrl',            'u1'),
            ('isMemRef',              'u1'),
            ('same_icache_line_hist', 'u8'),
            ('same_dcache_line_hist', 'u8'),
            ('same_page_hist',        'u8'),
            ('fetch_latency',         'u2'),
            ('exec_latency',          'u2'),
            ('dcache_hit_level',      'u1'),
            ('icache_hit_level',      'u1'),
            ('icache_hit',            'u1'),
            ('dcache_hit',            'u1'),
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
    int_reg = torch.from_numpy(batch_np["int_reg"].astype(np.int64)).unsqueeze(-1)
    fp_reg = torch.from_numpy(batch_np["fp_reg"].astype(np.int64)).unsqueeze(-1)
    same_icache_line_hist = torch.from_numpy(batch_np["same_icache_line_hist"].astype(np.int64)).unsqueeze(-1)
    same_dcache_line_hist = torch.from_numpy(batch_np["same_dcache_line_hist"].astype(np.int64)).unsqueeze(-1)
    same_page_hist = torch.from_numpy(batch_np["same_page_hist"].astype(np.int64)).unsqueeze(-1)
    branch_hist = torch.from_numpy(batch_np["branch_hist"].astype(np.int64)).unsqueeze(-1)

    shifts32 = torch.arange(32)
    shifts64 = torch.arange(64)
    int_reg_bits = (int_reg >> shifts32) & 1
    fp_reg_bits = (fp_reg >> shifts32) & 1
    same_icache_line_hist_bits = (same_icache_line_hist >> shifts64) & 1
    same_dcache_line_hist_bits = (same_dcache_line_hist >> shifts64) & 1
    same_page_hist_bits = (same_page_hist >> shifts64) & 1
    branch_hist_bits = (branch_hist >> shifts32) & 1

    is_control_feat = torch.from_numpy(batch_np["isControl"].astype(np.float32)).unsqueeze(-1)
    is_cond_ctrl_feat = torch.from_numpy(batch_np["isCondCtrl"].astype(np.float32)).unsqueeze(-1)
    is_mem_ref_feat = torch.from_numpy(batch_np["isMemRef"].astype(np.float32)).unsqueeze(-1)

    label = torch.cat([
        types,
        int_reg_bits,
        fp_reg_bits,
        same_icache_line_hist_bits,
        same_dcache_line_hist_bits,
        same_page_hist_bits,
        branch_hist_bits,
        is_control_feat,
        is_cond_ctrl_feat,
        is_mem_ref_feat,
    ], dim=1)

    fetch_latency = torch.from_numpy(batch_np["fetch_latency"].astype(np.int32))
    exec_latency = torch.from_numpy(batch_np["exec_latency"].astype(np.int32))
    # Branch prediction: 0=correct, 1=direction wrong, 2=target wrong
    branch_pred = np.where(
        batch_np["isMispredicted"] == 0, 0,
        np.where(batch_np["branch_target_wrong"] == 1, 2, 1)
    ).astype(np.int32)
    branch_pred = torch.from_numpy(branch_pred)
    icache_hit_level = torch.from_numpy(batch_np["icache_hit_level"].astype(np.int32))
    dcache_hit_level = torch.from_numpy(batch_np["dcache_hit_level"].astype(np.int32))
    dcache_hit_level = dcache_hit_level.clamp(max=2)
    is_control = torch.from_numpy(batch_np["isControl"].astype(np.int32))
    is_mem_ref = torch.from_numpy(batch_np["isMemRef"].astype(np.int32))
    ground_truth = torch.stack([
        fetch_latency, exec_latency, branch_pred, icache_hit_level, dcache_hit_level,
        is_control, is_mem_ref
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
