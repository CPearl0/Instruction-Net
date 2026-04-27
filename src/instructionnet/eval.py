from __future__ import annotations
from src.instructionnet.tao_model import TAOModel
from src.instructionnet.instructionnet_model import InstructionNet
from src.instructionnet.dataset import TAODataset, OverlappingSampler, collate_fn
from dataclasses import dataclass
import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

RECORD_DTYPE = np.dtype([
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

@dataclass
class EvalConfig:
    datasets: list[str]
    name: str

    hidden_dim: int = 768

    seq_len: int = 1024
    batch_size: int = 1
    window_size: int = 8
    device: str = "cpu"

    load_state_file: str = ""
    max_time_seconds: float | None = None
    save_results: bool = False
    output_dir: str = "eval_result"


@torch.no_grad()
def _save_eval_results(config: EvalConfig, model, device):
    os.makedirs(config.output_dir, exist_ok=True)
    ws = config.window_size
    bs = config.batch_size
    stride = config.seq_len - ws

    print("\n=== Evaluation with Result Saving ===")
    print("\nFetch Cycle Count Error:")

    for dataset_path in config.datasets:
        dataset = TAODataset(dataset_path)
        seq_length = len(dataset)

        fetch_pred_arr = np.zeros(seq_length, dtype=np.float32)
        exec_pred_arr = np.zeros(seq_length, dtype=np.float32)
        branch_pred_arr = np.zeros(seq_length, dtype=np.uint8)
        icache_pred_arr = np.zeros(seq_length, dtype=np.uint8)
        dcache_pred_arr = np.zeros(seq_length, dtype=np.uint8)
        has_pred = np.zeros(seq_length, dtype=np.bool_)

        dataloader = DataLoader(
            dataset,
            batch_sampler=OverlappingSampler(dataset, config.seq_len, ws, bs, False),
            collate_fn=collate_fn,
            num_workers=4,
        )

        true_cycles = 0.0
        pred_cycles = 0.0

        pbar = tqdm(dataloader, desc=os.path.basename(dataset_path))
        for batch_idx, (input_batch, target_batch) in enumerate(pbar):
            input_batch = input_batch.reshape(bs, config.seq_len, -1).to(device, non_blocking=True)
            target_batch = target_batch.reshape(bs, config.seq_len, -1).to(device, non_blocking=True)
            pred = model(input_batch)

            for j in range(bs):
                seq_start = (batch_idx * bs + j) * stride
                p_start = seq_start + ws
                p_len = min(config.seq_len - ws, seq_length - p_start)
                if p_len <= 0:
                    continue

                fetch_p = pred["fetch_cycle"][j, ws:ws + p_len].cpu().numpy()
                exec_p = pred["exec_cycle"][j, ws:ws + p_len].cpu().numpy()
                branch_p = pred["branch_mispred"][j, ws:ws + p_len].cpu().numpy().astype(np.uint8)
                icache_p = pred["icache_hit"][j, ws:ws + p_len].argmax(-1).cpu().numpy().astype(np.uint8)
                dcache_p = pred["dcache_hit"][j, ws:ws + p_len].argmax(-1).cpu().numpy().astype(np.uint8)

                fetch_target = target_batch[j, ws:ws + p_len, 0]
                true_cycles += fetch_target.sum().item()
                pred_cycles += float(fetch_p.sum())

                fetch_pred_arr[p_start:p_start + p_len] = fetch_p
                exec_pred_arr[p_start:p_start + p_len] = exec_p
                branch_pred_arr[p_start:p_start + p_len] = branch_p
                icache_pred_arr[p_start:p_start + p_len] = icache_p
                dcache_pred_arr[p_start:p_start + p_len] = dcache_p
                has_pred[p_start:p_start + p_len] = True

            if batch_idx % 5 == 0:
                error = abs((pred_cycles - true_cycles) / true_cycles) if true_cycles > 0 else 0.0
                pbar.set_postfix({"error": f"{error:.2%}"})

        error = (pred_cycles - true_cycles) / true_cycles if true_cycles > 0 else 0.0
        print(f"  {dataset_path}: {error:+.2%}")

        # Write output: copy original file and overwrite labels with predictions
        basename = os.path.basename(dataset_path)
        output_path = os.path.join(config.output_dir, basename)
        shutil.copy2(dataset_path, output_path)

        indices = np.where(has_pred)[0]
        out = np.memmap(output_path, dtype=RECORD_DTYPE, mode='r+', offset=8, shape=(seq_length,))

        out['fetch_latency'][indices] = np.clip(np.round(fetch_pred_arr[indices]), 0, 65535).astype(np.uint16)
        out['exec_latency'][indices] = np.clip(np.round(exec_pred_arr[indices]), 0, 65535).astype(np.uint16)

        icache_vals = icache_pred_arr[indices]
        out['icache_hit_level'][indices] = icache_vals
        out['icache_hit'][indices] = (icache_vals == 0).astype(np.uint8)

        is_mem = out['isMemRef'][indices]
        mem_mask = is_mem == 1
        if np.any(mem_mask):
            mem_idx = indices[mem_mask]
            d_vals = dcache_pred_arr[indices][mem_mask]
            out['dcache_hit_level'][mem_idx] = d_vals
            out['dcache_hit'][mem_idx] = (d_vals == 0).astype(np.uint8)

        is_ctrl = out['isControl'][indices]
        ctrl_mask = is_ctrl == 1
        if np.any(ctrl_mask):
            ctrl_idx = indices[ctrl_mask]
            b_vals = branch_pred_arr[indices][ctrl_mask]
            out['isMispredicted'][ctrl_idx] = (b_vals != 0).astype(np.uint8)
            out['branch_dir_wrong'][ctrl_idx] = (b_vals == 1).astype(np.uint8)
            out['branch_target_wrong'][ctrl_idx] = (b_vals == 2).astype(np.uint8)

        out.flush()
        del out

        print(f"  Saved {len(indices)}/{seq_length} predictions to {output_path}")


@torch.no_grad()
def eval(config: EvalConfig):
    import time

    if config.save_results:
        device = torch.device(config.device)
        if config.name == "tao":
            model = TAOModel(config.hidden_dim).to(device)
        else:
            model = InstructionNet(config.hidden_dim).to(device)
        if config.load_state_file:
            checkpoint = torch.load(config.load_state_file)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model loaded from {config.load_state_file}")
        model.eval()
        _save_eval_results(config, model, device)
        return

    device = torch.device(config.device)
    if config.name == "tao":
        model = TAOModel(config.hidden_dim).to(device)
    else:
        model = InstructionNet(config.hidden_dim).to(device)
    if config.load_state_file:
        checkpoint = torch.load(config.load_state_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {config.load_state_file}")
    model.eval()

    datasets = [TAODataset(f) for f in config.datasets]
    num_workers = 0 if config.device.startswith("cuda") else 4
    dataloaders = [DataLoader(
        dataset,
        batch_sampler=OverlappingSampler(dataset, config.seq_len, config.window_size, config.batch_size, True),
        collate_fn=collate_fn,
        num_workers=num_workers,
    ) for dataset in datasets]
    length = min(len(dataloader) for dataloader in dataloaders)

    true_cycles = [0.0] * len(config.datasets)
    pred_cycles = [0.0] * len(config.datasets)

    branch_correct = [0] * len(config.datasets)
    branch_total = [0] * len(config.datasets)
    icache_correct = [0] * len(config.datasets)
    icache_total = [0] * len(config.datasets)
    dcache_correct = [0] * len(config.datasets)
    dcache_total = [0] * len(config.datasets)

    # fetch_cycle classification accuracy
    fetch_cycle_class_correct = [0] * len(config.datasets)
    fetch_cycle_class_total = [0] * len(config.datasets)

    # fetch_cycle high cycle (>=11) regression error
    fetch_high_cycle_true_sum = [0.0] * len(config.datasets)
    fetch_high_cycle_pred_sum = [0.0] * len(config.datasets)
    fetch_high_cycle_count = [0] * len(config.datasets)

    # exec_cycle classification accuracy
    exec_cycle_class_correct = [0] * len(config.datasets)
    exec_cycle_class_total = [0] * len(config.datasets)

    # exec_cycle high cycle (>=11) regression error
    exec_high_cycle_true_sum = [0.0] * len(config.datasets)
    exec_high_cycle_pred_sum = [0.0] * len(config.datasets)
    exec_high_cycle_count = [0] * len(config.datasets)

    start_time = time.time()
    union_loader = zip(*dataloaders)
    pbar = tqdm(union_loader, total=length, unit="batch")
    nd = len(config.datasets)
    bs = config.batch_size
    for batch_idx, datas in enumerate(pbar):
        if config.max_time_seconds is not None and time.time() - start_time > config.max_time_seconds:
            pbar.close()
            break

        inputs = torch.stack([data[0] for data in datas])
        targets = torch.stack([data[1] for data in datas])
        input = inputs.reshape(nd * bs, config.seq_len, -1).to(device, non_blocking=True)
        target = targets.reshape(nd * bs, config.seq_len, -1).to(device, non_blocking=True)

        pred = model(input)

        for i in range(nd):
            for j in range(bs):
                idx = i * bs + j
                fetch_cycle_pred = pred["fetch_cycle"][idx, ..., config.window_size:]
                fetch_cycle_target = target[idx, ..., config.window_size:, 0]
                true_cycles[i] += torch.sum(fetch_cycle_target).item()
                pred_cycles[i] += torch.sum(fetch_cycle_pred).item()

                fetch_cycle_class_logits = pred["fetch_cycle_class_logits"][idx, ..., config.window_size:, :]
                fetch_cycle_class_pred = fetch_cycle_class_logits.argmax(dim=-1)
                fetch_cycle_class_target = torch.clamp(fetch_cycle_target.long() - 1, min=0, max=10)
                fetch_cycle_class_correct[i] += (fetch_cycle_class_pred == fetch_cycle_class_target).sum().item()
                fetch_cycle_class_total[i] += fetch_cycle_class_target.numel()

                fetch_high_cycle_mask = fetch_cycle_target >= 11
                if fetch_high_cycle_mask.any():
                    fetch_cycle_regression = pred["fetch_cycle_regression"][idx, ..., config.window_size:]
                    fetch_high_cycle_true_sum[i] += fetch_cycle_target[fetch_high_cycle_mask].sum().item()
                    fetch_high_cycle_pred_sum[i] += (fetch_cycle_regression[fetch_high_cycle_mask] * 100).sum().item()
                    fetch_high_cycle_count[i] += fetch_high_cycle_mask.sum().item()

                exec_cycle_target = target[idx, ..., config.window_size:, 1]
                exec_cycle_class_logits = pred["exec_cycle_class_logits"][idx, ..., config.window_size:, :]
                exec_cycle_class_pred = exec_cycle_class_logits.argmax(dim=-1)
                exec_cycle_class_target = torch.clamp(exec_cycle_target.long() - 1, min=0, max=20)
                exec_cycle_class_correct[i] += (exec_cycle_class_pred == exec_cycle_class_target).sum().item()
                exec_cycle_class_total[i] += exec_cycle_class_target.numel()

                exec_high_cycle_mask = exec_cycle_target >= 21
                if exec_high_cycle_mask.any():
                    exec_cycle_regression = pred["exec_cycle_regression"][idx, ..., config.window_size:]
                    exec_high_cycle_true_sum[i] += exec_cycle_target[exec_high_cycle_mask].sum().item()
                    exec_high_cycle_pred_sum[i] += (exec_cycle_regression[exec_high_cycle_mask] * 100).sum().item()
                    exec_high_cycle_count[i] += exec_high_cycle_mask.sum().item()

                branch_pred = pred["branch_mispred"][idx, ..., config.window_size:]
                branch_target = target[idx, ..., config.window_size:, 2]
                is_control = target[idx, ..., config.window_size:, 5].bool()
                if is_control.any():
                    branch_correct[i] += (branch_pred[is_control] == branch_target[is_control]).sum().item()
                    branch_total[i] += is_control.sum().item()

                icache_pred = pred["icache_hit"][idx, ..., config.window_size:, :]
                icache_target = target[idx, ..., config.window_size:, 3]
                icache_correct[i] += (icache_pred.argmax(-1).eq(icache_target)).sum().item()
                icache_total[i] += icache_target.numel()

                dcache_pred = pred["dcache_hit"][idx, ..., config.window_size:, :]
                dcache_target = target[idx, ..., config.window_size:, 4]
                is_mem_ref = target[idx, ..., config.window_size:, 6].bool()
                if is_mem_ref.any():
                    dcache_correct[i] += (dcache_pred[is_mem_ref].argmax(-1).eq(dcache_target[is_mem_ref])).sum().item()
                    dcache_total[i] += is_mem_ref.sum().item()

        if batch_idx % 5 == 0:
            errors = [abs((pred_cycles[i] - true_cycles[i]) / true_cycles[i]) if true_cycles[i] > 0 else 0.0
                      for i in range(nd)]
            max_error = max(errors) if errors else 0.0
            pbar.set_postfix({"max_error": f"{max_error:.2%}"})

    print("\n=== Evaluation Results ===")
    print("\nFetch Cycle Count Error:")
    for i, dataset_path in enumerate(config.datasets):
        error = (pred_cycles[i] - true_cycles[i]) / true_cycles[i] if true_cycles[i] > 0 else 0.0
        print(f"  {dataset_path}: {error:+.2%}")

    print("\nFetch Cycle Classification Accuracy (11 classes: 1-10 and 10+):")
    for i, dataset_path in enumerate(config.datasets):
        acc = fetch_cycle_class_correct[i] / fetch_cycle_class_total[i] if fetch_cycle_class_total[i] > 0 else 0.0
        print(f"  {dataset_path}: {acc:.2%}")

    print("\nFetch Cycle High Cycle (>=11) Regression Error:")
    for i, dataset_path in enumerate(config.datasets):
        if fetch_high_cycle_count[i] > 0:
            error = (fetch_high_cycle_pred_sum[i] - fetch_high_cycle_true_sum[i]) / fetch_high_cycle_true_sum[i]
            print(f"  {dataset_path}: {error:+.2%} (count: {fetch_high_cycle_count[i]})")
        else:
            print(f"  {dataset_path}: N/A (no high cycle samples)")

    print("\nExec Cycle Classification Accuracy (21 classes: 1-20 and 20+):")
    for i, dataset_path in enumerate(config.datasets):
        acc = exec_cycle_class_correct[i] / exec_cycle_class_total[i] if exec_cycle_class_total[i] > 0 else 0.0
        print(f"  {dataset_path}: {acc:.2%}")

    print("\nExec Cycle High Cycle (>=21) Regression Error:")
    for i, dataset_path in enumerate(config.datasets):
        if exec_high_cycle_count[i] > 0:
            error = (exec_high_cycle_pred_sum[i] - exec_high_cycle_true_sum[i]) / exec_high_cycle_true_sum[i]
            print(f"  {dataset_path}: {error:+.2%} (count: {exec_high_cycle_count[i]})")
        else:
            print(f"  {dataset_path}: N/A (no high cycle samples)")

    print("\nBranch Prediction Accuracy:")
    for i, dataset_path in enumerate(config.datasets):
        acc = branch_correct[i] / branch_total[i] if branch_total[i] > 0 else 0.0
        print(f"  {dataset_path}: {acc:.2%}")

    print("\nICache Hit Level Accuracy (3 classes: L1/L2/Memory):")
    for i, dataset_path in enumerate(config.datasets):
        acc = icache_correct[i] / icache_total[i] if icache_total[i] > 0 else 0.0
        print(f"  {dataset_path}: {acc:.2%}")

    print("\nDCache Hit Level Accuracy (3 classes: L1/L2/Memory):")
    for i, dataset_path in enumerate(config.datasets):
        acc = dcache_correct[i] / dcache_total[i] if dcache_total[i] > 0 else 0.0
        print(f"  {dataset_path}: {acc:.2%}")


def load_datasets(path="datasets.txt"):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=lambda s: s.lower(), choices=["tao", "inet"], default="inet")
    parser.add_argument("--dataset-file", type=str, default="datasets.txt")
    parser.add_argument("--eval-data", type=int, nargs="+", default=[])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--max-time", type=float, default=None, help="Maximum evaluation time in seconds")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_result")

    args = parser.parse_args()

    all_datasets = load_datasets(args.dataset_file)
    if args.eval_data:
        eval_datasets = [all_datasets[i] for i in args.eval_data]
    else:
        eval_datasets = all_datasets

    config = EvalConfig(
        datasets=eval_datasets,
        name=args.name,
        device=args.device,
        load_state_file=args.model,
        max_time_seconds=args.max_time,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        save_results=args.save_results,
        output_dir=args.output_dir,
    )
    eval(config)


if __name__ == "__main__":
    main()
