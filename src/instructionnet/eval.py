from __future__ import annotations
from src.instructionnet.tao_model import TAOModel
from src.instructionnet.instructionnet_model import InstructionNet
from src.instructionnet.dataset import TAODataset, OverlappingSampler, collate_fn
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

@dataclass
class EvalConfig:
    datasets: list[str]
    name: str

    hidden_dim: int = 768

    batch_size: int = 512
    window_size: int = 128
    device: str = "cpu"

    load_state_file: str = ""
    max_time_seconds: float | None = None


@torch.no_grad()
def eval(config: EvalConfig):
    import time

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
        batch_sampler=OverlappingSampler(dataset, config.batch_size, config.window_size, True),
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

    # fetch_cycle 分类准确率统计
    fetch_cycle_class_correct = [0] * len(config.datasets)
    fetch_cycle_class_total = [0] * len(config.datasets)

    # fetch_cycle 高周期样本(>=11)的回归误差统计
    fetch_high_cycle_true_sum = [0.0] * len(config.datasets)
    fetch_high_cycle_pred_sum = [0.0] * len(config.datasets)
    fetch_high_cycle_count = [0] * len(config.datasets)

    # exec_cycle 分类准确率统计
    exec_cycle_class_correct = [0] * len(config.datasets)
    exec_cycle_class_total = [0] * len(config.datasets)

    # exec_cycle 高周期样本(>=11)的回归误差统计
    exec_high_cycle_true_sum = [0.0] * len(config.datasets)
    exec_high_cycle_pred_sum = [0.0] * len(config.datasets)
    exec_high_cycle_count = [0] * len(config.datasets)

    start_time = time.time()
    union_loader = zip(*dataloaders)
    pbar = tqdm(union_loader, total=length, unit="batch")
    for batch_idx, datas in enumerate(pbar):
        if config.max_time_seconds is not None and time.time() - start_time > config.max_time_seconds:
            pbar.close()
            break

        input = torch.stack([data[0] for data in datas]).to(device, non_blocking=True)
        target = torch.stack([data[1] for data in datas]).to(device, non_blocking=True)

        pred = model(input)

        for i in range(len(config.datasets)):
            fetch_cycle_pred = pred["fetch_cycle"][i, ..., config.window_size:]
            fetch_cycle_target = target[i, ..., config.window_size:, 0]
            true_cycles[i] += torch.sum(fetch_cycle_target).item()
            pred_cycles[i] += torch.sum(fetch_cycle_pred).item()

            # fetch_cycle 分类准确率
            fetch_cycle_class_logits = pred["fetch_cycle_class_logits"][i, ..., config.window_size:, :]
            fetch_cycle_class_pred = fetch_cycle_class_logits.argmax(dim=-1)
            fetch_cycle_class_target = torch.clamp(fetch_cycle_target.long() - 1, min=0, max=10)
            fetch_cycle_class_correct[i] += (fetch_cycle_class_pred == fetch_cycle_class_target).sum().item()
            fetch_cycle_class_total[i] += fetch_cycle_class_target.numel()

            # fetch_cycle 高周期样本(>=11)的回归误差
            fetch_high_cycle_mask = fetch_cycle_target >= 11
            if fetch_high_cycle_mask.any():
                fetch_cycle_regression = pred["fetch_cycle_regression"][i, ..., config.window_size:]
                fetch_high_cycle_true_sum[i] += fetch_cycle_target[fetch_high_cycle_mask].sum().item()
                fetch_high_cycle_pred_sum[i] += (fetch_cycle_regression[fetch_high_cycle_mask] * 100).sum().item()
                fetch_high_cycle_count[i] += fetch_high_cycle_mask.sum().item()

            # exec_cycle 相关统计
            exec_cycle_target = target[i, ..., config.window_size:, 1]
            exec_cycle_class_logits = pred["exec_cycle_class_logits"][i, ..., config.window_size:, :]
            exec_cycle_class_pred = exec_cycle_class_logits.argmax(dim=-1)
            exec_cycle_class_target = torch.clamp(exec_cycle_target.long() - 1, min=0, max=10)
            exec_cycle_class_correct[i] += (exec_cycle_class_pred == exec_cycle_class_target).sum().item()
            exec_cycle_class_total[i] += exec_cycle_class_target.numel()

            # exec_cycle 高周期样本(>=11)的回归误差
            exec_high_cycle_mask = exec_cycle_target >= 11
            if exec_high_cycle_mask.any():
                exec_cycle_regression = pred["exec_cycle_regression"][i, ..., config.window_size:]
                exec_high_cycle_true_sum[i] += exec_cycle_target[exec_high_cycle_mask].sum().item()
                exec_high_cycle_pred_sum[i] += (exec_cycle_regression[exec_high_cycle_mask] * 100).sum().item()
                exec_high_cycle_count[i] += exec_high_cycle_mask.sum().item()

            branch_pred = pred["branch_mispred"][i, ..., config.window_size:]
            branch_target = target[i, ..., config.window_size:, 2]
            is_control = target[i, ..., config.window_size:, 5].bool()
            if is_control.any():
                branch_correct[i] += (branch_pred[is_control].gt(0.5).eq(branch_target[is_control])).sum().item()
                branch_total[i] += is_control.sum().item()

            icache_pred = pred["icache_hit"][i, ..., config.window_size:, :]
            icache_target = target[i, ..., config.window_size:, 3]
            icache_correct[i] += (icache_pred.argmax(-1).eq(icache_target)).sum().item()
            icache_total[i] += icache_target.numel()

            dcache_pred = pred["dcache_hit"][i, ..., config.window_size:, :]
            dcache_target = target[i, ..., config.window_size:, 4]
            is_mem_ref = target[i, ..., config.window_size:, 6].bool()
            if is_mem_ref.any():
                dcache_correct[i] += (dcache_pred[is_mem_ref].argmax(-1).eq(dcache_target[is_mem_ref])).sum().item()
                dcache_total[i] += is_mem_ref.sum().item()

        if batch_idx % 50 == 0:
            errors = [(pred_cycles[i] - true_cycles[i]) / true_cycles[i] if true_cycles[i] > 0 else 0.0
                      for i in range(len(config.datasets))]
            avg_error = sum(errors) / len(errors) if errors else 0.0
            pbar.set_postfix({"avg_error": f"{avg_error:+.2%}"})

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

    print("\nExec Cycle Classification Accuracy (11 classes: 1-10 and 10+):")
    for i, dataset_path in enumerate(config.datasets):
        acc = exec_cycle_class_correct[i] / exec_cycle_class_total[i] if exec_cycle_class_total[i] > 0 else 0.0
        print(f"  {dataset_path}: {acc:.2%}")

    print("\nExec Cycle High Cycle (>=11) Regression Error:")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=lambda s: s.lower(), choices=["tao", "inet"], default="tao")
    parser.add_argument("--dataset", type=str, nargs="+")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--max-time", type=float, default=None, help="Maximum evaluation time in seconds")

    args = parser.parse_args()

    config = EvalConfig(
        datasets=args.dataset,
        name=args.name,
        device=args.device,
        load_state_file=args.model,
        max_time_seconds=args.max_time
    )
    eval(config)


if __name__ == "__main__":
    main()
