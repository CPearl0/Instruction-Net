from __future__ import annotations
from src.instructionnet.instructionnet_model import (
    InstructionNet, BranchPredictor, ICachePredictor, DCachePredictor, build_main_input,
)
from src.instructionnet.dataset import TAODataset, OverlappingSampler, collate_fn
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

@dataclass
class EvalConfig:
    datasets: list[str]

    hidden_dim: int = 512
    seq_len: int = 1024
    batch_size: int = 1
    window_size: int = 8
    device: str = "cpu"
    load_state_file: str = ""
    max_time_seconds: float | None = None
    gt_components: bool = False


def _load_models(config):
    device = torch.device(config.device)
    branch_predictor = BranchPredictor().to(device)
    icache_predictor = ICachePredictor().to(device)
    dcache_predictor = DCachePredictor().to(device)
    main_model = InstructionNet(config.hidden_dim).to(device)

    if config.load_state_file:
        checkpoint = torch.load(config.load_state_file)
        branch_predictor.load_state_dict(checkpoint["branch_predictor"])
        icache_predictor.load_state_dict(checkpoint["icache_predictor"])
        dcache_predictor.load_state_dict(checkpoint["dcache_predictor"])
        main_model.load_state_dict(checkpoint["main_model"])
        print(f"Model loaded from {config.load_state_file}")

    branch_predictor.eval()
    icache_predictor.eval()
    dcache_predictor.eval()
    main_model.eval()

    return branch_predictor, icache_predictor, dcache_predictor, main_model, device


@torch.no_grad()
def _inference(branch_predictor, icache_predictor, dcache_predictor, main_model, component_inputs, window_size):
    branch_logits = branch_predictor(component_inputs["branch_hist"])
    icache_logits = icache_predictor(component_inputs["icache_hist"])
    dcache_logits = dcache_predictor(component_inputs["dcache_hist"], component_inputs["page_hist"])

    branch_onehot = F.one_hot(branch_logits.argmax(-1), 3).float()
    icache_onehot = F.one_hot(icache_logits.argmax(-1), 3).float()
    dcache_onehot = F.one_hot(dcache_logits.argmax(-1), 3).float()

    main_input = build_main_input(
        component_inputs["type_reg_flags"], branch_onehot, icache_onehot, dcache_onehot)
    main_pred = main_model(main_input, window_size)

    return {
        **main_pred,
        "branch_mispred": branch_logits.argmax(-1),
        "icache_hit": F.softmax(icache_logits, dim=-1),
        "dcache_hit": F.softmax(dcache_logits, dim=-1),
    }


@torch.no_grad()
def _inference_gt(main_model, component_inputs, target, window_size):
    branch_onehot = F.one_hot(target[..., 2].long(), num_classes=3).float()
    icache_onehot = F.one_hot(target[..., 3].long(), num_classes=3).float()
    dcache_onehot = F.one_hot(target[..., 4].long().clamp(max=2), num_classes=3).float()

    main_input = build_main_input(
        component_inputs["type_reg_flags"], branch_onehot, icache_onehot, dcache_onehot)
    main_pred = main_model(main_input, window_size)

    return {
        **main_pred,
        "branch_mispred": target[..., 2],
        "icache_hit": F.one_hot(target[..., 3].long(), num_classes=3).float(),
        "dcache_hit": F.one_hot(target[..., 4].long().clamp(max=2), num_classes=3).float(),
    }


def _move_component_inputs(component_inputs, device, nd, bs, seq_len):
    result = {}
    for key, val in component_inputs.items():
        val = val.reshape(nd * bs, seq_len, -1)
        result[key] = val.to(device, non_blocking=True)
    return result


@torch.no_grad()
def eval(config: EvalConfig):
    import time

    branch_predictor, icache_predictor, dcache_predictor, main_model, device = _load_models(config)

    datasets = [TAODataset(f) for f in config.datasets]
    num_workers = 0 if config.device.startswith("cuda") else 4
    dataloaders = [DataLoader(
        dataset,
        batch_sampler=OverlappingSampler(dataset, config.seq_len, config.window_size, config.batch_size, True),
        collate_fn=collate_fn,
        num_workers=num_workers,
    ) for dataset in datasets]
    length = min(len(dataloader) for dataloader in dataloaders)

    fetch_true_sum = [0.0] * len(config.datasets)
    fetch_pred_sum = [0.0] * len(config.datasets)

    branch_correct = [0] * len(config.datasets)
    branch_total = [0] * len(config.datasets)
    icache_correct = [0] * len(config.datasets)
    icache_total = [0] * len(config.datasets)
    dcache_correct = [0] * len(config.datasets)
    dcache_total = [0] * len(config.datasets)

    start_time = time.time()
    union_loader = zip(*dataloaders)
    pbar = tqdm(union_loader, total=length, unit="batch")
    nd = len(config.datasets)
    bs = config.batch_size
    for batch_idx, datas in enumerate(pbar):
        if config.max_time_seconds is not None and time.time() - start_time > config.max_time_seconds:
            pbar.close()
            break

        keys = datas[0][0].keys()
        component_inputs = {}
        for key in keys:
            component_inputs[key] = torch.stack([data[0][key] for data in datas])
        component_inputs = _move_component_inputs(component_inputs, device, nd, bs, config.seq_len)
        targets = torch.stack([data[1] for data in datas])
        target = targets.reshape(nd * bs, config.seq_len, -1).to(device, non_blocking=True)

        if config.gt_components:
            pred = _inference_gt(main_model, component_inputs, target, config.window_size)
        else:
            pred = _inference(branch_predictor, icache_predictor, dcache_predictor, main_model, component_inputs, config.window_size)

        for i in range(nd):
            for j in range(bs):
                idx = i * bs + j
                ws = config.window_size

                fetch_t = target[idx, ..., ws:, 0]
                fetch_true_sum[i] += fetch_t.sum().item()
                fetch_pred_sum[i] += (pred["fetch_cycle_avg"][idx] * pred["eff_len"]).item()

                branch_pred = pred["branch_mispred"][idx, ..., ws:]
                branch_target = target[idx, ..., ws:, 2]
                is_control = target[idx, ..., ws:, 5].bool()
                if is_control.any():
                    branch_correct[i] += (branch_pred[is_control] == branch_target[is_control]).sum().item()
                    branch_total[i] += is_control.sum().item()

                icache_pred = pred["icache_hit"][idx, ..., ws:, :]
                icache_target = target[idx, ..., ws:, 3]
                icache_correct[i] += (icache_pred.argmax(-1).eq(icache_target)).sum().item()
                icache_total[i] += icache_target.numel()

                dcache_pred = pred["dcache_hit"][idx, ..., ws:, :]
                dcache_target = target[idx, ..., ws:, 4]
                is_mem_ref = target[idx, ..., ws:, 6].bool()
                if is_mem_ref.any():
                    dcache_correct[i] += (dcache_pred[is_mem_ref].argmax(-1).eq(dcache_target[is_mem_ref])).sum().item()
                    dcache_total[i] += is_mem_ref.sum().item()

        if batch_idx % 5 == 0:
            errors = [abs((fetch_pred_sum[i] - fetch_true_sum[i]) / fetch_true_sum[i]) if fetch_true_sum[i] > 0 else 0.0
                      for i in range(nd)]
            max_error = max(errors) if errors else 0.0
            pbar.set_postfix({"max_error": f"{max_error:.2%}"})

    print("\n=== Evaluation Results ===")
    print("\nFetch Cycle Count Error:")
    for i, dataset_path in enumerate(config.datasets):
        error = (fetch_pred_sum[i] - fetch_true_sum[i]) / fetch_true_sum[i] if fetch_true_sum[i] > 0 else 0.0
        print(f"  {dataset_path}: {error:+.2%}")

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
    parser.add_argument("--dataset-file", type=str, default="datasets.txt")
    parser.add_argument("--eval-data", type=int, nargs="+", default=[])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--max-time", type=float, default=None, help="Maximum evaluation time in seconds")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gt-components", action="store_true", help="Use ground truth for branch/icache/dcache inputs")

    args = parser.parse_args()

    all_datasets = load_datasets(args.dataset_file)
    if args.eval_data:
        eval_datasets = [all_datasets[i] for i in args.eval_data]
    else:
        eval_datasets = all_datasets

    config = EvalConfig(
        datasets=eval_datasets,
        device=args.device,
        load_state_file=args.model,
        max_time_seconds=args.max_time,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        gt_components=args.gt_components,
    )
    eval(config)


if __name__ == "__main__":
    main()
