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

    hidden_dim: int = 512

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
    dataloaders = [DataLoader(
        dataset,
        batch_sampler=OverlappingSampler(dataset, config.batch_size, config.window_size, True),
        collate_fn=collate_fn,
        num_workers=12,
        pin_memory=True
    ) for dataset in datasets]
    length = min(len(dataloader) for dataloader in dataloaders)

    true_cycles = [0.0] * len(config.datasets)
    pred_cycles = [0.0] * len(config.datasets)

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
        fetch_cycle_pred = pred["fetch_cycle"][..., config.window_size:]
        fetch_cycle_target = target[..., config.window_size:, 0]

        for i in range(len(config.datasets)):
            true_cycles[i] += torch.sum(fetch_cycle_target[i]).item()
            pred_cycles[i] += torch.sum(fetch_cycle_pred[i]).item()

        if batch_idx % 50 == 0:
            errors = [(pred_cycles[i] - true_cycles[i]) / true_cycles[i] if true_cycles[i] > 0 else 0.0
                      for i in range(len(config.datasets))]
            avg_error = sum(errors) / len(errors) if errors else 0.0
            pbar.set_postfix({"avg_error": f"{avg_error:+.2%}"})

    print("\nEvaluation Results:")
    for i, dataset_path in enumerate(config.datasets):
        error = (pred_cycles[i] - true_cycles[i]) / true_cycles[i] if true_cycles[i] > 0 else 0.0
        print(f"{dataset_path}: {error:+.2%}")


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
