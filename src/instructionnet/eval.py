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


@torch.no_grad()
def eval(config: EvalConfig):
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
    for dataset_path in config.datasets:
        print(f"Starting evaluating {dataset_path}")
        dataset = TAODataset(dataset_path)
        dataloader = DataLoader(
            dataset,
            batch_sampler=OverlappingSampler(dataset, config.batch_size, config.window_size),
            collate_fn=collate_fn,
            num_workers=12,
            pin_memory=True
        )

        true_cycles = 0.0
        pred_cycles = 0.0

        pbar = tqdm(dataloader, total=len(dataloader), unit="batch")
        for batch_idx, (input, target) in enumerate(pbar):
            input: torch.Tensor = input.to(device, non_blocking=True)
            target: torch.Tensor = target.to(device, non_blocking=True)
            pred = model(input)
            fetch_cycle_pred = pred["fetch_cycle"][..., config.window_size:]
            fetch_cycle_target = target[..., config.window_size:, 0]
            true_cycles += torch.sum(fetch_cycle_target).item()
            pred_cycles += torch.sum(fetch_cycle_pred).item()

            if batch_idx % 50 == 0:
                error = abs(pred_cycles - true_cycles) / true_cycles
                pbar.set_postfix({
                    "error": f"{error:.2%}",
                })

        error = abs(pred_cycles - true_cycles) / true_cycles
        print(f"Error: {error:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=lambda s: s.lower(), choices=["tao", "inet"], default="tao")
    parser.add_argument("--dataset", type=str, nargs="+") # Support multi datasets
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="")

    args = parser.parse_args()

    config = EvalConfig(
        datasets=args.dataset,
        name=args.name,
        device=args.device,
        load_state_file=args.model
    )
    eval(config)


if __name__ == "__main__":
    main()
