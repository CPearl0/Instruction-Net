from __future__ import annotations
from src.instructionnet.tao_model import TAOModel
from src.instructionnet.instructionnet_model import InstructionNet
from src.instructionnet.dataset import TAODataset, OverlappingSampler, collate_fn
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from tqdm import tqdm
import argparse
import datetime
from pathlib import Path

@dataclass
class TrainConfig:
    datasets: list[str]
    name: str

    hidden_dim: int = 1024

    epochs: int = 16
    lr: float = 1e-5
    cycle_loss_weight: float = 1
    batch_size: int = 1024
    window_size: int = 16
    max_grad_norm: float = 10.0
    device: str = "cpu"

    load_state_file: str = ""


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, true):
        return self.mse(torch.log1p(pred), torch.log1p(true))


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function.

    Tasks:
    - fetch_cycle_class: CrossEntropy Loss (11 classes: 0-9 for cycles 1-10, 10 for 10+)
    - fetch_cycle_regression: Huber Loss (regression, only for samples with true value >= 11)
    - exec_cycle_class: CrossEntropy Loss (11 classes)
    - exec_cycle_regression: Huber Loss (regression, only for samples with true value >= 11)
    - branch_mispredict: BCE Loss (binary)
    - icache_hit: CrossEntropy Loss (3 classes: L1/L2/Memory)
    - dcache_hit: CrossEntropy Loss (3 classes: L1/L2/Memory)
    """
    def __init__(self, weights: dict[str, float | torch.Tensor], loss_start, device):
        super().__init__()
        self.weights = weights
        self.device = torch.device(device)

        self.loss_start = loss_start
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.cycle_ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # fetch_cycle
        fetch_cycle_class_logits = pred["fetch_cycle_class_logits"][..., self.loss_start:, :]
        fetch_cycle_regression = pred["fetch_cycle_regression"][..., self.loss_start:]

        # exec_cycle
        exec_cycle_class_logits = pred["exec_cycle_class_logits"][..., self.loss_start:, :]
        exec_cycle_regression = pred["exec_cycle_regression"][..., self.loss_start:]

        # Other tasks
        branch_mispredict_logits = pred["branch_mispred_logits"][..., self.loss_start:]
        icache_hit_logits = pred["icache_hit_logits"][..., self.loss_start:, :]
        dcache_hit_logits = pred["dcache_hit_logits"][..., self.loss_start:, :]

        # Labels
        fetch_cycle_target = target[..., self.loss_start:, 0].float()
        exec_cycle_target = target[..., self.loss_start:, 1].float()
        branch_mispredict_target = target[..., self.loss_start:, 2].float()
        icache_hit_target = target[..., self.loss_start:, 3].long()
        dcache_hit_target = target[..., self.loss_start:, 4].long()
        is_control = target[..., self.loss_start:, 5].bool()
        is_mem_ref = target[..., self.loss_start:, 6].bool()

        # fetch_cycle class target: min(cycle-1, 10)
        fetch_cycle_class_target = torch.clamp(fetch_cycle_target.long() - 1, min=0, max=10)

        # fetch_cycle class loss: compute for all samples
        fetch_cycle_class_loss = self.cycle_ce_loss(
            rearrange(fetch_cycle_class_logits, "... c -> (...) c"),
            rearrange(fetch_cycle_class_target, "... -> (...)")
        )

        # fetch_cycle regression loss: only for samples with true value >= 11
        fetch_high_cycle_mask = fetch_cycle_target >= 11
        if fetch_high_cycle_mask.any():
            fetch_cycle_regression_pred = fetch_cycle_regression[fetch_high_cycle_mask]
            fetch_cycle_regression_target = fetch_cycle_target[fetch_high_cycle_mask] / 100.0
            fetch_cycle_regression_loss = self.mse_loss(fetch_cycle_regression_pred, fetch_cycle_regression_target)
        else:
            fetch_cycle_regression_loss = torch.tensor(0.0, device=self.device)

        # exec_cycle class target: min(cycle-1, 10)
        exec_cycle_class_target = torch.clamp(exec_cycle_target.long() - 1, min=0, max=10)

        # exec_cycle class loss: compute for all samples
        exec_cycle_class_loss = self.cycle_ce_loss(
            rearrange(exec_cycle_class_logits, "... c -> (...) c"),
            rearrange(exec_cycle_class_target, "... -> (...)")
        )

        # exec_cycle regression loss: only for samples with true value >= 11
        exec_high_cycle_mask = exec_cycle_target >= 11
        if exec_high_cycle_mask.any():
            exec_cycle_regression_pred = exec_cycle_regression[exec_high_cycle_mask]
            exec_cycle_regression_target = exec_cycle_target[exec_high_cycle_mask] / 100.0
            exec_cycle_regression_loss = self.mse_loss(exec_cycle_regression_pred, exec_cycle_regression_target)
        else:
            exec_cycle_regression_loss = torch.tensor(0.0, device=self.device)

        # Other task losses
        if is_control.any():
            branch_mispredict_loss = self.bce_loss(
                branch_mispredict_logits[is_control],
                branch_mispredict_target[is_control]
            )
        else:
            branch_mispredict_loss = torch.tensor(0.0, device=self.device)
        icache_hit_loss = self.ce_loss(
            rearrange(icache_hit_logits, "... c -> (...) c"),
            rearrange(icache_hit_target, "... -> (...)")
        )
        if is_mem_ref.any():
            dcache_hit_loss = self.ce_loss(
                rearrange(dcache_hit_logits[is_mem_ref], "... c -> (...) c"),
                rearrange(dcache_hit_target[is_mem_ref], "... -> (...)")
            )
        else:
            dcache_hit_loss = torch.tensor(0.0, device=self.device)

        loss_dict: dict[str, torch.Tensor] = {
            "fetch_cycle_class": fetch_cycle_class_loss,
            "fetch_cycle_regression": fetch_cycle_regression_loss,
            "exec_cycle_class": exec_cycle_class_loss,
            "exec_cycle_regression": exec_cycle_regression_loss,
            "branch_mispredict": branch_mispredict_loss,
            "icache_hit": icache_hit_loss,
            "dcache_hit": dcache_hit_loss,
        }
        total_loss = torch.tensor(0.0, device=self.device)
        for name, loss in loss_dict.items():
            total_loss += self.weights[name] * loss
        loss_dict["total"] = total_loss
        return loss_dict


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.datasets = [TAODataset(f) for f in config.datasets]
        self.dataloaders = [DataLoader(
            dataset,
            batch_sampler=OverlappingSampler(dataset, config.batch_size, config.window_size, True),
            collate_fn=collate_fn,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True
        ) for dataset in self.datasets]
        self.length = min(len(dataloader) for dataloader in self.dataloaders)

        self.device = torch.device(config.device)
        if self.config.name == "tao":
            self.model = TAOModel(config.hidden_dim).to(self.device)
        else:
            self.model = InstructionNet(config.hidden_dim).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            config.lr,
            epochs=self.config.epochs,
            steps_per_epoch=self.length
        )
        if config.load_state_file:
            self.load_checkpoint(config.load_state_file)
        self.loss = MultiTaskLoss({
            "fetch_cycle_class": config.cycle_loss_weight,
            "fetch_cycle_regression": 5 * config.cycle_loss_weight,
            "exec_cycle_class": config.cycle_loss_weight,
            "exec_cycle_regression": 5 * config.cycle_loss_weight,
            "branch_mispredict": 10.0,
            "icache_hit": 10.0,
            "dcache_hit": 10.0,
        }, config.window_size, config.device)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f"logs/{self.config.name}_{timestamp}")

    def save_checkpoint(self, file: str = ""):
        if not file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file = f"model/{self.config.name}-{timestamp}.model"
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, file)
        print(f"Model saved to {file}")

    def load_checkpoint(self, file: str):
        checkpoint = torch.load(file)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Model loaded from {file}")

    def train(self):
        print("Starting training...")
        Path("model").mkdir(parents=True, exist_ok=True)
        self.model.train()
        for epoch in range(self.config.epochs):
            union_loader = zip(*self.dataloaders)
            pbar = tqdm(union_loader, total=self.length, unit="batch", desc=f"Epoch {epoch + 1} / {self.config.epochs}")
            for batch_idx, datas in enumerate(pbar):
                global_idx = epoch * self.length + batch_idx

                input = torch.stack([data[0] for data in datas]).to(self.device, non_blocking=True) # type: ignore
                target = torch.stack([data[1] for data in datas]).to(self.device, non_blocking=True) # type: ignore

                pred = self.model(input)
                loss = self.loss(pred, target)

                self.optimizer.zero_grad()
                loss["total"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                if batch_idx % 50 == 0:
                    pbar.set_postfix({
                        "loss": f"{loss['total'].item():.3g}",
                    })
                    self.writer.add_scalar("train/loss_total", loss["total"].item(), global_idx)
                    self.writer.add_scalar("train/loss_fetch_cycle_class", loss["fetch_cycle_class"].item(), global_idx)
                    self.writer.add_scalar("train/loss_fetch_cycle_regression", loss["fetch_cycle_regression"].item(), global_idx)
                    self.writer.add_scalar("train/loss_exec_cycle_class", loss["exec_cycle_class"].item(), global_idx)
                    self.writer.add_scalar("train/loss_exec_cycle_regression", loss["exec_cycle_regression"].item(), global_idx)
                    self.writer.add_scalar("train/loss_branch_mispred", loss["branch_mispredict"].item(), global_idx)
                    self.writer.add_scalar("train/loss_icache_hit", loss["icache_hit"].item(), global_idx)
                    self.writer.add_scalar("train/loss_dcache_hit", loss["dcache_hit"].item(), global_idx)
                    self.writer.add_scalar("train/grad_norm", grad_norm, global_idx)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/learning_rate", current_lr, global_idx)

                if batch_idx % 2500 == 0:
                    self.save_checkpoint(f"model/{self.config.name}-latest.model")
            self.save_checkpoint(f"model/{self.config.name}-epoch{epoch}.model")

        # Save the final model
        self.save_checkpoint()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=lambda s: s.lower(), choices=["tao", "inet"], default="tao")
    parser.add_argument("--dataset", type=str, nargs="+") # Support multi datasets
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--cycle-loss-weight", type=float, default=0.1)

    args = parser.parse_args()

    config = TrainConfig(
        datasets=args.dataset,
        name=args.name,
        device=args.device,
        load_state_file=args.model,
        epochs=args.epochs,
        cycle_loss_weight=args.cycle_loss_weight,
    )
    trainer = Trainer(config)

    trainer.train()


if __name__ == "__main__":
    main()
