from __future__ import annotations
from src.instructionnet.tao_model import TAOModel
from src.instructionnet.instructionnet_model import InstructionNet
from src.instructionnet.dataset import TAODataset, OverlappingSampler, collate_fn
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from tqdm import tqdm
import argparse
import datetime
import time
from pathlib import Path

@dataclass
class TrainConfig:
    datasets: list[str]
    val_datasets: list[str]
    test_datasets: list[str]
    name: str

    hidden_dim: int = 768

    epochs: int = 16
    lr: float = 1e-3
    cycle_loss_weight: float = 1
    seq_len: int = 1024
    batch_size: int = 1
    window_size: int = 8
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
    - exec_cycle_class: CrossEntropy Loss (21 classes)
    - exec_cycle_regression: Huber Loss (regression, only for samples with true value >= 21)
    - branch_predict: CrossEntropy Loss (3 classes: correct/dir_wrong/target_wrong)
    - icache_hit: CrossEntropy Loss (3 classes: L1/L2/Memory)
    - dcache_hit: CrossEntropy Loss (3 classes: L1/L2/Memory)
    """
    def __init__(self, weights: dict[str, float | torch.Tensor], loss_start, device):
        super().__init__()
        self.weights = weights
        self.device = torch.device(device)

        self.loss_start = loss_start
        self.mse_loss = nn.MSELoss()
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
        branch_mispredict_logits = pred["branch_mispred_logits"][..., self.loss_start:, :]
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

        # exec_cycle class target: min(cycle-1, 20)
        exec_cycle_class_target = torch.clamp(exec_cycle_target.long() - 1, min=0, max=20)

        # exec_cycle class loss: compute for all samples
        exec_cycle_class_loss = self.cycle_ce_loss(
            rearrange(exec_cycle_class_logits, "... c -> (...) c"),
            rearrange(exec_cycle_class_target, "... -> (...)")
        )

        # exec_cycle regression loss: only for samples with true value >= 21
        exec_high_cycle_mask = exec_cycle_target >= 21
        if exec_high_cycle_mask.any():
            exec_cycle_regression_pred = exec_cycle_regression[exec_high_cycle_mask]
            exec_cycle_regression_target = exec_cycle_target[exec_high_cycle_mask] / 100.0
            exec_cycle_regression_loss = self.mse_loss(exec_cycle_regression_pred, exec_cycle_regression_target)
        else:
            exec_cycle_regression_loss = torch.tensor(0.0, device=self.device)

        # Other task losses
        if is_control.any():
            branch_mispredict_loss = self.ce_loss(
                rearrange(branch_mispredict_logits[is_control], "... c -> (...) c"),
                rearrange(branch_mispredict_target[is_control].long(), "... -> (...)")
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
            batch_sampler=OverlappingSampler(dataset, config.seq_len, config.window_size, config.batch_size, True),
            collate_fn=collate_fn,
            num_workers=12,
            pin_memory=True,
            persistent_workers=True
        ) for dataset in self.datasets]
        self.length = min(len(dataloader) for dataloader in self.dataloaders)

        # Val/Test setup
        self.val_dataset_paths = config.val_datasets
        self.test_dataset_paths = config.test_datasets
        self.val_dataloaders, self.val_length = self._make_eval_dataloaders(config.val_datasets)
        self.test_dataloaders, self.test_length = self._make_eval_dataloaders(config.test_datasets)
        self.best_val_max_error = float('inf')
        self.global_step = 0

        self.device = torch.device(config.device)
        if self.config.name == "tao":
            self.model = TAOModel(config.hidden_dim).to(self.device)
        else:
            self.model = InstructionNet(config.hidden_dim).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config.lr, weight_decay=0.05)
        warmup_steps = self.length
        total_steps = self.config.epochs * self.length
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps) *
                0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))
        )
        if config.load_state_file:
            self.load_checkpoint(config.load_state_file)
        self.loss = MultiTaskLoss({
            "fetch_cycle_class": config.cycle_loss_weight,
            "fetch_cycle_regression": 40 * config.cycle_loss_weight,
            "exec_cycle_class": 0.4 * config.cycle_loss_weight,
            "exec_cycle_regression": 16 * config.cycle_loss_weight,
            "branch_mispredict": 2.0,
            "icache_hit": 2.0,
            "dcache_hit": 2.0,
        }, config.window_size, config.device)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f"logs/{self.config.name}_{timestamp}")

    def _make_eval_dataloaders(self, dataset_paths):
        if not dataset_paths:
            return [], 0
        datasets = [TAODataset(f) for f in dataset_paths]
        num_workers = 0 if self.config.device.startswith("cuda") else 4
        dataloaders = [DataLoader(
            dataset,
            batch_sampler=OverlappingSampler(dataset, self.config.seq_len, self.config.window_size, self.config.batch_size, True),
            collate_fn=collate_fn,
            num_workers=num_workers,
        ) for dataset in datasets]
        return dataloaders, min(len(dl) for dl in dataloaders)

    @torch.no_grad()
    def _eval_on_dataloaders(self, dataloaders, length, max_time_seconds, desc):
        self.model.eval()
        true_cycles = [0.0] * len(dataloaders)
        pred_cycles = [0.0] * len(dataloaders)

        start = time.time()
        union_loader = zip(*dataloaders)
        pbar = tqdm(union_loader, total=length, unit="batch", desc=desc)
        nd = len(dataloaders)
        bs = self.config.batch_size
        for batch_idx, datas in enumerate(pbar):
            if time.time() - start > max_time_seconds:
                pbar.close()
                break

            inputs = torch.stack([data[0] for data in datas])
            targets = torch.stack([data[1] for data in datas])
            input = inputs.reshape(nd * bs, self.config.seq_len, -1).to(self.device, non_blocking=True)
            target = targets.reshape(nd * bs, self.config.seq_len, -1).to(self.device, non_blocking=True)
            pred = self.model(input)

            for i in range(nd):
                for j in range(bs):
                    idx = i * bs + j
                    fetch_pred = pred["fetch_cycle"][idx, ..., self.config.window_size:]
                    fetch_target = target[idx, ..., self.config.window_size:, 0]
                    true_cycles[i] += torch.sum(fetch_target).item()
                    pred_cycles[i] += torch.sum(fetch_pred).item()

            if batch_idx % 5 == 0:
                errors = [abs((pred_cycles[i] - true_cycles[i]) / true_cycles[i]) if true_cycles[i] > 0 else 0.0
                          for i in range(nd)]
                pbar.set_postfix({"max_error": f"{max(errors):.2%}"})

        self.model.train()
        errors = [abs((pred_cycles[i] - true_cycles[i]) / true_cycles[i]) if true_cycles[i] > 0 else 0.0
                  for i in range(len(dataloaders))]
        return max(errors) if errors else float('inf'), errors

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

        if self.val_dataloaders or self.test_dataloaders:
            val_max_error, test_max_error = self.eval_quick()
            if val_max_error < self.best_val_max_error:
                self.best_val_max_error = val_max_error
                best_path = f"model/{self.config.name}-best.model"
                torch.save(checkpoint, best_path)
                print(f"New best model! Val max error: {val_max_error:.2%} -> {best_path}")
            self.writer.add_scalar("eval/val_max_error", val_max_error, self.global_step)
            self.writer.add_scalar("eval/test_max_error", test_max_error, self.global_step)

    def eval_quick(self, max_time_seconds=30):
        val_max_error, test_max_error = float('inf'), float('inf')
        if self.val_dataloaders:
            val_max_error, val_errors = self._eval_on_dataloaders(
                self.val_dataloaders, self.val_length, max_time_seconds, "Val Eval")
            print(f"\nVal max abs error: {val_max_error:.2%}")
            for i, path in enumerate(self.val_dataset_paths):
                print(f"  {path}: {val_errors[i]:.2%}")
        if self.test_dataloaders:
            test_max_error, test_errors = self._eval_on_dataloaders(
                self.test_dataloaders, self.test_length, max_time_seconds, "Test Eval")
            print(f"\nTest max abs error: {test_max_error:.2%}")
            for i, path in enumerate(self.test_dataset_paths):
                print(f"  {path}: {test_errors[i]:.2%}")
        return val_max_error, test_max_error

    def load_checkpoint(self, file: str):
        checkpoint = torch.load(file)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
                self.global_step = global_idx

                inputs = torch.stack([data[0] for data in datas])
                targets = torch.stack([data[1] for data in datas])
                nd = inputs.size(0)
                bs = inputs.size(1) // self.config.seq_len
                input = inputs.reshape(nd * bs, self.config.seq_len, -1).to(self.device, non_blocking=True)
                target = targets.reshape(nd * bs, self.config.seq_len, -1).to(self.device, non_blocking=True)

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


def load_datasets(path="datasets.txt"):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=lambda s: s.lower(), choices=["tao", "inet"], default="inet")
    parser.add_argument("--dataset-file", type=str, default="datasets.txt")
    parser.add_argument("--train-data", type=int, nargs="+", default=[])
    parser.add_argument("--val-data", type=int, nargs="+", default=[])
    parser.add_argument("--test-data", type=int, nargs="+", default=[])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cycle-loss-weight", type=float, default=1.0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()

    all_datasets = load_datasets(args.dataset_file)
    if args.train_data:
        train_datasets = [all_datasets[i] for i in args.train_data]
    else:
        val_test_indices = set(args.val_data) | set(args.test_data)
        train_datasets = [d for i, d in enumerate(all_datasets) if i not in val_test_indices]
    val_datasets = [all_datasets[i] for i in args.val_data]
    test_datasets = [all_datasets[i] for i in args.test_data]
    print(f"Val data: {val_datasets}")
    print(f"Test data: {test_datasets}")
    print(f"Train data: {train_datasets}")

    config = TrainConfig(
        datasets=train_datasets,
        val_datasets=val_datasets,
        test_datasets=test_datasets,
        name=args.name,
        device=args.device,
        load_state_file=args.model,
        epochs=args.epochs,
        lr=args.lr,
        cycle_loss_weight=args.cycle_loss_weight,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    trainer = Trainer(config)

    trainer.train()


if __name__ == "__main__":
    main()
