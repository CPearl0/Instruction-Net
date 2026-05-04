from __future__ import annotations
from src.instructionnet.instructionnet_model import (
    InstructionNet, BranchPredictor, ICachePredictor, DCachePredictor, build_main_input,
)
from src.instructionnet.dataset import TAODataset, OverlappingSampler, collate_fn
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    hidden_dim: int = 512

    epochs: int = 16
    lr: float = 1e-3
    cycle_loss_weight: float = 1
    seq_len: int = 1024
    batch_size: int = 1
    window_size: int = 8
    max_grad_norm: float = 10.0
    device: str = "cpu"

    load_state_file: str = ""


class ComponentLoss(nn.Module):
    def __init__(self, loss_start, device):
        super().__init__()
        self.loss_start = loss_start
        self.device = torch.device(device)
        d = self.device
        self.branch_ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.5, 8.0], device=d))
        self.icache_ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 8.0], device=d))
        self.dcache_ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 7.0, 17.0], device=d))

    def forward(self, branch_logits, icache_logits, dcache_logits, target):
        s = self.loss_start

        branch_target = target[..., s:, 2].float()
        icache_target = target[..., s:, 3].long()
        dcache_target = target[..., s:, 4].long()
        is_control = target[..., s:, 5].bool()
        is_mem_ref = target[..., s:, 6].bool()

        if is_control.any():
            branch_loss = self.branch_ce(
                rearrange(branch_logits[..., s:, :][is_control], "... c -> (...) c"),
                rearrange(branch_target[is_control].long(), "... -> (...)"))
        else:
            branch_loss = torch.tensor(0.0, device=self.device)

        icache_loss = self.icache_ce(
            rearrange(icache_logits[..., s:, :], "... c -> (...) c"),
            rearrange(icache_target, "... -> (...)"))

        if is_mem_ref.any():
            dcache_loss = self.dcache_ce(
                rearrange(dcache_logits[..., s:, :][is_mem_ref], "... c -> (...) c"),
                rearrange(dcache_target[is_mem_ref], "... -> (...)"))
        else:
            dcache_loss = torch.tensor(0.0, device=self.device)

        loss_dict = {
            "branch": branch_loss,
            "icache": icache_loss,
            "dcache": dcache_loss,
            "total": branch_loss + icache_loss + dcache_loss,
        }
        return loss_dict


class LatencyLoss(nn.Module):
    def __init__(self, loss_start, device):
        super().__init__()
        self.loss_start = loss_start
        self.device = torch.device(device)
        self.huber = nn.HuberLoss()

    def forward(self, pred, target):
        s = self.loss_start
        target_avg = target[..., s:, 0].float().mean(dim=-1)  # (batch,)
        pred_avg = pred["fetch_cycle_avg"]  # (batch,)
        fetch_loss = self.huber(pred_avg, target_avg)
        return {
            "fetch": fetch_loss,
            "total": fetch_loss,
        }


def _stack_component_inputs(datas):
    keys = datas[0][0].keys()
    stacked = {}
    for key in keys:
        stacked[key] = torch.stack([data[0][key] for data in datas])
    return stacked


def _make_lr_lambda(warmup_steps, total_steps):
    return lambda step: min(1.0, step / warmup_steps) * \
        0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))


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

        self.val_dataset_paths = config.val_datasets
        self.test_dataset_paths = config.test_datasets
        self.val_dataloaders, self.val_length = self._make_eval_dataloaders(config.val_datasets)
        self.test_dataloaders, self.test_length = self._make_eval_dataloaders(config.test_datasets)
        self.best_val_max_error = float('inf')
        self.global_step = 0

        self.device = torch.device(config.device)

        # Component models
        self.branch_predictor = BranchPredictor().to(self.device)
        self.icache_predictor = ICachePredictor().to(self.device)
        self.dcache_predictor = DCachePredictor().to(self.device)
        # Main model
        self.main_model = InstructionNet(config.hidden_dim).to(self.device)

        comp_params = (list(self.branch_predictor.parameters()) +
                       list(self.icache_predictor.parameters()) +
                       list(self.dcache_predictor.parameters()))
        self.comp_optimizer = torch.optim.AdamW(comp_params, lr=config.lr, weight_decay=0.05)
        self.main_optimizer = torch.optim.AdamW(self.main_model.parameters(), lr=config.lr, weight_decay=0.05)

        warmup_steps = self.length
        total_steps = config.epochs * self.length
        lr_lambda = _make_lr_lambda(warmup_steps, total_steps)
        self.comp_scheduler = torch.optim.lr_scheduler.LambdaLR(self.comp_optimizer, lr_lambda=lr_lambda)
        self.main_scheduler = torch.optim.lr_scheduler.LambdaLR(self.main_optimizer, lr_lambda=lr_lambda)

        if config.load_state_file:
            self.load_checkpoint(config.load_state_file)

        self.comp_loss = ComponentLoss(config.window_size, config.device)
        self.latency_loss = LatencyLoss(config.window_size, config.device)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f"logs/inet_{timestamp}")

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

    def _move_component_inputs(self, component_inputs, nd, bs):
        result = {}
        for key, val in component_inputs.items():
            val = val.reshape(nd * bs, self.config.seq_len, -1)
            result[key] = val.to(self.device, non_blocking=True)
        return result

    @torch.no_grad()
    def _inference(self, component_inputs):
        branch_logits = self.branch_predictor(component_inputs["branch_hist"])
        icache_logits = self.icache_predictor(component_inputs["icache_hist"])
        dcache_logits = self.dcache_predictor(component_inputs["dcache_hist"], component_inputs["page_hist"])

        branch_onehot = F.one_hot(branch_logits.argmax(-1), 3).float()
        icache_onehot = F.one_hot(icache_logits.argmax(-1), 3).float()
        dcache_onehot = F.one_hot(dcache_logits.argmax(-1), 3).float()

        main_input = build_main_input(
            component_inputs["type_reg_flags"], branch_onehot, icache_onehot, dcache_onehot)
        main_pred = self.main_model(main_input, self.config.window_size)

        return {
            **main_pred,
            "branch_mispred": branch_logits.argmax(-1),
            "icache_hit": F.softmax(icache_logits, dim=-1),
            "dcache_hit": F.softmax(dcache_logits, dim=-1),
        }

    @torch.no_grad()
    def _eval_on_dataloaders(self, dataloaders, length, max_time_seconds, desc):
        self.branch_predictor.eval()
        self.icache_predictor.eval()
        self.dcache_predictor.eval()
        self.main_model.eval()

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

            component_inputs = _stack_component_inputs(datas)
            targets = torch.stack([data[1] for data in datas])
            component_inputs = self._move_component_inputs(component_inputs, nd, bs)
            target = targets.reshape(nd * bs, self.config.seq_len, -1).to(self.device, non_blocking=True)
            pred = self._inference(component_inputs)

            for i in range(nd):
                for j in range(bs):
                    idx = i * bs + j
                    fetch_target = target[idx, ..., self.config.window_size:, 0]
                    true_cycles[i] += torch.sum(fetch_target).item()
                    pred_cycles[i] += (pred["fetch_cycle_avg"][idx] * pred["eff_len"]).item()

            if batch_idx % 5 == 0:
                errors = [abs((pred_cycles[i] - true_cycles[i]) / true_cycles[i]) if true_cycles[i] > 0 else 0.0
                          for i in range(nd)]
                pbar.set_postfix({"max_error": f"{max(errors):.2%}"})

        self.branch_predictor.train()
        self.icache_predictor.train()
        self.dcache_predictor.train()
        self.main_model.train()

        errors = [abs((pred_cycles[i] - true_cycles[i]) / true_cycles[i]) if true_cycles[i] > 0 else 0.0
                  for i in range(len(dataloaders))]
        return max(errors) if errors else float('inf'), errors

    def save_checkpoint(self, file: str = ""):
        if not file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file = f"model/inet-{timestamp}.model"
        checkpoint = {
            "branch_predictor": self.branch_predictor.state_dict(),
            "icache_predictor": self.icache_predictor.state_dict(),
            "dcache_predictor": self.dcache_predictor.state_dict(),
            "main_model": self.main_model.state_dict(),
            "comp_optimizer": self.comp_optimizer.state_dict(),
            "main_optimizer": self.main_optimizer.state_dict(),
            "comp_scheduler": self.comp_scheduler.state_dict(),
            "main_scheduler": self.main_scheduler.state_dict(),
        }
        torch.save(checkpoint, file)
        print(f"Model saved to {file}")

        if self.val_dataloaders or self.test_dataloaders:
            val_max_error, test_max_error = self.eval_quick()
            if val_max_error < self.best_val_max_error:
                self.best_val_max_error = val_max_error
                best_path = "model/inet-best.model"
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
        self.branch_predictor.load_state_dict(checkpoint["branch_predictor"])
        self.icache_predictor.load_state_dict(checkpoint["icache_predictor"])
        self.dcache_predictor.load_state_dict(checkpoint["dcache_predictor"])
        self.main_model.load_state_dict(checkpoint["main_model"])
        self.comp_optimizer.load_state_dict(checkpoint["comp_optimizer"])
        self.main_optimizer.load_state_dict(checkpoint["main_optimizer"])
        if "comp_scheduler" in checkpoint:
            self.comp_scheduler.load_state_dict(checkpoint["comp_scheduler"])
            self.main_scheduler.load_state_dict(checkpoint["main_scheduler"])
        print(f"Model loaded from {file}")

    def train(self):
        print("Starting training...")
        Path("model").mkdir(parents=True, exist_ok=True)
        self.branch_predictor.train()
        self.icache_predictor.train()
        self.dcache_predictor.train()
        self.main_model.train()

        for epoch in range(self.config.epochs):
            union_loader = zip(*self.dataloaders)
            pbar = tqdm(union_loader, total=self.length, unit="batch", desc=f"Epoch {epoch + 1} / {self.config.epochs}")
            for batch_idx, datas in enumerate(pbar):
                global_idx = epoch * self.length + batch_idx
                self.global_step = global_idx

                component_inputs = _stack_component_inputs(datas)
                targets = torch.stack([data[1] for data in datas])
                nd = len(datas)
                bs = component_inputs["branch_hist"].size(1) // self.config.seq_len
                component_inputs = self._move_component_inputs(component_inputs, nd, bs)
                target = targets.reshape(nd * bs, self.config.seq_len, -1).to(self.device, non_blocking=True)

                # --- Train component models ---
                branch_logits = self.branch_predictor(component_inputs["branch_hist"])
                icache_logits = self.icache_predictor(component_inputs["icache_hist"])
                dcache_logits = self.dcache_predictor(component_inputs["dcache_hist"], component_inputs["page_hist"])

                comp_loss = self.comp_loss(branch_logits, icache_logits, dcache_logits, target)
                self.comp_optimizer.zero_grad()
                comp_loss["total"].backward()
                comp_grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.branch_predictor.parameters()) +
                    list(self.icache_predictor.parameters()) +
                    list(self.dcache_predictor.parameters()),
                    self.config.max_grad_norm)
                self.comp_optimizer.step()
                self.comp_scheduler.step()

                # --- Train main model with GT ---
                with torch.no_grad():
                    branch_onehot = F.one_hot(target[..., 2].long(), num_classes=3).float()
                    icache_onehot = F.one_hot(target[..., 3].long(), num_classes=3).float()
                    dcache_onehot = F.one_hot(target[..., 4].long().clamp(max=2), num_classes=3).float()
                main_input = build_main_input(
                    component_inputs["type_reg_flags"], branch_onehot, icache_onehot, dcache_onehot)

                main_pred = self.main_model(main_input, self.config.window_size)
                main_loss = self.latency_loss(main_pred, target)
                self.main_optimizer.zero_grad()
                main_loss["total"].backward()
                main_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.main_model.parameters(), self.config.max_grad_norm)
                self.main_optimizer.step()
                self.main_scheduler.step()

                if batch_idx % 50 == 0:
                    total_loss = comp_loss["total"].item() + main_loss["total"].item()
                    pbar.set_postfix({"loss": f"{total_loss:.3g}"})
                    self.writer.add_scalar("train/comp_loss_total", comp_loss["total"].item(), global_idx)
                    self.writer.add_scalar("train/comp_loss_branch", comp_loss["branch"].item(), global_idx)
                    self.writer.add_scalar("train/comp_loss_icache", comp_loss["icache"].item(), global_idx)
                    self.writer.add_scalar("train/comp_loss_dcache", comp_loss["dcache"].item(), global_idx)
                    self.writer.add_scalar("train/main_loss_total", main_loss["total"].item(), global_idx)
                    self.writer.add_scalar("train/main_loss_fetch", main_loss["fetch"].item(), global_idx)
                    self.writer.add_scalar("train/comp_grad_norm", comp_grad_norm, global_idx)
                    self.writer.add_scalar("train/main_grad_norm", main_grad_norm, global_idx)
                    current_lr = self.comp_optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/learning_rate", current_lr, global_idx)

                if batch_idx % 2500 == 0 and global_idx > 0:
                    self.save_checkpoint("model/inet-latest.model")
            self.save_checkpoint(f"model/inet-epoch{epoch}.model")

        self.save_checkpoint()


def load_datasets(path="datasets.txt"):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
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
