# InstructionNet - ML-Assisted CPU Performance Modeling

## Project Overview

This project implements a machine learning approach to CPU performance modeling. Given instruction traces from the gem5 simulator (SPEC benchmarks), the model predicts the number of execution cycles for each instruction. This is formulated as a supervised learning task.

The architecture uses a decoupled design with **component models** (lightweight MLPs) for predicting microarchitectural events and a **main Transformer model** that consumes those predictions to estimate total fetch latency for a window of instructions.

## Architecture

### Overall Design

```
                         +-----------------+
  Branch History ------->| BranchPredictor |--- branch_pred (3-class)
                         |   (2-layer MLP) |
                         +-----------------+
                         +-----------------+
  ICache History ------->| ICachePredictor |--- icache_pred (3-class)
                         |   (2-layer MLP) |
                         +-----------------+
                         +-----------------+
  DCache History ------->| DCachePredictor |--- dcache_pred (3-class)
  Page History --------->|   (2-layer MLP) |
                         +-----------------+
                                |
                                v (one-hot encoded predictions)
  +-----------------------------------------------------------+
  |                    Main Model (InstructionNet)             |
  |                                                            |
  |  Input: type(1) + reg(64) + branch(3) + icache(3)        |
  |         + dcache(3) + flags(3) = 77 dims                  |
  |                                                            |
  |  InstructionEncoder -> Transformer (3 layers)             |
  |  -> Mean Pooling (positions [loss_start:])                 |
  |  -> MLP -> softplus -> total fetch cycles                  |
  +-----------------------------------------------------------+
```

### Training vs Inference

- **Training**: Component models and the main model are trained **separately** per batch. First, component models forward their inputs, compute CE losses, and update via their own optimizer. Then, the main model forwards with **ground truth** one-hot labels as input and updates via its own optimizer. The two sets of models have separate optimizers, schedulers, and gradient flows.
- **Inference**: Component models predict branch/icache/dcache results first, then their **argmax one-hot** predictions are fed to the main model.

### Component Models

#### BranchPredictor
- Input: `branch_hist` (32-bit bitmap, indicating taken/not-taken for last 32 branches)
- Architecture: Linear(32, 256) -> SiLU -> Linear(256, 256) -> SiLU -> Linear(256, 256) -> SiLU -> Linear(256, 3)
- Output: 3-class logits (0=correct, 1=direction wrong, 2=target wrong)
- Applied only to control flow instructions during loss computation

#### ICachePredictor
- Input: `same_icache_line_hist` (64-bit bitmap, indicating whether each of the last 64 fetch addresses shared the same cache line)
- Architecture: Linear(64, 256) -> SiLU -> Linear(256, 256) -> SiLU -> Linear(256, 256) -> SiLU -> Linear(256, 3)
- Output: 3-class logits (0=L1 hit, 1=L2, 2=beyond)

#### DCachePredictor
- Input: concatenation of `same_dcache_line_hist` (64-bit) + `same_page_hist` (64-bit) = 128 dims
- Architecture: Linear(128, 256) -> SiLU -> Linear(256, 256) -> SiLU -> Linear(256, 256) -> SiLU -> Linear(256, 3)
- Output: 3-class logits (0=L1 hit, 1=L2, 2=beyond)
- Applied only to memory reference instructions during loss computation

### Main Model

#### InstructionEncoder
Encodes raw instruction features into a dense representation (default: 512 dims).

| Feature | Input Dims | Encoding | Output Dims |
|---------|-----------|----------|-------------|
| Instruction type | 1 (token ID) | `nn.Embedding(157, 256)` | 256 |
| Register bitmap | 64 (32 int + 32 FP bits) | `nn.Linear(64, 192)` | 192 |
| Branch prediction | 3 (one-hot) | Direct pass-through | 3 |
| ICache prediction | 3 (one-hot) | Direct pass-through | 3 |
| DCache prediction | 3 (one-hot) | Direct pass-through | 3 |
| Flags | 3 (isControl, isCondCtrl, isMemRef) | `nn.Linear(3, 32)` | 32 |

Total concat dim = 256 + 192 + 3 + 3 + 3 + 32 = 489

Flow: Concat -> SiLU -> Linear(489, 512) -> LayerNorm -> SiLU -> output (512)

#### Transformer Backbone
- **Positional encoding**: RotaryEmbedding (dim=128, shared across all layers)
- **Layers**: 3x TransformerBlock
- Each block:
  - MultiHeadSelfAttention: 4 heads, window_size=128 (sliding window)
  - SwiGLU FFN: d_model=512, d_ff=1365
  - Pre-norm with RMSNorm
  - Dropout=0.2
- **Attention mask**: causal sliding window, each position attends to [max(0, i-128), i]

#### OutputHead (Window-Level Total Latency)
- Mean pooling over effective positions `[loss_start:]` -> LayerNorm -> SiLU -> Linear(512, 256) -> SiLU -> Linear(256, 1) -> softplus
- Output: single scalar = total fetch cycles for the window of 1016 effective instructions
- Rationale: per-instruction prediction failed because dcache miss effects propagate to subsequent instructions with variable delay. Window-level aggregation captures these effects without requiring per-instruction attribution.

## Dataset Format

### Binary File Layout
Each file: `[8-byte header] [N x 59-byte records]`

**Header** (`sequence_header`, 8 bytes):
```cpp
struct sequence_header {
    uint32_t seq_length;     // Number of instructions
    uint32_t reserved;
};
```

**Record** (`inst_record`, 59 bytes, packed):
```cpp
#pragma pack(push, 1)
struct inst_record {
    uint64_t pc;                        // Program counter
    uint8_t type;                       // Instruction type (token ID, vocab=157)
    uint32_t int_reg;                   // Integer register usage (32-bit bitmap)
    uint32_t fp_reg;                    // FP register usage (32-bit bitmap)
    uint32_t branch_hist;               // Branch history (32-bit bitmap)
    uint8_t isMispredicted;             // 0 or 1
    uint8_t branch_dir_wrong;           // 1 if direction prediction wrong
    uint8_t branch_target_wrong;        // 1 if target prediction wrong
    uint8_t isControl;                  // Control flow instruction
    uint8_t isCondCtrl;                 // Conditional control instruction
    uint8_t isMemRef;                   // Memory reference instruction
    uint64_t same_icache_line_hist;     // 64-bit: last 64 fetches same cache line?
    uint64_t same_dcache_line_hist;     // 64-bit: last 64 loads/stores same cache line?
    uint64_t same_page_hist;            // 64-bit: last 64 loads/stores same page?
    // Labels:
    uint16_t fetch_latency;             // Delta fetch latency
    uint16_t exec_latency;              // Delta exec latency
    uint8_t dcache_hit_level;           // 0=L1, 1=L2, 255=non-memory
    uint8_t icache_hit_level;           // 0=L1, 1=L2, 2=beyond
    uint8_t icache_hit;                 // 0 or 1 (redundant with hit_level)
    uint8_t dcache_hit;                 // 0 or 1 (redundant with hit_level)
};
#pragma pack(pop)
```

### Feature Engineering (`collate_fn`)

The `collate_fn` in `dataset.py` transforms raw records into:

**Component inputs** (dict):
- `branch_hist`: (batch, 32) float tensor - bit-unpacked branch history
- `icache_hist`: (batch, 64) float tensor - bit-unpacked icache line history
- `dcache_hist`: (batch, 64) float tensor - bit-unpacked dcache line history
- `page_hist`: (batch, 64) float tensor - bit-unpacked page history
- `type_reg_flags`: (batch, 68) float tensor - type(1) + int_reg(32) + fp_reg(32) + isControl(1) + isCondCtrl(1) + isMemRef(1)

**Ground truth** (tensor, 7 dims):
- [0] fetch_latency (int32)
- [1] exec_latency (int32)
- [2] branch_pred: 0=correct, 1=direction wrong, 2=target wrong (derived from isMispredicted, branch_target_wrong)
- [3] icache_hit_level (int32)
- [4] dcache_hit_level (int32, clamped to max 2)
- [5] isControl (int32)
- [6] isMemRef (int32)

## File Structure

```
InstructionNet/
  CLAUDE.md                    # Project instructions
  PROJECT.md                   # This file
  datasets.txt                 # Dataset file paths (21 entries)
  .gitignore
  logs/                        # TensorBoard logs
  data/                        # Binary dataset files (20 SPEC benchmarks)
  fine_data/                   # Filtered/merged dataset
  tools/
    dedup.cc                   # C++ tool: deduplicate/filter by PC diversity
    stats.cc                   # C++ tool: PC-level statistics
  src/
    instructionnet/
      __init__.py
      dataset.py               # TAODataset, collate_fn, OverlappingSampler
      instructionnet_model.py  # Component MLPs + main Transformer model
      train.py                 # Training script (Trainer class)
      eval.py                  # Evaluation script
      inspect_dataset.py       # CLI tool for dataset inspection
```

## Training

### Separate Training

Each batch performs two independent training steps:

1. **Component models step**: Forward branch/icache/dcache predictors, compute CE losses, backward, update component optimizer.
2. **Main model step**: Build main input using **ground truth** one-hot labels, forward main model, compute latency losses, backward, update main optimizer.

The two groups have separate AdamW optimizers and cosine schedulers (same lr and schedule).

### Loss Functions

**Component losses** (ComponentLoss, equal weight):

| Loss | Type | Scope |
|------|------|-------|
| branch | CrossEntropy (3 classes) | Only control instructions |
| icache | CrossEntropy (3 classes) | All samples |
| dcache | CrossEntropy (3 classes) | Only memory references |

**Main model loss** (LatencyLoss):

| Loss | Type |
|------|------|
| fetch | Huber loss on predicted total vs sum(fetch_latency[loss_start:]) |

The main model predicts a single scalar per window: total fetch cycles for the effective 1016 positions. The target is the sum of `fetch_latency` over those positions.

`loss_start` = `window_size` (default 8): first 8 positions in each sequence are excluded from loss computation (insufficient left context).

### Optimizer & Scheduler
- Two separate AdamW optimizers (component models + main model), same lr, weight_decay=0.05
- Cosine annealing with linear warmup for each
- Gradient clipping: max_norm=10.0

### Data Loading
- Multiple datasets loaded in parallel via joint zip of DataLoaders
- OverlappingSampler: seq_len=1024, overlap=8 (stride=1016)
- batch_size per dataset (default 1)

### Checkpointing
- Every 2500 steps: save latest checkpoint
- End of each epoch: save epoch checkpoint
- Best model tracked by validation max absolute cycle count error
- Checkpoint contains: branch_predictor, icache_predictor, dcache_predictor, main_model, comp_optimizer, main_optimizer, comp_scheduler, main_scheduler

### Usage
```bash
python -m src.instructionnet.train --dataset-file datasets.txt \
  --train-data 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --val-data 0 1 --test-data 2 3 \
  --device cuda --epochs 16 --lr 1e-3 --seq-len 1024 --batch-size 1
```

## Evaluation

### Metrics Tracked
- Fetch cycle count error: (sum(pred_window_totals) - sum(true_window_totals)) / sum(true) per dataset
- Branch prediction accuracy (3 classes, control instructions only)
- ICache hit level accuracy (3 classes)
- DCache hit level accuracy (3 classes, memory instructions only)

### Usage
```bash
# Standard evaluation
python -m src.instructionnet.eval --model path/to/model \
  --eval-data 0 1 2 --device cuda --max-time 60

# Evaluate main model with GT components (isolates main model performance)
python -m src.instructionnet.eval --model path/to/model \
  --eval-data 0 1 2 --device cuda --gt-components
```

## Key Design Decisions

1. **Decoupled architecture with separate training**: Component MLPs handle microarchitectural event prediction (branch, icache, dcache) independently, while the main Transformer focuses on latency prediction. They are trained with separate optimizers and backward passes - no gradient flows between them.

2. **Ground truth for main model training**: The main model receives ground truth labels for branch/icache/dcache predictions rather than component model outputs. This prevents error propagation during training.

3. **Window-level total prediction**: The main model predicts the total fetch cycles for a window of 1016 instructions (via mean pooling over Transformer outputs), rather than per-instruction latency. This avoids the need for per-instruction attribution of pipeline stall effects (e.g., dcache miss latency propagating to subsequent instructions with variable delay).

4. **Huber loss for cycle prediction**: Huber loss on predicted window total vs ground truth window total (sum of fetch_latency over effective positions).

5. **Sliding window attention**: Each instruction attends to at most 128 preceding instructions, providing local context while keeping memory usage manageable.

6. **Overlapping sequences**: Training sequences overlap by 8 positions, with the first 8 excluded from loss. This ensures every position has sufficient left context.

## Archived: Per-Instruction Latency Prediction

Before switching to window-level total prediction, the main model predicted per-instruction fetch_cycle and exec_cycle. This was abandoned because the model could not learn that dcache miss effects propagate to subsequent instructions (peak latency at ~5 instructions after the miss, mean=85 vs baseline=5).

**Output head** (`MultiTaskOutputHead`): LayerNorm -> SiLU -> Linear(512, 256) -> SiLU -> Linear(256, 2) -> softplus. Outputs: fetch_cycle and exec_cycle per position.

**Loss**: Huber loss per instruction, averaged over all effective positions.

## Archived: Classification+Regression Hybrid Scheme

The following scheme was used before switching to Huber loss. It can be restored if needed.

**Output head** (`MultiTaskOutputHead`): Linear(768, 256) -> SiLU -> Linear(256, **34**)

| Output | Indices | Description |
|--------|---------|-------------|
| fetch_cycle_class_logits | [0:11] | 11 classes (0-9 for cycles 1-10, 10 for >=11) |
| fetch_cycle_regression | [11] | Softplus regression (scaled *100, for cycles >=11) |
| exec_cycle_class_logits | [12:33] | 21 classes (0-19 for cycles 1-20, 20 for >=21) |
| exec_cycle_regression | [33] | Softplus regression (scaled *100, for cycles >=21) |

**Final cycle prediction**: if argmax class < threshold (10 for fetch, 20 for exec), prediction = class_index + 1; otherwise = regression_value * 100.

**Loss** (`LatencyLoss`): 4 components with weights:
- fetch_cycle_class: CrossEntropy, weight = 1.0 * cycle_loss_weight
- fetch_cycle_regression: MSE (target/100), weight = 40 * cycle_loss_weight, only for fetch >= 11
- exec_cycle_class: CrossEntropy, weight = 0.4 * cycle_loss_weight
- exec_cycle_regression: MSE (target/100), weight = 16 * cycle_loss_weight, only for exec >= 21

**Eval metrics** (additional): fetch/exec cycle classification accuracy, high-cycle regression error.

**Key code pattern**:
```python
# Output head out_linear2 = nn.Linear(256, 34)
fetch_cycle_class_logits = out[..., 0:11]
fetch_cycle_regression = F.softplus(out[..., 11])
exec_cycle_class_logits = out[..., 12:33]
exec_cycle_regression = F.softplus(out[..., 33])
fetch_cycle = torch.where(class_pred < 10, (class_pred + 1).float(), regression * 100)
exec_cycle = torch.where(class_pred < 20, (class_pred + 1).float(), regression * 100)
```

## Parameter Count Estimate

| Component | Parameters |
|-----------|-----------|
| BranchPredictor | ~200K |
| ICachePredictor | ~265K |
| DCachePredictor | ~330K |
| InstructionEncoder | ~0.6M |
| Transformer (3 layers) | ~10M |
| WindowTotalOutputHead | ~0.13M |
| **Total** | **~11.5M** |
