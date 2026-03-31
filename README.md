
# resGPT: Attention Residuals on nanoGPT

本项目基于 nanoGPT 改造，核心目标是验证和分析 **Attention Residuals (AttnRes)** 在 TinyStories 上的训练表现。

与原版相比，这个仓库重点增加了：

- 可切换的 AttnRes 残差路径（`use_attn_res`）
- 深度聚合相关参数（`attn_res_rms_eps`）
- 训练诊断日志（RMS、梯度范数、AttnRes query 熵/权重）
- 一组可直接对比的 TinyStories 配置（12L / 16L / 20L，baseline vs attnres）

## 1. 项目结构

```text
.
|- model.py                 # GPT + AttnRes 实现
|- train.py                 # 训练主脚本（单卡/DDP）
|- sample.py                # 采样脚本
|- bench.py                 # 简化 benchmark
|- configurator.py          # 配置覆盖器（配置文件 + --key=value）
|- config/
|  |- train_tinystories.py
|  |- train_tinystories_attnres.py
|  |- train_tinystories_16l.py
|  |- train_tinystories_16l_attnres.py
|  |- train_tinystories_20l.py
|  |- train_tinystories_20l_attnres.py
|  `- ... 其他 nanoGPT 兼容配置
|- data/
|  |- tinystories/prepare.py
|  `- ... 其他数据集准备脚本
`- out-tinystories-*/       # 已训练输出目录（若存在）
```

## 2. 环境安装

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

可选：如需记录实验日志，先登录 W&B。

```bash
wandb login
```

## 3. 数据准备（TinyStories）

首次训练前执行：

```bash
python data/tinystories/prepare.py
```

默认会下载 `roneneldan/TinyStories`，并在 `data/tinystories/` 生成：

- `train.bin`
- `val.bin`
- `meta.pkl`

可选子集参数（快速 smoke test）：

```bash
python data/tinystories/prepare.py --max_train_examples=200000 --max_val_examples=2000
```

## 4. 快速开始：baseline vs AttnRes

### 4.1 12 层对照实验

Baseline：

```bash
python train.py config/train_tinystories.py
```

AttnRes：

```bash
python train.py config/train_tinystories_attnres.py
```

### 4.2 16 层 / 20 层对照实验

```bash
python train.py config/train_tinystories_16l.py
python train.py config/train_tinystories_16l_attnres.py

python train.py config/train_tinystories_20l.py
python train.py config/train_tinystories_20l_attnres.py
```

### 4.3 采样

从训练输出目录采样：

```bash
python sample.py --out_dir=out-tinystories-12l --start="Once upon a time"
python sample.py --out_dir=out-tinystories-12l-attnres --start="Once upon a time"
```


## 5. 关键参数说明

训练脚本和配置文件支持同名参数覆盖。典型用法：

```bash
python train.py config/train_tinystories_attnres.py --batch_size=32 --compile=False
```

常用参数：

- `use_attn_res`: 是否启用 Attention Residuals
- `attn_res_rms_eps`: AttnRes 中 RMSNorm 的 eps
- `track_diagnostics`: 是否启用诊断指标收集
- `diagnostics_interval`: 诊断日志间隔
- `compile`: 是否使用 `torch.compile`
- `init_from`: `scratch` / `resume` / `gpt2*`

## 6. 诊断与日志

当 `wandb_log=True` 且 `track_diagnostics=True` 时，训练会记录：

- 各层 attention/MLP 输出 RMS
- 各层 block 输出 RMS
- AttnRes query 熵、最大权重、最近分量权重
- 各层梯度范数，以及 AttnRes 模块梯度范数

这套指标用于分析 AttnRes 是否改善深层信息聚合和梯度分布。

## 7. 多卡训练（DDP）

单机多卡：

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_tinystories_attnres.py
```

多机场景可按 `train.py` 顶部示例传入 `--nnodes`、`--node_rank`、`--master_addr`、`--master_port`。

## 8. Windows 注意事项

- 若 `torch.compile` 不稳定，可加 `--compile=False`
- 默认 `backend='nccl'` 主要用于 CUDA 训练；CPU/非 CUDA 环境请按需改为 `gloo`
- `auto_shutdown=True` 在 Windows 下会调用系统关机命令，使用前请确认

## 9. 预置输出目录

仓库中如果已有如下目录，通常表示曾跑过对应实验：

- `out-tinystories-12l`
- `out-tinystories-12l-attnres`
- `out-tinystories-16l`
- `out-tinystories-16l-attnres`
- `out-tinystories-20l`
- `out-tinystories-20l-attnres`

每个目录通常包含 `ckpt.pt`，可直接用于 `sample.py` 推理。

## 10. 致谢

本项目继承了 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的整体训练框架，并在此基础上扩展 AttnRes 相关实验能力。
