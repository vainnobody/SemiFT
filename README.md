# SemiFT

SemiFT 是一个基于 PyTorch 的半监督语义分割研究仓库，核心目标是在 **DINOv2 / DINOv3 + DPT** 框架上，以 **参数高效微调（PEFT）** 的方式完成半监督训练。

当前仓库的主方法是 **SemiFT**：
- 以 `peft/tuners/semift.py` 中的 `semift` 适配器为核心；
- 默认作用在 Transformer 的 `mlp` 模块；
- 支持 Mixture-of-Experts / prefix / conv expert 等配置；
- 已集成到 `unimatchv2_peft.py` 训练入口中。

除了 SemiFT，本仓库也保留了多种半监督分割方法与实验入口，例如 `fixmatch.py`、`unimatch.py`、`fixmatch_rgcr.py`、`unimatch_v2.py`、`rankmatch.py` 等，方便做对比实验。

---

## 1. 仓库结构

```text
SemiFT/
├── configs/                 # 数据集/训练配置
├── dataset/                 # 数据集与增强
├── model/
│   ├── backbone/            # DINOv2 / DINOv3
│   └── semseg/              # DPT / UperNet 等分割头
├── peft/                    # PEFT 与 SemiFT 适配器实现
├── splits/                  # 半监督划分文件
├── test/                    # 轻量测试
├── unimatchv2_peft.py       # UniMatchV2 + PEFT（推荐 SemiFT 入口）
├── fixmatch_rgcr.py         # FixMatch-RGCR 对比入口
└── README.md
```

---

## 2. SemiFT 方法概览

从代码实现看，仓库中的 SemiFT 训练流程由两部分组成：

1. **半监督训练框架**
   - `unimatchv2_peft.py`

2. **PEFT / SemiFT 适配器**
   - `peft/tuners/semift.py`

默认配置下：
- `method: semift`
- `target_modules: ["mlp"]`
- `freeze_backbone: True`
- `modules_to_save: ["head"]`

也就是说，主干 DINO 参数通常冻结，只训练：
- SemiFT 适配器参数
- 解码头参数
- 配置中指定的可保存模块

这也是当前仓库最推荐的使用方式。

---

## 3. 环境安装

推荐直接使用仓库内的一键 conda 安装脚本。默认会创建 `semift` 环境，并安装主线训练所需依赖（`unimatchv2_peft.py` + DPT + DINO + PEFT）。

### 3.1 一键创建 conda 环境（推荐）

```bash
bash scripts/setup_conda_env.sh
```

常用变体：

```bash
# 指定环境名
bash scripts/setup_conda_env.sh --env-name semift

# 若环境已存在，显式删除并重建
bash scripts/setup_conda_env.sh --env-name semift --recreate-existing

# 强制安装 CUDA 12.1 版 PyTorch
bash scripts/setup_conda_env.sh --env-name semift-gpu --cuda 12.1

# 额外安装 xformers（仅 Linux + CUDA）
bash scripts/setup_conda_env.sh --env-name semift-gpu --cuda 12.1 --with-xformers

# 额外安装 mmseg 生态（用于部分 UPerNet / mmseg 相关实验）
bash scripts/setup_conda_env.sh --env-name semift-mmseg --with-mmseg
```

脚本行为说明：
- 默认 `--cuda auto`：Linux 且检测到 `nvidia-smi` 时安装 CUDA 版 PyTorch，否则安装 CPU 版
- 默认 Python 版本为 `3.10`
- 会安装仓库常用依赖，并将 PyTorch 家族固定为兼容组合：`pytorch 2.4.1` + `torchvision 0.19.1`
- GPU 环境使用 `--strict-channel-priority` + `pytorch`/`nvidia` channel 安装，避免 `torch` / `torchvision` 混装
- 若环境已存在，默认直接报错；只有显式传入 `--recreate-existing` 才会删除并重建
- 安装完成后会自动执行一次 import smoke test，并验证 `dataset.transform` 可正常导入

### 3.2 手动安装（备用）

如果你不想使用脚本，也可以手动安装：

```bash
conda create -n semift python=3.10 -y
conda activate semift
conda install -y --strict-channel-priority -c pytorch pytorch=2.4.1 torchvision=0.19.1 cpuonly
pip install -r requirements.txt
```

如果使用 GPU，请改为：

```bash
conda create -n semift python=3.10 -y
conda activate semift
conda install -y --strict-channel-priority -c pytorch -c nvidia pytorch=2.4.1 torchvision=0.19.1 pytorch-cuda=12.1
pip install -r requirements.txt
```

> 注意：不要在 conda 安装完 `pytorch` / `torchvision` 后，再用 `pip install torch torchvision` 覆盖它们。`requirements.txt` 已刻意移除了这两个包，避免破坏 conda 环境中的官方匹配组合。

如果要使用分布式训练，确保本机已经正确安装：
- PyTorch
- NCCL / CUDA
- TensorBoard（训练日志会写入 `SummaryWriter`）

可选依赖说明：
- `xformers`：DINOv2 中会自动探测；未安装时会回退到普通 attention
- `mmseg` / `mmengine` / `mmcv-lite`：仅部分 mmseg / UPerNet 相关模块需要，不是主线训练必需依赖

---

## 4. 数据与预训练权重准备

### 4.1 数据集路径
在 `configs/*.yaml` 中设置：

```yaml
data_root: Your/Dataset/Path
```

### 4.2 半监督划分文件
训练命令通常需要：
- `--labeled-id-path`
- `--unlabeled-id-path`

例如：
- `splits/pascal/1_16/labeled.txt`
- `splits/pascal/1_16/unlabeled.txt`

请根据你实际的划分文件名替换。

### 4.3 Backbone 预训练权重
代码默认从如下位置读取主干权重：

```text
./pretrained/<backbone>.pth
```

例如：
- `./pretrained/dinov2_small.pth`
- `./pretrained/dinov2_base.pth`
- `./pretrained/dinov3_small.pth`

请先手动准备好对应权重文件。

---

## 5. 推荐训练入口

## 5.1 UniMatchV2 + SemiFT（推荐）

这是当前最标准的 **SemiFT 主方法入口**。

### 单机多卡训练
```bash
torchrun --nproc_per_node=4 unimatchv2_peft.py \
  --config configs/pascal.yaml \
  --labeled-id-path splits/pascal/1_16/labeled.txt \
  --unlabeled-id-path splits/pascal/1_16/unlabeled.txt \
  --save-path experiments/semift_pascal \
  --port 29500
```

### 配置文件示例
`configs/pascal.yaml` 中默认已经给出 SemiFT 配置：

```yaml
peft:
  method: semift
  target_modules: ["mlp"]
  freeze_backbone: True
  modules_to_save: ["head"]
  bias: lora_only
  moe_num_experts: 4
  moe_topk: 2
  moe_router_balance_mode: deepseek_v3
  moe_router_bias_update_speed: 0.001
  moe_router_bias_clip: 0.05
```

### 命令行覆盖示例
```bash
torchrun --nproc_per_node=4 unimatchv2_peft.py \
  --config configs/pascal.yaml \
  --labeled-id-path splits/pascal/1_16/labeled.txt \
  --unlabeled-id-path splits/pascal/1_16/unlabeled.txt \
  --save-path experiments/semift_pascal_mlp \
  --peft-method semift \
  --peft-target-modules mlp \
  --freeze-backbone \
  --port 29500
```

---

## 6. PEFT 方法支持

`peft/tuners/semift.py` 当前支持以下方法：

- `semift`
- `semift_samoe`
- `samoev4`
- `samoev5`
- `samoev6`
- `samoev7`
- `samoev8`
- `semift_scalegate`
- `lora`
- `ssf`
- `bitfit`
- `adaptformer`
- `fact_tt`
- `fact_tk`
- `conv_lora`
- `hydralora`

默认目标模块：

| method | default target_modules |
|---|---|
| semift | `["mlp"]` |
| semift_samoe | `["mlp"]` |
| samoev4 | `["mlp"]` |
| samoev5 | `["mlp"]` |
| samoev6 | `["mlp"]` |
| samoev7 | `["mlp"]` |
| samoev8 | `["mlp"]` |
| semift_scalegate | `["mlp"]` |
| lora | `["qkv", "proj", "fc1", "fc2"]` |
| ssf | `["patch_embed", "norm1", "norm2", "qkv", "proj", "fc1", "fc2"]` |
| bitfit | `["qkv", "proj", "fc1", "fc2", "norm1", "norm2", "head"]` |
| adaptformer | `["mlp"]` |
| fact_tt | `["mlp"]` |
| fact_tk | `["mlp"]` |
| conv_lora | `["qkv", "proj", "fc1", "fc2"]` |
| hydralora | `["qkv", "proj", "fc1", "fc2"]` |

如果你只想使用主方法，请优先采用：

```yaml
peft:
  method: semift
  target_modules: ["mlp"]
  freeze_backbone: True
```

---

## 7. 其他对比方法入口

仓库中保留了多个对比方法脚本，便于做消融和横向比较：

- `supervised.py`
- `fixmatch.py`
- `fixmatch_rgcr.py`
- `unimatch_v2.py`
- `rankmatch.py`
- `wscl.py`
- `dwl.py`

例如，`fixmatch_rgcr.py` 是一个较完整的半监督训练入口，包含：
- 配置读取
- DDP 初始化
- DPT 模型构建
- teacher / student 训练逻辑
- 验证与 checkpoint 保存

本仓库新的 PEFT 入口基本延续了这种训练脚本组织方式，但把重点放在 **SemiFT 适配器 + 半监督分割** 上。

---

## 8. 常用配置说明

以 `configs/pascal.yaml` 为例：

```yaml
dataset: pascal
nclass: 21
crop_size: 518

epochs: 60
batch_size: 4
lr: 0.000005
lr_multi: 40.0
conf_thresh: 0.95

model: dpt
backbone: dinov2_small
```

说明：
- `batch_size` 为 **每张卡** 的 batch size
- `lr_multi` 通常用于 decoder / 非 backbone 参数组
- `conf_thresh` 是伪标签置信度阈值
- `backbone` 目前主要围绕 `dinov2_*` / `dinov3_*`

---

## 9. 训练输出

训练过程中通常会在 `--save-path` 下保存：

- `latest.pth`
- `best.pth`
- TensorBoard 日志

推荐用 TensorBoard 查看：

```bash
tensorboard --logdir experiments
```

---

## 10. 无人值守批处理

如果你希望在**无人监管**的情况下顺序跑完一批实验，可以使用新的 YAML 清单式批处理入口：

### 10.1 先做 dry-run 校验

```bash
python scripts/batch_train.py \
  --manifest manifests/batch_jobs.example.yaml \
  --dry-run
```

它会校验以下内容：
- 训练脚本是否存在且属于仓库支持的入口
- `config` / `labeled_id_path` / `unlabeled_id_path` 是否存在
- `name` 与 `save_subdir` 是否唯一
- 每个任务最终分配到的 `save_path` 与 `port`

### 10.2 正式执行

```bash
python scripts/batch_train.py \
  --manifest manifests/batch_jobs.example.yaml
```

manifest 采用 `global + jobs` 结构，例如：

```yaml
global:
  nproc_per_node: 4
  save_root: experiments/batch_runs
  port_base: 29500
  continue_on_error: true
  max_retries: 1

jobs:
  - name: semift_pascal_1_16
    script: unimatchv2_peft.py
    config: configs/pascal.yaml
    config_overrides:
      backbone: dinov2_small
      conf_thresh: 0.9
    labeled_id_path: splits/pascal/92/labeled.txt
    unlabeled_id_path: splits/pascal/92/unlabeled.txt
    save_subdir: pascal/unimatchv2_peft/semift_92
    extra_args: ["--peft-method", "semift", "--peft-target-modules", "mlp", "--freeze-backbone"]
```

完整示例见：`manifests/batch_jobs.example.yaml`。

GPU 分配支持两种模式：

- **固定 GPU 模式**：在 `global.env` 或 `job.env` 中显式设置 `CUDA_VISIBLE_DEVICES`
- **动态 GPU 模式**：不设置 `CUDA_VISIBLE_DEVICES`，并在启动时加 `--wait-for-gpu`
  - 此时 `nproc_per_node` 会被视为该任务需要的 GPU 数量
  - runner 会持续轮询 `nvidia-smi`
  - 一旦检测到至少 `nproc_per_node` 张空闲卡，就自动挑选这些空闲卡启动训练

例如，如果你想“谁空闲就用谁”，并且一次需要 6 张卡，可以这样写：

```yaml
global:
  nproc_per_node: 6
  save_root: experiments/batch_runs
  port_base: 29500
  continue_on_error: true
  max_retries: 1
```

然后运行：

```bash
python scripts/batch_train.py \
  --manifest manifests/batch_jobs.example.yaml \
  --wait-for-gpu
```

这时 runner 会等到任意 6 张 GPU 同时空闲，再把它们写入该 job 的 `CUDA_VISIBLE_DEVICES` 后启动。

如果你不想为每个实验复制一份 YAML，可以直接在 job 里写 `config_overrides`。runner 会在运行前基于原始 `config` 递归合并这些字段，并自动生成一个临时 config，例如：

```yaml
  - name: fixmatch_rgcrv6_vaihingen_split4_dinov3_small
    script: fixmatch_rgcrv6.py
    config: configs/vaihingen.yaml
    config_overrides:
      backbone: dinov3_small
    labeled_id_path: splits/vaihingen/4/labeled.txt
    unlabeled_id_path: splits/vaihingen/4/unlabeled.txt
    save_subdir: vaihingen/fixmatch_rgcrv6/dinov3_small/split4
```

生成后的 config 会写到 `<save_root>/_batch/generated_configs/<manifest_name>/<job_name>.yaml`，命令实际传入的是这个生成后的文件。

### 10.3 批处理行为

- 默认使用当前 `python` 解释器调用 `torch.distributed.run` 启动每个任务
- 按 `port_base + job_index` 自动分配端口
- 每个任务日志写入 `<save_path>/out.log`
- 运行时终端会输出批处理进度，例如当前是第几个任务、任务开始/结束状态
- 成功任务会写入 `<save_path>/.batch_done.json`，下次批处理会自动跳过
- 若 `<save_path>/latest.pth` 已存在且没有完成标记，runner 会按同一路径重新启动，让训练脚本自动 resume
- 单个任务失败后默认**继续后续任务**，并按 `max_retries` 自动重试
- 批次级汇总会写到 `<save_root>/_batch/`

这样可以确保 `python scripts/batch_train.py ...` 启动出的训练进程与当前激活的 Conda / venv 环境保持一致，避免 `torchrun` 从其他环境的 `PATH` 中被解析出来。

### 10.4 常用筛选与覆盖

只执行部分任务：

```bash
python scripts/batch_train.py \
  --manifest manifests/batch_jobs.example.yaml \
  --only semift_pascal_1_16
```

临时覆盖输出根目录：

```bash
python scripts/batch_train.py \
  --manifest manifests/batch_jobs.example.yaml \
  --save-root /tmp/semift_batch_runs
```

等待指定 GPU 或自动选择空闲 GPU：

```bash
python scripts/batch_train.py \
  --manifest manifests/batch_jobs.example.yaml \
  --wait-for-gpu
```

- 如果 manifest 中设置了 `CUDA_VISIBLE_DEVICES`，则等待这些指定 GPU 全部空闲
- 如果没有设置 `CUDA_VISIBLE_DEVICES`，则动态挑选 `nproc_per_node` 张空闲 GPU

---

## 11. 测试与调试

仓库中有一部分轻量测试，建议在修改后至少做语法或定向测试。

示例：

```bash
python -m py_compile unimatchv2_peft.py peft/tuners/semift.py
```

如果本地安装了 `pytest`：

```bash
python -m pytest test/test_unimatchv2_peft_config.py
python -m pytest test/test_batch_train.py
```

如果当前环境没有安装 `pytest`，也可以直接用标准库执行批处理测试：

```bash
python -m unittest discover -s test -p 'test_batch_train.py' -v
```

---

## 12. 实用建议

1. **优先从 `unimatchv2_peft.py` 跑通 SemiFT。**  
2. 如果显存紧张：
   - 降低 `batch_size`
   - 使用 `dinov2_small`
   - 减少 `crop_size`
   - 保持 `freeze_backbone: True`
3. 如果训练很慢：
   - 避免过大的多尺度范围
   - 检查 `img_scales`
   - 优先使用默认 SemiFT 配置
4. 若更换 PEFT 方法，建议同步检查：
   - `target_modules`
   - `freeze_backbone`
   - `modules_to_save`

---

## 13. 总结

如果你只关心本仓库的主方法，可以直接记住下面这条：

> **SemiFT = `unimatchv2_peft.py` + `peft/tuners/semift.py` + `configs/*.yaml` 中的 `peft.method: semift` 配置。**

推荐起步命令：

```bash
torchrun --nproc_per_node=4 unimatchv2_peft.py \
  --config configs/pascal.yaml \
  --labeled-id-path splits/pascal/1_16/labeled.txt \
  --unlabeled-id-path splits/pascal/1_16/unlabeled.txt \
  --save-path experiments/semift_pascal \
  --port 29500
```
