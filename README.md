# AF_DRL

本仓库现在按论文 **Stabilizing and Accelerating Autofocus with Expert Trajectory Regularized Deep Reinforcement Learning (CVPR 2025)** 的方法重构了训练、专家轨迹生成、第二阶段强化学习训练与测试流程；唯一保留为工程自定义部分的是 **数据读取格式**，继续使用当前仓库的 TXT + RAW/NPY 数据组织方式。

## 1. 当前工程覆盖的论文方法

### Phase 1：Actor 预训练
- 使用 **MobileNetV2 backbone + RoI PE + Lens PE + temperature**。
- 输出为 **相对镜头移动分布**，不是绝对焦点位置分布。
- 使用论文补充材料 Sec.7 的 **ordinal regression loss**。
- 默认训练超参数与论文对齐：
  - `lr=1e-3`
  - `Adam(beta1=0.5, beta2=0.999)`
  - `batch_size=128`
  - `temperature=2.0`
  - `iterations=10000`

### 专家轨迹生成
- 先用 **Phase 1 预训练 actor** 对训练集每个起始焦点位置 rollout 原始轨迹。
- 然后严格按论文生成专家轨迹：
  - **Algorithm 1**：对原始轨迹做镜像、clip、重新排序。
  - **Algorithm 2**：保留初始位置，之后直接到 GT，后续动作为 0。
  - **Algorithm 3**：按 `d <- int(d / m)` 逐步衰减地靠近 GT。
- 默认 `max_steps=4`、`m=5`，与论文实现细节一致。

### Phase 2：PPO-CLIP + Expert Trajectory Regularization
- Reward 采用论文公式：
  - `R(st, at) = -|k_{t+1} - kgt| + Rfh`
- Focus hunting penalty 默认 `-1.5`。
- PPO 默认超参数：
  - `lr=1e-5`
  - `clip_eps=0.2`
  - `mini_batch_size=32`
  - `expert_lambda=1e-3`
  - `max_env_steps=4`
- 专家正则化在 PPO 更新时从离线专家轨迹集中 **按 trajectory batch 采样**。

### 测试 / 评估
- `evaluate_phase1.py`：支持 Phase 1 单步指标 + 多步 rollout 指标。
- `evaluate_phase2.py`：支持 Phase 2 多步 rollout 指标。
- 评估指标包括：
  - `<=0`, `<=1`, `<=2`, `<=4`
  - `MAE`, `RMSE`
  - `FH`（focus hunting 率）
  - `avg_steps_to_gt`

---

## 2. 数据格式

继续沿用当前工程数据格式：

```text
SceneName LeftRawPrefix RightRawPrefix FocusIndex GTFocusIndex PatchX PatchY Temperature
```

示例：

```text
scene_001 left/0001 right/0001 12 18 3 7 6500
```

说明：
- `LeftRawPrefix` / `RightRawPrefix` 不包含扩展名。
- 训练脚本默认会拼接 `raw_suffix`，默认是 `.npy`。
- 如果你的数据是 `.raw` / `.png` / `.tiff`，可以通过 `--raw_suffix` 改。

---

## 3. 依赖

建议 Python 3.10+，并安装：

- `torch`
- `torchvision`
- `gymnasium`
- `numpy`

如果使用 GPU，请安装对应 CUDA 版本的 PyTorch。

---

## 4. 推荐目录结构

```text
AF_DRL/
├── train_phase1.py
├── trajectory_builder.py
├── train_phase2.py
├── evaluate_phase1.py
├── evaluate_phase2.py
├── scripts/
│   ├── train_phase1.sh
│   ├── build_expert_trajectories.sh
│   └── train_phase2.sh
├── checkpoints/
└── artifacts/
```

---

## 5. 完整流程

### Step 1：Phase 1 预训练

可以直接改 bash 脚本顶部变量：

```bash
bash scripts/train_phase1.sh
```

也可以直接运行 Python：

```bash
python train_phase1.py \
  --txt_path /path/to/train.txt \
  --val_txt_path /path/to/val.txt \
  --data_root /path/to/raw \
  --output_dir /path/to/checkpoints/phase1 \
  --device cuda
```

### Step 2：生成专家轨迹（Phase 2 前）

```bash
bash scripts/build_expert_trajectories.sh
```

等价 Python 命令：

```bash
python trajectory_builder.py \
  --txt_path /path/to/train.txt \
  --data_root /path/to/raw \
  --policy_ckpt /path/to/checkpoints/phase1/best_model.pth \
  --output /path/to/artifacts/expert_trajectories.json \
  --save_original_json /path/to/artifacts/original_trajectories.json \
  --max_steps 4 \
  --m 5 \
  --algos 1,2,3 \
  --device cuda
```

### Step 3：Phase 2 PPO 训练

```bash
bash scripts/train_phase2.sh
```

等价 Python 命令：

```bash
python train_phase2.py \
  --txt_path /path/to/train.txt \
  --val_txt_path /path/to/val.txt \
  --data_root /path/to/raw \
  --pretrained /path/to/checkpoints/phase1/best_model.pth \
  --expert_json /path/to/artifacts/expert_trajectories.json \
  --output_dir /path/to/checkpoints/phase2 \
  --device cuda
```

---

## 6. 测试 / 评估

### 6.1 评估 Phase 1

```bash
python evaluate_phase1.py \
  --txt_path /path/to/test.txt \
  --data_root /path/to/raw \
  --checkpoint /path/to/checkpoints/phase1/best_model.pth \
  --device cuda
```

可选：
- `--rollout_json /path/to/phase1_rollout.json`

### 6.2 评估 Phase 2

```bash
python evaluate_phase2.py \
  --txt_path /path/to/test.txt \
  --data_root /path/to/raw \
  --checkpoint /path/to/checkpoints/phase2/best_model.pth \
  --max_steps 4 \
  --device cuda
```

可选：
- `--rollout_json /path/to/phase2_rollout.json`

---

## 7. 关键脚本说明

### `train_phase1.py`
主要参数：
- `--txt_path`：训练集 TXT
- `--val_txt_path`：验证集 TXT
- `--data_root`：原始数据目录
- `--iterations`：训练 iteration 数，默认 10000
- `--batch_size`：默认 128
- `--lr`：默认 1e-3
- `--temperature`：ordinal regression 温度，默认 2.0
- `--resume`：从中断 checkpoint 恢复训练
- `--imagenet_pretrained / --no-imagenet_pretrained`

输出：
- `best_model.pth`
- `last_model.pth`
- `checkpoint_iter*.pth`

### `trajectory_builder.py`
主要参数：
- `--policy_ckpt`：Phase 1 checkpoint
- `--max_steps`：原始轨迹 rollout 长度，默认 4
- `--algos`：默认 `1,2,3`
- `--m`：Algorithm 3 的除数，默认 5
- `--save_original_json`：保存原始轨迹，便于检查

输出：
- 专家轨迹 JSON
- 可选原始轨迹 JSON

### `train_phase2.py`
主要参数：
- `--pretrained`：加载 Phase 1 actor 权重
- `--resume`：恢复 Phase 2 训练
- `--expert_json`：专家轨迹 JSON
- `--updates`：PPO 更新轮数
- `--rollout_steps`：每次更新前采样环境步数
- `--mini_batch_size`：默认 32
- `--expert_batch_trajectories`：每次专家损失采样的 trajectory 数量
- `--expert_lambda`：默认 1e-3
- `--fh_penalty`：默认 -1.5
- `--freeze_backbone`：Phase 2 时冻结 backbone

输出：
- `best_model.pth`
- `last_model.pth`
- `checkpoint_update*.pth`

---

## 8. 关于“严格按论文复现”的范围说明

本仓库当前已经在以下核心点上改成论文一致：

1. Phase 1 输出改为 **相对 movement distribution**。
2. Phase 1 checkpoint 可直接初始化 Phase 2 actor head。
3. Action range 改为论文中的 **[-kmax, kmax]** 离散动作空间。
4. Reward 改为论文的 **negative distance + focus hunting penalty**。
5. Expert regularization 改为 **离线专家轨迹采样**。
6. Expert trajectory 生成流程改为：
   - 先 rollout 原始轨迹
   - 再按 Algorithm 1 / 2 / 3 生成专家轨迹

保留不变的只有数据读取接口，以适配你自己的数据格式与路径组织。

---

## 9. 建议使用顺序

1. 先跑 `train_phase1.py`
2. 再跑 `trajectory_builder.py`
3. 再跑 `train_phase2.py`
4. 最后用 `evaluate_phase1.py` / `evaluate_phase2.py` 测试

如果你希望进一步贴近论文实验，还建议：
- 保持 `max_steps=4`
- 保持 `expert_lambda=1e-3`
- 保持 `fh_penalty=-1.5`
- 保持 `m=5`
- 评估时使用 deterministic rollout（当前默认就是 argmax）
