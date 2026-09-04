# tsproj_stf

面向能源场景的可复现时空预测（Spatio-Temporal Forecasting）实验工程。

项目采用“自有数据与评估契约 + BasicTS 训练后端”：由本项目负责时间切分、滑窗、缺失掩码、归一化、指标和实验产物，BasicTS 1.1.0 提供 STID 训练能力；Graph WaveNet 由项目以纯 PyTorch 实现。

## 当前路线

1. **M0**：工程规范与依赖兼容性；
2. **M1**：数据、评估、Persistence 与 artifact 契约；
3. **M2**：METR-LA + STID；
4. **M3**：复现加固与 BasicTS parity；
5. **M4**：Graph WaveNet fixed/adaptive/hybrid 图消融；
6. **M5**：电力数据 adapter、多图和分位数接口。

设计与实施依据：

- `docs/design/2026-09-04-时空预测基线工程设计.md`
- `docs/plans/2026-09-04-时空预测基线工程实施计划.md`

## 开发环境

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

所有 Python 命令必须通过项目根目录 `.venv` 的 `uv` 环境运行。正式数据、结果和 checkpoint 不进入 Git。

## 目录

```text
configs/          数据集与实验配置
docs/             调研、设计与实施文档
scripts/          数据准备、实验运行与结果汇总入口
src/tsproj_stf/   项目源码
tests/            单元与集成测试
```

## 首个公开基准

- 数据：METR-LA
- 输入/输出长度：12/12
- 时间切分：7:1:2
- 基线：Persistence、STID、Graph WaveNet fixed/adaptive/hybrid
- 指标：反归一化后的 MAE、RMSE、MAPE、WAPE，并报告 overall、h3、h6、h12

### 准备 METR-LA

原始 CSV 和邻接矩阵路径由 `configs/datasets/metr_la.yaml` 显式指定。准备器会校验
5 分钟等间隔时间戳、节点顺序和图 shape；下载或输入校验失败时不会生成占位数据。

```bash
uv run python scripts/prepare_data.py --config configs/datasets/metr_la.yaml
```

### 运行与汇总基线

```bash
uv run python scripts/run_experiment.py \
  --config configs/experiments/metr_la_persistence.yaml
uv run python scripts/run_experiment.py \
  --config configs/experiments/metr_la_stid.yaml --seed 42
uv run python scripts/run_experiment.py \
  --config configs/experiments/metr_la_stid.yaml --seed 43
uv run python scripts/run_experiment.py \
  --config configs/experiments/metr_la_stid.yaml --seed 44
uv run python scripts/summarize_runs.py \
  --root results/METR-LA --prefix metr_la_stid --seeds 42 43 44
```

Graph WaveNet 三种图模式使用相同模型容量和训练协议，只改变 support 来源：

```bash
uv run python scripts/run_experiment.py \
  --config configs/experiments/metr_la_gwn_fixed.yaml
uv run python scripts/run_experiment.py \
  --config configs/experiments/metr_la_gwn_adaptive.yaml
uv run python scripts/run_experiment.py \
  --config configs/experiments/metr_la_gwn_hybrid.yaml
```

当前 Graph WaveNet 配置是面向 macOS CPU 可执行性的参考消融配置，不等同于论文容量或
CUDA 调优配置。fixed 使用 `physical` 图的正向/反向扩散，adaptive 只使用可学习非对称图，
hybrid 同时使用两类 support。STID 仍是节点身份基线，不读取邻接矩阵。

Persistence 对有历史节点始终使用窗口内最后有效值；若完整输入窗没有任何有效历史，正式
METR-LA 配置使用该节点的 train-only 有效均值作为 cold-start，不把物理零伪装成历史值。

run ID 为 `<name>-seed<seed>`。已有同 ID、同配置的 run 默认拒绝覆盖，并提示显式选择：

- `--resume`：仅恢复同一 resolved config 的未完成 run；completed run 不可变。
- `--force-new-run`：保留原目录，分配 `-run2`、`-run3` 等新目录。

同 run ID 对应不同 resolved config 时始终失败。两种选项互斥。

runner 在创建 run 目录前重算 manifest 中全部标准 processed 文件的 SHA-256；run 内
`data_manifest.json` 额外记录源 manifest fingerprint 和精确 train/validation/test 边界，
`environment.json` 记录 Git 状态及 `uv.lock` fingerprint。多 seed 汇总要求除 seed 外的
resolved config 与完整 data manifest 一致；训练模型缺 checkpoint 或指标非有限时拒绝汇总。

### 指标语义

项目指标以独立 `observed` mask 为事实来源，并在反归一化后计算。固定数组协议对照确认，
项目 MAE、RMSE、MAPE 与 BasicTS 1.1.0 在使用同一 prediction、target 和 mask 时一致；
MAPE 均返回 0～1 比例并排除绝对值不超过 `5e-5` 的 target。

WAPE 不要求与 BasicTS 相等：项目定义为全部有效目标的全局
`sum(abs(error)) / sum(abs(target))`；BasicTS 1.1.0 先沿张量第 1 维计算比率再取平均。
正式对比统一使用项目定义，避免因 batch/horizon 聚合方式改变结论。

## 电力数据迁移契约

`configs/datasets/power_grid.example.yaml` 展示通用电力宽表配置。准备命令为：

```bash
uv run python scripts/prepare_data.py \
  --config configs/datasets/power_grid.example.yaml
```

真实配置必须提供：

- `csv_path`：本地 CSV；不得用下载失败后的占位文件。
- `timestamp_column`：时间列名；加载后排序，但重复或非等间隔时间戳直接失败。
- `node_ids`：唯一且有序的 canonical 节点列表。
- `feature_columns[feature][node_id]`：每个 feature、每个节点对应的 CSV 列名。
- `target_features`：显式目标 feature；不根据列名前后缀猜测。
- `fill_value`：仅用于存储非有限缺失值；真实缺失事实始终由 `observed` mask 表示，零值不会自动视为缺失。

例如两个节点、负荷与电价两个 feature 必须明确映射为：

```yaml
node_ids: [substation_a, substation_b]
feature_columns:
  load:
    substation_a: load_a
    substation_b: load_b
  price:
    substation_a: price_a
    substation_b: price_b
target_features: [load]
```

### 电力多图输入

图构建接口位于 `tsproj_stf.data.graphs`，所有输出都严格遵循 `node_ids` 顺序：

- 物理图：edge list 每行包含 `source`、`target`、非负 `weight`；是否有向必须显式指定。
- 电气距离图：距离矩阵必须同时提供自身 `distance_node_ids`；接口先重排，再应用阈值高斯核。
- 统计相关图：必须显式指定 target feature 和 `train_slice`，按 pairwise observed mask 计算；
  `threshold` 与逐行 `top_k` 稀疏化采用稳定节点索引打破 ties。

不同关系以 `graphs.npz` 中不同 key 保存，不在预处理阶段混成单图。fixed Graph WaveNet
一次显式选择一个 `graph_name`；hybrid 在所选固定图之外增加 adaptive support。

### 分位数接口

`QuantileHead` 接收 `[B,H,N,F]` 并输出 `[B,H,N,Q]`，通过累计非负增量保证分位点不交叉；
`pinball_loss` 使用与确定性指标相同的 target mask。当前阶段已完成合成电力链路 smoke，
但未接入或运行任何私有电力数据，因此不能把本阶段称为真实电力实验结果。