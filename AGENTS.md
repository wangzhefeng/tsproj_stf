# AGENTS.md

本文件是 `tsproj_stf` 的权威工程规范。项目目标是建立可复现的时空预测实验链路，优先保证数据协议、评估协议和实验产物可信，再比较模型。

## 开发环境

- Python 固定使用项目根目录 `.venv`，环境和命令统一通过 `uv` 管理。
- 新增、删除、更新依赖使用 `uv add`、`uv remove`，不直接维护 pip 环境。
- 源码采用 `src/tsproj_stf/` layout，测试位于 `tests/`。
- 默认中文注释和文档；标识符、配置键和命令使用英文。

## 数据契约

- 项目内部时空数据统一为 `values[time, node, feature]`。
- `observed` 与 `values` shape 完全相同，是真实缺失状态的唯一来源；不得把零值默认视为缺失。
- `timestamps` 必须严格递增且等间隔。
- `node_ids` 唯一；所有邻接矩阵必须与其顺序一致。
- 图以 `graphs[name][node, node]` 保存，允许有向、非对称和多图。
- 任何模型适配都必须显式选择 target feature；不得静默 reshape、flatten 或丢弃 feature。

## 训练与评估

- train/validation/test 只按时间连续切分，严禁随机切分。
- 各 split 内独立生成滑窗，窗口不得跨 split 边界。
- train DataLoader 可在窗口生成后 shuffle；validation/test 不 shuffle。
- scaler 只用 train split 的有效观测拟合。
- 默认 one-shot multi-horizon，不递归滚动生成训练标签。
- 正式指标在反归一化后计算，统一使用同一目标 mask。
- 至少报告 overall、h3、h6、h12 的 MAE、RMSE、MAPE，并保留 WAPE。
- 多 seed 实验必须保留每个 run，汇总均值与样本标准差。

## 模型边界

- BasicTS 1.1.0 是训练后端，不是数据和指标事实来源。
- STID 是节点身份基线，不读取邻接矩阵，不得表述为图模型。
- Graph WaveNet 由本项目实现，支持 fixed、adaptive、hybrid 三种图模式。
- 电力图的统计相关部分只能使用 train split 构建。

## 实验产物

正式 run 至少保存：

- `resolved_config.yaml`
- `environment.json`
- `data_manifest.json`
- `metrics.json`
- `predictions.npz`
- 训练模型的 checkpoint
- `run.log`

产物必须记录 Git 状态、依赖版本、数据 fingerprint、节点顺序 fingerprint、split 边界、seed 和模型配置。相同 run ID 对应不同配置时必须失败，不得静默覆盖。

## 开发纪律

- 功能代码按 RED→GREEN→REFACTOR 开发；先看到对应测试因缺失行为而失败，再写最小实现。
- 只修改当前阶段需要的文件，不做无关重构。
- 不用跳过、注释或弱化测试来掩盖错误。
- 数据下载失败时停止并报告，不生成伪造或占位数据。
- 不提交、不 push，除非 wangzf 明确授权。

## 验证命令

常规修改至少运行相关定向测试。阶段收口运行：

```bash
uv sync
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src scripts
```

设计依据见 `docs/design/2026-09-04-时空预测基线工程设计.md`，实施顺序见 `docs/plans/2026-09-04-时空预测基线工程实施计划.md`。
