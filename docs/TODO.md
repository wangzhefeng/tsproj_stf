# TODO

本文件跟踪 M0～M5 收口后遗留的待办事项。代码与文档层面的基线工程已全部完成并合并到
`main`（`45a475b`，123 tests 通过）；以下三项均不阻塞当前 main，等待外部条件或授权。

## 1. 补齐多 seed 实验并纳入新 artifact 契约

**背景**：STID 正式 run（seed 42/43/44）与 Graph WaveNet 三模式 run（各 seed 42）生成于
artifact 加固之前，`data_manifest.json` 缺少 `source_manifest_sha256` 与精确 split bounds；
Graph WaveNet 缺 seed 43/44，fixed/adaptive/hybrid 的数值差异不能下稳定性结论。

**内容**：

- 以新契约重训 STID seed 42/43/44，替换历史 `_formal` run（完成后删除旧目录）。
- 补训 Graph WaveNet fixed/adaptive/hybrid 的 seed 43/44。
- 用 `scripts/summarize_runs.py` 产出两组多 seed 汇总，比较图模式差异。

**验收**：所有 run 通过加固后的 summary 校验（config 一致、manifest 一致、checkpoint 存在、
指标有限）；汇总含 mean ± std。

**成本约束**：CPU 单 epoch 成本高，需安排在空闲时段批量跑。

## 2. 真实电力数据端到端实验

**背景**：M5 只用非敏感合成 fixture 验证了迁移契约（宽表显式映射、多图 provenance、
分位数接口）；私有电力宽表数据未提供。

**前置条件**：wangzf 提供真实电力宽表数据与节点拓扑/电气距离信息。

**内容**：

- 按 `configs/datasets/power_grid.example.yaml` 编写真实数据配置，运行 prepare。
- 构建 physical / distance / correlation 多图（correlation 严格 train-only）。
- 跑 STID 与 Graph WaveNet（hybrid）基线，评估分位数预测对报价策略的价值。

**验收**：真实数据 run 的 artifacts 完整，指标在反归一化后按 overall/h3/h6/h12 报告。

## 3. METR-LA graph 存储从 pickle 迁移到 .npz

**背景**：`adj_mx.pkl` 反序列化已通过 `_RestrictedGraphUnpickler` 限制到 NumPy allowlist，
任意代码执行面已封死； pickle 格式本身仍是残留低风险（独立 review 结论）。

**内容**：

- `prepare_data.py` 阶段将邻接矩阵转存为 `.npz`，processed 数据不再依赖 pickle。
- 保留 raw 下载校验（SHA-256）不变；loader 改为 `np.load`。
- 更新 `metr_la.yaml`、相关测试与设计文档记录。

**验收**：processed 目录不含 pickle 读取路径；全量测试通过。

---

设计依据见 `design/2026-09-04-时空预测基线工程设计.md` §16；
独立复审记录见同文件 §16.4。
