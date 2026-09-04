---
author:
- Bing Yu
- Haoteng Yin
- Zhanxing Zhu
created: 2026-04-06
created_at: 2026-04-06
description: 'STGCN 原始论文（IJCAI 2018）arXiv 摘要页剪藏：时空图卷积网络开山之作'
source_type: web
status: inbox
tags:
- clippings
- STGCN
- graph-neural-network
- traffic-forecasting
- 时空预测
title: 'Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for
  Traffic Forecasting'
source_url: https://arxiv.org/abs/1709.04875
published_at: 2017-09-14
related_concepts:
- Spatio-Temporal Graph Convolutional Network
- Graph Convolutional Network
- Traffic Forecasting
topics:
- timeseries-analysis
- 时间序列分析
---

## Title: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting

Authors: [Bing Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu,+B), [Haoteng Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin,+H), [Zhanxing Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu,+Z)

[View PDF](https://arxiv.org/pdf/1709.04875) | [IJCAI'18 正式版](https://www.ijcai.org/proceedings/2018/0505.pdf)

> Abstract: Timely accurate traffic forecast is crucial for urban traffic control and guidance. Due to the high nonlinearity and complexity of traffic flow, traditional methods cannot satisfy the requirements of mid-and-long term prediction tasks and often neglect spatial and temporal dependencies. In this paper, we propose a novel deep learning framework, Spatio-Temporal Graph Convolutional Networks (STGCN), to tackle the time series prediction problem in traffic domain. Instead of applying regular convolutional and recurrent units, we formulate the problem on graphs and build the model with complete convolutional structures, which enable much faster training speed with fewer parameters. Experiments show that our model STGCN effectively captures comprehensive spatio-temporal correlations through modeling multi-scale traffic networks and consistently outperforms state-of-the-art baselines on various real-world traffic datasets.

| Comments: | Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI 2018) |
| --- | --- |
| Subjects: | Machine Learning (cs.LG); Machine Learning (stat.ML) |
| Cite as: | [arXiv:1709.04875](https://arxiv.org/abs/1709.04875) \[cs.LG\] |
|  | (or [arXiv:1709.04875v4](https://arxiv.org/abs/1709.04875v4) \[cs.LG\] for this version) |
|  | [https://doi.org/10.48550/arXiv.1709.04875](https://doi.org/10.48550/arXiv.1709.04875) |
| Related DOI: | [https://doi.org/10.24963/ijcai.2018/505](https://doi.org/10.24963/ijcai.2018/505) |

## Submission history

From: Haoteng Yin  
**[\[v1\]](https://arxiv.org/abs/1709.04875v1)** Thu, 14 Sep 2017 16:54:41 UTC (1,050 KB)  
**[\[v2\]](https://arxiv.org/abs/1709.04875v2)** Mon, 25 Sep 2017 09:17:45 UTC (1,006 KB)  
**[\[v3\]](https://arxiv.org/abs/1709.04875v3)** Thu, 1 Feb 2018 13:52:01 UTC (560 KB)  
**\[v4\]** Thu, 12 Jul 2018 07:55:09 UTC (514 KB)

---

## 论文核心信息（2026-08 从原文补充）

### 核心贡献

1. **首个纯卷积时空建模框架**：不用常规卷积/循环单元，直接在图上建模，用全卷积结构同时从图结构时间序列中提取时空特征；
2. **ST-Conv Block 结构**：时空卷积块 = 时间门控卷积 (TCN) + 中间图卷积 (ChebNet / 一阶近似 GCN)，三明治堆叠，带残差连接和瓶颈策略；
3. **效率优势**：相比 RNN 类模型（GCGRU/DCRNN），参数更少、训练更快、支持并行；
4. **效果**：在 PeMSD7(M/L) 上 15/30/45 分钟预测全面超越 HA/LSVR/ARIMA/FNN/FC-LSTM/GCGRU 等基线。

### 关键设计

- 时空预测形式化：给定图 $G$ 上前 $M$ 个观测，预测后 $H$ 个时间步（最大化对数似然）；
- 图卷积两种实现：ChebConv（K 阶切比雪夫多项式近似）与一阶近似（K=1），论文中两者精度接近（STGCN(Cheb) 与 STGCN(1st)）；
- 数据：PeMSD7（加州高速公路 228 / 1,026 个传感器，5 分钟粒度），距离阈值高斯核构造邻接矩阵，Z-score 标准化。

### 与本项目的关联

- STGCN 是时空图预测范式的起点，后续 ASTGCN / Graph WaveNet / MTGNN / AGCRN 等均以此为基线；
- 其「纯卷积 + 图卷积交替」思路可迁移到电力负荷/新能源出力的空间-时间联合预测（如区域电网、场站群拓扑）。

### 复现入口

- 官方代码（TF 1.x）：https://github.com/VeritasYin/STGCN_IJCAI-18
- PyTorch 实现：PyG Temporal 的 `STConv` 层 https://github.com/benedekrozemberczki/pytorch_geometric_temporal

## BibTeX

```bibtex
@inproceedings{yu2018spatio,
    title={Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting},
    author={Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
    booktitle={Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI)},
    year={2018}
}
```
