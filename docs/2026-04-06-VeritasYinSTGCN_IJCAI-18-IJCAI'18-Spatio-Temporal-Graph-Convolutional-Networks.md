---
author: null
created: 2026-04-06
created_at: 2026-04-06
description: '[IJCAI''18] Spatio-Temporal Graph Convolutional Networks - VeritasYin/STGCN_IJCAI-18'
source_type: web
status: inbox
tags:
- null
- clippings
title: 'VeritasYin/STGCN_IJCAI-18: [IJCAI''18] Spatio-Temporal Graph Convolutional
  Networks'
source_url: https://github.com/VeritasYin/STGCN_IJCAI-18
published_at: null
related_concepts: []
topics:
  - timeseries-analysis
  - 时间序列分析
---

## Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting

## Introduction

We propose a novel deep learning framework, **STGCN**, to tackle time series prediction problem in traffic domain. Instead of applying regular convolutional and recurrent units, we formulate the problem on graphs and build the model with complete convolutional structures. To the best of our knowledge, it is the first time that to apply purely convolutional structures to extract spatio-temporal features simultaneously from graph-structured time series in a traffic study.

## Problem Formulation

Traffic forecast is a typical time-series prediction problem, i.e. predicting the most likely traffic measurements (e.g. speed or traffic flow) in the next $H$ time steps given the previous $M$ observations from traffic network $G$ as,

[![[raw/assets/attachments/deeplearning/Image 60.svg]]](https://camo.githubusercontent.com/8d8ba847dc34f82eec0043d7dde65fd4e4d95f1ce9496564fbd191d49369ce19/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f5c6c617267652673706163653b5c6861747b767d5f7b742b317d2c2673706163653b2e2e2e2c2673706163653b5c6861747b767d5f7b742b487d3d5c6d6174686f707b5c6172675c6d61787d5f7b765f7b742b317d2c2673706163653b2e2e2e2c2673706163653b765f7b742b487d7d2673706163653b5c6c6f672673706163653b5028765f7b742b317d2c2673706163653b2e2e2e2c765f7b742b487d7c765f7b742d4d2b317d2c2673706163653b2e2e2e2c765f743b4729)

**Fig.1 Graph-structured traffic data.**  
Each $v
    t$ indicates a frame of current traffic status at time step $t$ , which is recorded in a graph-structured data matrix.

## Network Structure

[![[raw/assets/attachments/deeplearning/STGCN.png]]](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/figures/STGCN.png)

**Fig. 2 Architecture of spatio-temporal graph convolutional networks.**  
The framework STGCN consists of two spatio-temporal convolutional blocks (ST-Conv blocks) and a fully-connected output layer in the end. Each ST-Conv block contains two temporal gated convolution layers and one spatial graph convolution layer in the middle. The residual connection and bottleneck strategy are applied inside each block. The input $v
    
      t
      −
      M
      +
      1
    
  
  ,
  .
  .
  .
  ,
  
    v
    t$ is uniformly processed by ST-Conv blocks to explore spatial and temporal dependencies coherently. Comprehensive features are integrated by an output layer to generate the final prediction $v
      ^$ .

## Results

| Model |  | PeMSD7(M) (15/30/45 min) |  |  | PeMSD7(L) (15/30/45 min) |  |
| --- | --- | --- | --- | --- | --- | --- |
| / | MAE | MAPE(%) | RMSE | MAE | MAPE(%) | RMSE |
| HA | 4.01 | 10.61 | 7.20 | 4.60 | 12.50 | 8.05 |
| LSVR | 2.50/3.63/4.54 | 5.81/8.88/11.50 | 4.55/6.67/8.28 | 2.69/3.85/4.79 | 6.27/9.48/12.42 | 4.88/7.10/8.72 |
| ARIMA | 5.55/5.86/6.27 | 12.92/13.94/15.20 | 9.00/9.13/9.38 | 5.50/5.87/6.30 | 12.30/13.54/14.85 | 8.63/8.96/9.39 |
| FNN | 2.74/4.02/5.04 | 6.38/9.72/12.38 | 4.75/6.98/8.58 | 2.74/3.92/4.78 | 7.11/10.89/13.56 | 4.87/7.02/8.46 |
| FC-LSTM | 3.57/3.94/4.16 | 8.60/9.55/10.10 | 6.20/7.03/7.51 | 4.38/4.51/4.66 | 11.10/11.41/11.69 | 7.68/7.94/8.20 |
| GCGRU | 2.37/3.31/4.01 | 5.54/8.06/9.99 | 4.21/5.96/7.13 | 2.48/3.43/4.12∗ | 5.76/8.45/10.51∗ | 4.40/6.25/7.49∗ |
| **STGCN(Cheb)** | **2.25/3.03/3.57** | 5.26/ **7.33/8.69** | **4.04/5.70/6.77** | **2.37/3.27/3.97** | **5.56/7.98/9.73** | **4.32/6.21/7.45** |
| **STGCN(1st)** | 2.26/3.09/3.79 | **5.24** /7.39/9.12 | 4.07/5.77/7.03 | 2.40/3.31/4.01 | 5.63/8.21/10.12 | 4.38/6.43/7.81 |

**Table 1: Performance comparison of different approaches on the dataset PeMSD7.**

**Fig. 3: Speed prediction in the morning peak and evening rush hours of the dataset PeMSD7.**

**Fig. 4: Time consumptions of training on the dataset PeMSD7 (M, left) and (L, right)**

## Requirements

Our code is based on Python3 (>= 3.6). There are a few dependencies to run the code. The major libraries are listed as follows:

- TensorFlow (>= 1.9.0)
- NumPy (>= 1.15)
- SciPy (>= 1.1.0)
- Pandas (>= 0.23)

The implementation of Spatio-Temporal Graph Convolutional Layer with PyTorch is available in [PyG Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/attention/stgcn.py). You might refer to [STConv](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#temporal-graph-attention-layers) that supports ChebConv Graph Convolutions.

## Dataset

### Data Source

**[PeMSD7](http://pems.dot.ca.gov/)** was collected from Caltrans Performance Measurement System (PeMS) in real-time by over 39, 000 sensor stations, deployed across the major metropolitan areas of California state highway system. The dataset is also aggregated into 5-minute interval from 30-second data samples. We randomly select a medium and a large scale among the District 7 of California containing **228** and **1, 026** stations, labeled as PeMSD7(M) and PeMSD7(L), respectively, as data sources. The time range of PeMSD7 dataset is in the weekdays of **May and June of 2012**. We select the first month of historical speed records as training set, and the rest serves as validation and test set respectively.

Dataset PeMSD7(M/L) is now available under `dataset` folder (station list included). Please refer [issue #6](https://github.com/VeritasYin/STGCN_IJCAI-18/issues/6) for how to download metadata from PeMS.

### Data Format

You can make your customized dataset by the following format:

- PeMSD7\_V\_{ `$num_route` }.csv: Historical Speed Records with shape of \[len\_seq \* num\_road\] (len\_seq = day\_slot \* num\_dates).
- PeMSD7\_W\_{ `$num_route` }.csv: Weighted Adjacency Matrix with shape of \[num\_road \* num\_road\].

Note: please replace the `$num_route` with the number of routes in your dataset. '\*.csv' should not contain any index or header in the file.

### Data Preprocessing

The standard time interval is set to 5 minutes. Thus, every node of the road graph contains **288** data points per day (day\_slot = 288). The linear interpolation method is used to fill missing values after data cleaning. In addition, data input are normalized by Z-Score method.  
In PeMSD7, the adjacency matrix of the road graph is computed based on the distances among stations in the traffic network. The weighted adjacency matrix W can be formed as,

All of our experiments use 60 minutes as the historical time window, a.k.a. 12 observed data points (M = 12) are used to forecast traffic conditions in the next 15, 30, and 45 minutes (H = 3, 6, 9).

## Model Details

### Training

python main.py --n\_route { `$num_route` } --graph { `$weight_matrix_file` }

**Default settings**:

- Training configs: argparse is used for passing parameters.
	- n\_route=228, graph='default', ks=3, kt=3, n\_his=12, n\_pred=9
		- batch\_size=50, epoch=50, lr=0.001, opt='RMSProp', inf\_mode='merge', save=10
- Data source will be searched in dataset\_dir = './dataset', including speed records and the weight matrix.
- Trained models will be saved in save\_path = './output/models' every `args.save=10` epochs.
- Training logs will be saved in sum\_path = './output/tensorboard'.

Note: it normally takes around 6s on a NVIDIA TITAN Xp for one epoch with the batch size of 50 and n\_route of 228.

### Folder structure

```
├── data_loader
│   ├── data_utils.py
│   └── __init__.py
├── dataset
│   ├── PeMSD7_V_228.csv
│   ├── PeMSD7_W_228.csv
│   ├── PeMSD7_V_1026.csv
│   └── PeMSD7_W_1026.csv
├── main.py
├── models
│   ├── base_model.py
│   ├── __init__.py
│   ├── layers.py
│   ├── tester.py
│   └── trainer.py
├── output
│   ├── models
│   └── tensorboard
├── README.md
└── utils
    ├── __init__.py
    ├── math_graph.py
    └── math_utils.py
```

## Updates

**Feb. 22, 2022**:

- Sensor Station List of PeMSD7-M released.

**Feb. 11, 2022**:

- Dataset PeMSD7-L (1,026 nodes) released.
- Fix the issue in size calculation of temporal channel. Thanks to @KingWang93 and @cheershuaizhao.

**Apr. 18, 2019**:

- Dataset PeMSD7-M (228 nodes) released.

**Jan. 14, 2019**:

- Code refactoring based on the [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template), following the PEP 8 code style;
- Function model\_save(), model\_test() and tensorboard support are added;
- The process of model training and inference is optimized;
- Corresponding code comments are updated.

## Citation

Please refer to our paper. Bing Yu\*, Haoteng Yin\*, Zhanxing Zhu. [Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://www.ijcai.org/proceedings/2018/0505). In *Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI)*, 2018

```
@inproceedings{yu2018spatio,
    title={Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting},
    author={Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
    booktitle={Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI)},
    year={2018}
}
```